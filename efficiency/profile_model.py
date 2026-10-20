import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_model(model_key, checkpoint_path=None):
    """
    Factory to instantiate models given a model_key:
    'res2net' -> Network from lib.Network_Res2Net_GRA_NCD
    'ghostnet' -> Network from lib.Network_GhostNet_GRA_NCD (Step 2)
    'shufflenet' -> Network from lib.Network_ShuffleNetV2_GRA_NCD (Step 2)
    """
    key = model_key.lower().replace("-", "_").replace(" ", "_")
    
    if 'res2net' in key:
        from lib.Network_Res2Net_GRA_NCD import Network
        model = Network(imagenet_pretrained=False)
        backbone_attr = 'resnet'
    elif 'ghostnet' in key:
        from lib.Network_GhostNet_GRA_NCD import Network
        model = Network(imagenet_pretrained=False)
        backbone_attr = 'backbone'
    elif 'shufflenet' in key:
        from lib.Network_ShuffleNetV2_GRA_NCD import Network
        model = Network(imagenet_pretrained=False)
        backbone_attr = 'backbone'
    else:
        raise ValueError(f"Unknown model_key: {model_key}. Supported: res2net, ghostnet, shufflenet")

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"[+] Loaded weights from {checkpoint_path}")
    elif checkpoint_path:
        print(f"[-] Checkpoint {checkpoint_path} not found. Using randomly initialized weights for profiling.")
        
    return model, backbone_attr


def count_parameters(model, backbone_attr):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    backbone = getattr(model, backbone_attr)
    backbone_params = sum(p.numel() for p in backbone.parameters())
    decoder_params = total_params - backbone_params
    
    return {
        'total_params_M': round(total_params / 1e6, 4),
        'trainable_params_M': round(trainable_params / 1e6, 4),
        'backbone_params_M': round(backbone_params / 1e6, 4),
        'decoder_params_M': round(decoder_params / 1e6, 4),
        'raw_total': total_params,
        'raw_backbone': backbone_params,
        'raw_decoder': decoder_params
    }


def compute_macs_and_flops(model, input_shape=(1, 3, 352, 352)):
    import copy
    dummy_input = torch.randn(*input_shape, device='cpu')
    
    # 1. thop on an isolated copy to prevent hook pollution
    macs_thop = None
    try:
        import thop
        model_thop = copy.deepcopy(model).to('cpu').eval()
        macs_thop, _ = thop.profile(model_thop, inputs=(dummy_input,), verbose=False)
        del model_thop
    except Exception as e:
        print(f"[-] thop profiling failed: {e}")
        
    # 2. Cross-check with ptflops or fvcore on an isolated copy
    macs_cross_check = None
    cross_check_tool = None
    try:
        from ptflops import get_model_complexity_info
        model_ptflops = copy.deepcopy(model).to('cpu').eval()
        macs_str, _ = get_model_complexity_info(
            model_ptflops, (3, input_shape[2], input_shape[3]), 
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
        macs_cross_check = macs_str
        cross_check_tool = "ptflops"
        del model_ptflops
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            model_fvcore = copy.deepcopy(model).to('cpu').eval()
            flops_fvcore = FlopCountAnalysis(model_fvcore, dummy_input).total()
            macs_cross_check = flops_fvcore / 2.0
            cross_check_tool = "fvcore"
            del model_fvcore
        except Exception:
            pass

    primary_macs = macs_thop if macs_thop is not None else macs_cross_check
    if primary_macs is None:
        raise RuntimeError("Both thop and ptflops/fvcore failed to compute MACs.")
        
    macs_G = round(primary_macs / 1e9, 4)
    flops_G = round((primary_macs * 2) / 1e9, 4)
    
    result = {
        'macs_G': macs_G,
        'flops_G': flops_G,
        'convention': "FLOPs = 2 x MACs (1 Multiply-Accumulate = 2 Floating Point Operations)",
        'thop_macs_G': round(macs_thop / 1e9, 4) if macs_thop is not None else None,
        'cross_check_tool': cross_check_tool,
        'cross_check_macs_G': round(macs_cross_check / 1e9, 4) if macs_cross_check is not None else None
    }
    return result


def measure_gpu_latency(model, input_shape=(1, 3, 352, 352), dtype=torch.float32, 
                        warmup_iters=50, timed_iters=200):
    if not torch.cuda.is_available():
        return None
        
    device = torch.device('cuda')
    model_cuda = model.to(device).to(dtype=dtype).eval()
    dummy_input = torch.randn(*input_shape, device=device, dtype=dtype)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model_cuda(dummy_input)
    torch.cuda.synchronize()
    
    latencies = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for _ in range(timed_iters):
            starter.record()
            _ = model_cuda(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender))  # milliseconds
            
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    
    return {
        'mean_ms': round(mean_lat, 3),
        'std_ms': round(std_lat, 3),
        'p95_ms': round(p95_lat, 3),
        'fps': round(fps, 2),
        'iterations': timed_iters,
        'warmup_iterations': warmup_iters
    }


def measure_peak_memory(model, input_shape=(1, 3, 352, 352)):
    if not torch.cuda.is_available():
        return None
        
    device = torch.device('cuda')
    model_cuda = model.to(device).float().eval()
    dummy_input = torch.randn(*input_shape, device=device, dtype=torch.float32)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model_cuda(dummy_input)
    torch.cuda.synchronize()
    
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = round(peak_bytes / (1024 * 1024), 2)
    return peak_mb


def measure_cpu_latency(model, input_shape=(1, 3, 352, 352), 
                        warmup_iters=10, timed_iters=50):
    device = torch.device('cpu')
    model_cpu = model.to(device).float().eval()
    dummy_input = torch.randn(*input_shape, device=device, dtype=torch.float32)
    thread_count = torch.get_num_threads()
    
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model_cpu(dummy_input)
            
    latencies = []
    with torch.no_grad():
        for _ in range(timed_iters):
            t0 = time.perf_counter()
            _ = model_cpu(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
            
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    
    return {
        'mean_ms': round(mean_lat, 3),
        'std_ms': round(std_lat, 3),
        'p95_ms': round(p95_lat, 3),
        'fps': round(fps, 2),
        'threads': thread_count,
        'iterations': timed_iters,
        'warmup_iterations': warmup_iters
    }


def profile_model(model_key, checkpoint_path=None, output_dir="efficiency/results", save_name=None):
    os.makedirs(output_dir, exist_ok=True)
    if save_name is None:
        save_name = model_key.lower().replace("-", "_")
    output_json = os.path.join(output_dir, f"{save_name}.json")
    
    print(f"\n{'='*70}")
    print(f" Profiling Model: {model_key}")
    print(f" Checkpoint    : {checkpoint_path}")
    print(f" Output Target : {output_json}")
    print(f"{'='*70}")
    
    # 1. Instantiate model
    model, backbone_attr = get_model(model_key, checkpoint_path)
    
    # 2. Checkpoint Size
    ckpt_size_mb = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_size_mb = round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2)
    else:
        # Measure in-memory state_dict size
        import io
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        ckpt_size_mb = round(len(buf.getvalue()) / (1024 * 1024), 2)
        
    # 3. Parameters breakdown
    param_info = count_parameters(model, backbone_attr)
    print(f"[+] Total Parameters   : {param_info['total_params_M']} M")
    print(f"    - Backbone ({backbone_attr}): {param_info['backbone_params_M']} M")
    print(f"    - Decoder (RFB/NCD): {param_info['decoder_params_M']} M")
    print(f"[+] Checkpoint Size    : {ckpt_size_mb} MB")
    
    # 4. MACs & FLOPs
    print("[*] Calculating MACs and FLOPs...")
    flops_info = compute_macs_and_flops(model)
    print(f"[+] MACs               : {flops_info['macs_G']} G (thop: {flops_info['thop_macs_G']} G, cross-check {flops_info['cross_check_tool']}: {flops_info['cross_check_macs_G']} G)")
    print(f"[+] FLOPs (2x MACs)    : {flops_info['flops_G']} G")
    
    # 5. Peak CUDA memory
    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = measure_peak_memory(model)
        print(f"[+] Peak CUDA VRAM     : {peak_vram_mb} MB")
    else:
        print("[-] CUDA not available for VRAM measurement.")
        
    # 6. GPU Latency (FP32 & FP16)
    gpu_fp32 = None
    gpu_fp16 = None
    if torch.cuda.is_available():
        print("[*] Benchmarking GPU Latency (FP32: 50 warmup, 200 iters)...")
        gpu_fp32 = measure_gpu_latency(model, dtype=torch.float32)
        print(f"    -> GPU FP32: {gpu_fp32['mean_ms']} ms ± {gpu_fp32['std_ms']} ms (p95: {gpu_fp32['p95_ms']} ms) | {gpu_fp32['fps']} FPS")
        
        print("[*] Benchmarking GPU Latency (FP16: 50 warmup, 200 iters)...")
        gpu_fp16 = measure_gpu_latency(model, dtype=torch.float16)
        print(f"    -> GPU FP16: {gpu_fp16['mean_ms']} ms ± {gpu_fp16['std_ms']} ms (p95: {gpu_fp16['p95_ms']} ms) | {gpu_fp16['fps']} FPS")
        
    # 7. CPU Latency
    print("[*] Benchmarking CPU Latency (10 warmup, 50 iters)...")
    cpu_info = measure_cpu_latency(model)
    print(f"    -> CPU ({cpu_info['threads']} threads): {cpu_info['mean_ms']} ms ± {cpu_info['std_ms']} ms (p95: {cpu_info['p95_ms']} ms) | {cpu_info['fps']} FPS")
    
    # Compile results
    results = {
        'model_key': model_key,
        'checkpoint_path': checkpoint_path,
        'checkpoint_size_MB': ckpt_size_mb,
        'parameters': param_info,
        'computation': flops_info,
        'peak_cuda_memory_MB': peak_vram_mb,
        'latency_gpu_fp32': gpu_fp32,
        'latency_gpu_fp16': gpu_fp16,
        'latency_cpu': cpu_info,
        'device_info': {
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'cpu_threads': cpu_info['threads']
        }
    }
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Profile written to {output_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile model parameters, FLOPs/MACs, latency, and memory.")
    parser.add_argument('--model', type=str, default='res2net', choices=['res2net', 'ghostnet', 'shufflenet'],
                        help='Model backbone architecture')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--out_dir', type=str, default='efficiency/results',
                        help='Output directory for profiling JSONs')
    parser.add_argument('--save_name', type=str, default=None,
                        help='File name prefix (defaults to model key)')
    args = parser.parse_args()
    
    profile_model(args.model, args.checkpoint, args.out_dir, args.save_name)


if __name__ == '__main__':
    main()
