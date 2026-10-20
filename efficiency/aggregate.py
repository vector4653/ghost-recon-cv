import os
import sys
import json
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_profile_and_metrics(model_key, results_dir="efficiency/results"):
    json_path = os.path.join(results_dir, f"{model_key}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing profile JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def generate_efficiency_table(results_dir="efficiency/results"):
    # Target models
    models = [
        ('res2net', 'SINet-V2 Res2Net-50 (MC1)'),
        ('ghostnet', 'GhostNet-SINet (MC1)'),
        ('shufflenet', 'ShuffleNetV2-SINet (MC1)')
    ]
    
    rows = []
    base_res = None
    
    for key, display_name in models:
        data = load_profile_and_metrics(key, results_dir)
        params = data['parameters']['total_params_M']
        backbone_p = data['parameters']['backbone_params_M']
        decoder_p = data['parameters']['decoder_params_M']
        macs = data['computation']['macs_G']
        flops = data['computation']['flops_G']
        size_mb = data['checkpoint_size_MB']
        
        gpu_fp32 = data['latency_gpu_fp32']['mean_ms'] if data['latency_gpu_fp32'] else None
        gpu_fp16 = data['latency_gpu_fp16']['mean_ms'] if data['latency_gpu_fp16'] else None
        fps = data['latency_gpu_fp32']['fps'] if data['latency_gpu_fp32'] else None
        cpu_lat = data['latency_cpu']['mean_ms'] if data['latency_cpu'] else None
        peak_vram = data['peak_cuda_memory_MB']
        
        # Eval metrics
        eval_m = data.get('evaluation', {})
        sm = eval_m.get('sm', None)
        em = eval_m.get('em', None)
        wfm = eval_m.get('wfm', None)
        mae = eval_m.get('mae', None)
        b_iou = eval_m.get('boundary_iou_4px', None)
        fpr = eval_m.get('fpr_background', None)
        
        row_dict = {
            'model_key': key,
            'model_name': display_name,
            'params_M': params,
            'backbone_params_M': backbone_p,
            'decoder_params_M': decoder_p,
            'macs_G': macs,
            'flops_G': flops,
            'size_MB': size_mb,
            'gpu_lat_fp32_ms': gpu_fp32,
            'gpu_lat_fp16_ms': gpu_fp16,
            'fps': fps,
            'cpu_lat_ms': cpu_lat,
            'peak_vram_MB': peak_vram,
            'sm': sm,
            'em': em,
            'wfm': wfm,
            'mae': mae,
            'b_iou': b_iou,
            'fpr': fpr
        }
        
        if key == 'res2net':
            base_res = row_dict
            
        rows.append(row_dict)
        
    # Compute derived metrics
    base_params = base_res['params_M']
    base_macs = base_res['macs_G']
    base_gpu_lat = base_res['gpu_lat_fp32_ms']
    base_wfm = base_res['wfm'] if base_res['wfm'] is not None else 0.7773
    
    for r in rows:
        r['param_reduction_x'] = round(base_params / r['params_M'], 2) if r['params_M'] > 0 else 0
        r['macs_reduction_x'] = round(base_macs / r['macs_G'], 2) if r['macs_G'] > 0 else 0
        r['speedup_x'] = round(base_gpu_lat / r['gpu_lat_fp32_ms'], 2) if (base_gpu_lat and r['gpu_lat_fp32_ms']) else 0
        
        cur_wfm = r['wfm'] if r['wfm'] is not None else 0.0
        r['pct_wfm_retained'] = round((cur_wfm / base_wfm) * 100, 2) if base_wfm > 0 else 0
        r['wfm_per_gmac'] = round(cur_wfm / r['macs_G'], 3) if r['macs_G'] > 0 else 0
        cur_sm = r['sm'] if r['sm'] is not None else 0.0
        r['sm_per_m_param'] = round(cur_sm / r['params_M'], 3) if r['params_M'] > 0 else 0
        
        # Lightweight criterion: >= 4x fewer params AND MACs with >= 95% F-beta-w retained
        is_lw = (r['param_reduction_x'] >= 4.0 and r['macs_reduction_x'] >= 4.0 and r['pct_wfm_retained'] >= 95.0)
        r['is_lightweight'] = "YES (Certified)" if is_lw else ("Baseline Reference" if r['model_key'] == 'res2net' else "NO")

    # Save CSV
    csv_path = os.path.join(results_dir, "efficiency_table.csv")
    fieldnames = [
        'Model', 'Params (M)', 'MACs (G)', 'FLOPs (G)', 'Size (MB)',
        'GPU FP32 (ms)', 'GPU FP16 (ms)', 'FPS', 'CPU Latency (ms)', 'Peak VRAM (MB)',
        'S_alpha', 'E_phi', 'F_beta_w', 'MAE', 'Boundary IoU (4px)', 'FPR (%)',
        'Param Reduction (x)', 'MACs Reduction (x)', 'Speedup (x)',
        '% F_beta_w Retained', 'F_w / GMAC', 'S_alpha / M Params', 'Lightweight Criterion'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in rows:
            writer.writerow([
                r['model_name'], r['params_M'], r['macs_G'], r['flops_G'], r['size_MB'],
                r['gpu_lat_fp32_ms'], r['gpu_lat_fp16_ms'], r['fps'], r['cpu_lat_ms'], r['peak_vram_MB'],
                r['sm'], r['em'], r['wfm'], r['mae'], r['b_iou'],
                round(r['fpr'] * 100, 2) if r['fpr'] is not None else None,
                f"{r['param_reduction_x']}x", f"{r['macs_reduction_x']}x", f"{r['speedup_x']}x",
                f"{r['pct_wfm_retained']}%", r['wfm_per_gmac'], r['sm_per_m_param'], r['is_lightweight']
            ])
            
    print(f"[+] Saved efficiency CSV to {csv_path}")
    
    # Save Markdown
    md_path = os.path.join(results_dir, "efficiency_table.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Tactical Camouflage Edge Efficiency & Lightweight Backbone Analysis\n\n")
        f.write("| Model | Params (M) | MACs (G) | FLOPs (G) | Size (MB) | GPU FP32 | GPU FP16 | FPS | CPU (ms) | S_α | E_φ | F_β^w | MAE | B-IoU | FPR | Param Red. | MACs Red. | % F_β^w Retained | Lightweight? |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in rows:
            fpr_str = f"{r['fpr']*100:.2f}%" if r['fpr'] is not None else "N/A"
            f.write(f"| **{r['model_name']}** | {r['params_M']:.2f} | {r['macs_G']:.2f} | {r['flops_G']:.2f} | {r['size_MB']:.1f} | "
                    f"{r['gpu_lat_fp32_ms']:.2f}ms | {r['gpu_lat_fp16_ms']:.2f}ms | {r['fps']:.1f} | {r['cpu_lat_ms']:.1f}ms | "
                    f"{r['sm']:.4f} | {r['em']:.4f} | {r['wfm']:.4f} | {r['mae']:.4f} | {r['b_iou']:.4f} | {fpr_str} | "
                    f"**{r['param_reduction_x']}×** | **{r['macs_reduction_x']}×** | **{r['pct_wfm_retained']}%** | **{r['is_lightweight']}** |\n")
        f.write("\n\n> 🔍 **Lightweight Criterion Definition:** A model is classified as **Lightweight** if and only if it achieves $\\ge 4.0\\times$ reduction in parameters AND $\\ge 4.0\\times$ reduction in MACs compared to the Res2Net-50 baseline while retaining $\\ge 95.0\\%$ of weighted F-measure ($F_\\beta^w$).\n")

    print(f"[+] Saved efficiency Markdown table to {md_path}")
    return rows


def generate_figures(rows, figures_dir="efficiency/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Pareto Plot: F_beta_w vs MACs (G) with bubble size = params
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for r in rows:
        x = r['macs_G']
        y = r['wfm'] if r['wfm'] is not None else 0.77
        size = r['params_M'] * 25
        color = '#1f77b4' if 'Res2Net' in r['model_name'] else ('#2ca02c' if 'GhostNet' in r['model_name'] else '#ff7f0e')
        scatter = ax.scatter(x, y, s=size, color=color, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(f"{r['model_name'].split()[0]}\n({r['params_M']:.1f}M params)", (x, y),
                    textcoords="offset points", xytext=(12, 5), ha='left', fontsize=10, fontweight='bold')
                    
    ax.set_xlabel("Computational Cost: MACs (GFLOPs / 2) ↓", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"Detection Accuracy: $F_\beta^w$ ↑", fontsize=12, fontweight='bold')
    ax.set_title("Pareto Efficiency: Boundary Quality vs. Computational Cost", fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    pareto_path = os.path.join(figures_dir, "pareto_fbeta_vs_macs.png")
    plt.savefig(pareto_path)
    plt.close()
    print(f"[+] Generated figure: {pareto_path}")
    
    # 2. Params and MACs comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    labels = [r['model_name'].split()[0] for r in rows]
    params_vals = [r['params_M'] for r in rows]
    macs_vals = [r['macs_G'] for r in rows]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    b1 = ax1.bar(labels, params_vals, color=colors, edgecolor='black', width=0.55)
    ax1.set_ylabel("Parameters (Million) ↓", fontsize=11, fontweight='bold')
    ax1.set_title("Model Parameter Footprint", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.2f}M", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    b2 = ax2.bar(labels, macs_vals, color=colors, edgecolor='black', width=0.55)
    ax2.set_ylabel("MACs (GigaOps) ↓", fontsize=11, fontweight='bold')
    ax2.set_title("Compute Complexity (MACs)", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{h:.2f}G", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    pm_path = os.path.join(figures_dir, "params_and_macs_comparison.png")
    plt.savefig(pm_path)
    plt.close()
    print(f"[+] Generated figure: {pm_path}")
    
    # 3. Latency comparison bar chart (GPU FP32 vs FP16 vs CPU)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(labels))
    width = 0.28
    
    gpu_fp32_vals = [r['gpu_lat_fp32_ms'] or 0 for r in rows]
    gpu_fp16_vals = [r['gpu_lat_fp16_ms'] or 0 for r in rows]
    
    r1 = ax.bar(x - width/2, gpu_fp32_vals, width, label='GPU FP32 (RTX 4050)', color='#3366cc', edgecolor='black')
    r2 = ax.bar(x + width/2, gpu_fp16_vals, width, label='GPU FP16 (Half Precision)', color='#109618', edgecolor='black')
    
    ax.set_ylabel("Inference Latency (ms/frame) ↓", fontsize=12, fontweight='bold')
    ax.set_title("Edge Inference Latency on NVIDIA RTX 4050", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in list(r1) + list(r2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}ms", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    lat_path = os.path.join(figures_dir, "latency_comparison.png")
    plt.savefig(lat_path)
    plt.close()
    print(f"[+] Generated figure: {lat_path}")

    # 4. Backbone vs Decoder parameter breakdown
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    backbone_vals = [r['backbone_params_M'] for r in rows]
    decoder_vals = [r['decoder_params_M'] for r in rows]
    
    ax.bar(labels, backbone_vals, label='Backbone (Feature Extractor)', color='#4285f4', edgecolor='black', width=0.55)
    ax.bar(labels, decoder_vals, bottom=backbone_vals, label='Decoder (RFB + NCD + GRA)', color='#f4b400', edgecolor='black', width=0.55)
    
    ax.set_ylabel("Parameters (Million)", fontsize=11, fontweight='bold')
    ax.set_title("Architectural Parameter Decomposition (Backbone vs. Decoder)", fontsize=13, fontweight='bold', pad=12)
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    for i, r in enumerate(rows):
        tot = r['params_M']
        bb = r['backbone_params_M']
        dec = r['decoder_params_M']
        ax.text(i, tot + 0.4, f"{tot:.1f}M\n(BB: {bb:.1f}M | Dec: {dec:.1f}M)", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    decomp_path = os.path.join(figures_dir, "backbone_vs_decoder_breakdown.png")
    plt.savefig(decomp_path)
    plt.close()
    print(f"[+] Generated figure: {decomp_path}")


def generate_qualitative_grid(figures_dir="efficiency/figures"):
    # Same 6 scenarios as qualitative_analysis/visual_comparison_grid.png
    curated_samples = [
        ("image974", "Concealed Long-Range Sniper"),
        ("dataset01_09_00014463", "Woodland Brush Operator"),
        ("dataset18_11_00016470", "Prone Position in Foliage"),
        ("image1005", "Multi-Terrain Tactical Operator"),
        ("image721", "Urban/Woodland Camouflage"),
        ("image697", "Close-Range Tactical Stance")
    ]
    
    test_img_dir = "./Dataset/TestDataset/Images"
    test_gt_dir = "./Dataset/TestDataset/GT"
    
    grid_cols = [
        ('Original RGB', 'RGB'),
        ('Ground Truth', 'GT'),
        ('Res2Net-50 (MC1)', './res/Exp_MC1'),
        ('GhostNet (MC1)', './res/Exp_MC1_ghostnet'),
        ('ShuffleNetV2 (MC1)', './res/Exp_MC1_shufflenet')
    ]
    
    # Check if prediction dirs exist
    for title, path in grid_cols[2:]:
        if not os.path.exists(path):
            print(f"[-] Prediction directory {path} not ready yet. Skipping qualitative grid generation until evaluation completes.")
            return

    n_rows = len(curated_samples)
    n_cols = len(grid_cols)
    
    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.0), dpi=300)
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.04, hspace=0.12)
    
    for row_idx, (stem, label) in enumerate(curated_samples):
        # Image
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(test_img_dir, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        # GT
        gt_path = os.path.join(test_gt_dir, stem + ".png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(test_gt_dir, stem + ".jpg")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        for col_idx, (col_title, col_source) in enumerate(grid_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if col_source == 'RGB':
                ax.imshow(rgb)
            elif col_source == 'GT':
                ax.imshow(gt, cmap='gray')
            else:
                pred_path = os.path.join(col_source, stem + ".png")
                if not os.path.exists(pred_path):
                    pred_path = os.path.join(col_source, stem + ".jpg")
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                if pred is not None:
                    if pred.shape != (h, w):
                        pred = cv2.resize(pred, (w, h))
                    ax.imshow(pred, cmap='gray')
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                    
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_title, fontsize=11, fontweight='bold', pad=8)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, fontweight='bold', rotation=90, labelpad=8)
                
    plt.suptitle("Tactical Camouflaged Soldier Detection: Lightweight Backbones vs. Baseline",
                 fontsize=15, fontweight='bold', y=0.995)
    grid_path = os.path.join(figures_dir, "lightweight_visual_comparison_grid.png")
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved visual comparison grid to {grid_path}")


def main():
    rows = generate_efficiency_table()
    generate_figures(rows)
    generate_qualitative_grid()

if __name__ == '__main__':
    main()
