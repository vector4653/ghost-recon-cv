from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
import sys
from pathlib import Path
import numpy as np
import cv2

app = FastAPI(title="Ghost-Recon-CV Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SIZE = 352

# Table 6 reference metrics — used both for telemetry and to scale synthetic mask fidelity
MODEL_METRICS = {
    "Baseline": {"sAlpha": 0.7647, "ePhi": 0.8061, "fBetaW": 0.6036, "mae": 0.0593, "boundaryIoU": 0.2332, "fpr": 1.41},
    "Exp M":    {"sAlpha": 0.8638, "ePhi": 0.9046, "fBetaW": 0.7718, "mae": 0.0348, "boundaryIoU": 0.3372, "fpr": 2.02},
    "Exp C1":   {"sAlpha": 0.7572, "ePhi": 0.8045, "fBetaW": 0.5886, "mae": 0.0538, "boundaryIoU": 0.2225, "fpr": 1.58},
    "Exp C2":   {"sAlpha": 0.6223, "ePhi": 0.6616, "fBetaW": 0.3541, "mae": 0.1007, "boundaryIoU": 0.1414, "fpr": 0.81},
    "Exp MC1":  {"sAlpha": 0.8675, "ePhi": 0.9095, "fBetaW": 0.7773, "mae": 0.0322, "boundaryIoU": 0.3437, "fpr": 1.55},
    "Exp MC2":  {"sAlpha": 0.8460, "ePhi": 0.8765, "fBetaW": 0.7276, "mae": 0.0423, "boundaryIoU": 0.3119, "fpr": 2.19},
    "Exp MCC":  {"sAlpha": 0.8518, "ePhi": 0.8802, "fBetaW": 0.7379, "mae": 0.0408, "boundaryIoU": 0.3199, "fpr": 1.99},
}

# Checkpoint directory naming follows run_experiment.py: ./snapshot/Exp_{M,C1,C2,MC1,MC2,MCC}/,
# with the off-the-shelf baseline staged separately at ./snapshot/SINet_V2/
CHECKPOINT_DIRS = {
    "Baseline": "SINet_V2",
    "Exp M": "Exp_M",
    "Exp C1": "Exp_C1",
    "Exp C2": "Exp_C2",
    "Exp MC1": "Exp_MC1",
    "Exp MC2": "Exp_MC2",
    "Exp MCC": "Exp_MCC",
}

# ---------------------------------------------------------------------------
# Optional PyTorch backend. If torch, the SINet-V2 module, or a checkpoint is
# unavailable, we transparently fall back to a synthetic mask generator so the
# service never crashes and the rest of the platform stays fully demoable.
# ---------------------------------------------------------------------------
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    sys.path.insert(0, str(PROJECT_ROOT))
    from lib.Network_Res2Net_GRA_NCD import Network

    TORCH_AVAILABLE = True
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    PREPROCESS = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((TEST_SIZE, TEST_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
except ImportError:
    TORCH_AVAILABLE = False

_model_cache = {}


def get_model(model_key: str):
    """Lazily load and cache the PyTorch checkpoint for a given experiment. Returns None on any failure."""
    if not TORCH_AVAILABLE:
        return None
    if model_key in _model_cache:
        return _model_cache[model_key]

    ckpt_dir = CHECKPOINT_DIRS.get(model_key, CHECKPOINT_DIRS["Exp MC1"])
    pth_path = PROJECT_ROOT / "snapshot" / ckpt_dir / "Net_epoch_best.pth"
    if not pth_path.exists():
        _model_cache[model_key] = None
        return None

    try:
        net = Network(imagenet_pretrained=False)
        net.load_state_dict(torch.load(str(pth_path), map_location="cpu"))
        net.to(DEVICE)
        net.eval()
        _model_cache[model_key] = net
        return net
    except Exception as e:
        print(f"[ai_service] Failed to load checkpoint for {model_key}: {e}")
        _model_cache[model_key] = None
        return None


def run_pytorch_inference(net, img_bgr: np.ndarray) -> np.ndarray:
    """Runs real SINet-V2 inference. Returns a uint8 grayscale mask at original resolution."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = PREPROCESS(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, _, _, s3_pred = net(tensor)
        res = F.interpolate(s3_pred, size=(h, w), mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()

    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return (res * 255).astype(np.uint8)


def run_synthetic_inference(img_bgr: np.ndarray, model_key: str) -> np.ndarray:
    """Deterministic-ish synthetic mask used when no trained checkpoint is available.
    Mask fidelity is scaled by the model's real Table 6 weighted F-measure so weaker
    training regimes visibly produce noisier, less precise masks."""
    h, w = img_bgr.shape[:2]
    metrics = MODEL_METRICS.get(model_key, MODEL_METRICS["Exp MC1"])

    mask = np.zeros((h, w), dtype=np.uint8)
    center_y, center_x = h // 2, w // 2
    cv2.circle(mask, (center_x, center_y), min(h, w) // 4, 255, -1)

    noise_strength = 1.0 - metrics["fBetaW"]
    if noise_strength > 0.05:
        noise = (np.random.rand(h, w) < noise_strength * 0.4).astype(np.uint8) * 255
        mask = cv2.max(mask, cv2.GaussianBlur(noise, (0, 0), sigmaX=3))

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5)
    return mask


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "SINet-V2 Res2Net-50",
        "pytorchAvailable": TORCH_AVAILABLE,
        "device": str(DEVICE) if TORCH_AVAILABLE else None,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...), model: str = Form("Exp MC1")):
    model_key = model if model in MODEL_METRICS else "Exp MC1"
    metrics = MODEL_METRICS[model_key]

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode uploaded image"}

    h, w = img.shape[:2]
    net = get_model(model_key)
    engine = "pytorch"
    if net is not None:
        try:
            mask = run_pytorch_inference(net, img)
        except Exception as e:
            print(f"[ai_service] Inference failed for {model_key}, falling back to synthetic: {e}")
            mask = run_synthetic_inference(img, model_key)
            engine = "synthetic"
    else:
        mask = run_synthetic_inference(img, model_key)
        engine = "synthetic"

    _, buffer = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(buffer).decode("utf-8")

    camouflage_area = float(np.sum(mask > 128) / (h * w) * 100)
    confidence = round(((metrics["sAlpha"] + metrics["ePhi"]) / 2) * 100, 1)

    if camouflage_area > 15:
        threat_level = "CRITICAL"
    elif camouflage_area > 5:
        threat_level = "ELEVATED"
    else:
        threat_level = "LOW"

    return {
        "maskBase64": mask_b64,
        "telemetry": {
            "camouflageAreaPercent": round(camouflage_area, 2),
            "confidenceScore": confidence,
            "threatLevel": threat_level,
            "boundaryIoU": metrics["boundaryIoU"],
            "fpr": metrics["fpr"],
            "sAlpha": metrics["sAlpha"],
            "ePhi": metrics["ePhi"],
            "fBetaW": metrics["fBetaW"],
            "mae": metrics["mae"],
            "inferenceEngine": engine,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
