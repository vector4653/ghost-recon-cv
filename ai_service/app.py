from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
import sys
import time
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

# Decision threshold on the raw sigmoid map. Every per-image metric and the drawn
# overlay are derived from this same cut so the panel and the picture agree.
DETECTION_THRESHOLD = 0.5
# A contour smaller than this share of the frame is speckle, not a detection.
MIN_REGION_FRACTION = 0.0005

# Table 6 reference metrics — dataset-level scores from the paper. They are NOT
# per-image quantities (they need ground-truth masks), so they never reach the
# inference telemetry; they live on the Results/Diagnostics pages instead. Kept
# here only to scale synthetic mask fidelity when no checkpoint is available.
MODEL_METRICS = {
    "Baseline": {"sAlpha": 0.7647, "ePhi": 0.8061, "fBetaW": 0.6036, "mae": 0.0593, "boundaryIoU": 0.2332, "fpr": 1.41},
    "Exp M":    {"sAlpha": 0.8638, "ePhi": 0.9046, "fBetaW": 0.7718, "mae": 0.0348, "boundaryIoU": 0.3372, "fpr": 2.02},
    "Exp C1":   {"sAlpha": 0.7572, "ePhi": 0.8045, "fBetaW": 0.5886, "mae": 0.0538, "boundaryIoU": 0.2225, "fpr": 1.58},
    "Exp C2":   {"sAlpha": 0.6223, "ePhi": 0.6616, "fBetaW": 0.3541, "mae": 0.1007, "boundaryIoU": 0.1414, "fpr": 0.81},
    "Exp MC1":  {"sAlpha": 0.8675, "ePhi": 0.9095, "fBetaW": 0.7773, "mae": 0.0322, "boundaryIoU": 0.3437, "fpr": 1.55},
    "Exp MC2":  {"sAlpha": 0.8460, "ePhi": 0.8765, "fBetaW": 0.7276, "mae": 0.0423, "boundaryIoU": 0.3119, "fpr": 2.19},
    "Exp MCC":  {"sAlpha": 0.8518, "ePhi": 0.8802, "fBetaW": 0.7379, "mae": 0.0408, "boundaryIoU": 0.3199, "fpr": 1.99},
    "MC1 GhostNet": {"sAlpha": 0.8376, "ePhi": 0.8590, "fBetaW": 0.7078, "mae": 0.0451, "boundaryIoU": 0.2495, "fpr": 2.48},
    "MC1 ShuffleNet": {"sAlpha": 0.8294, "ePhi": 0.8515, "fBetaW": 0.6929, "mae": 0.0472, "boundaryIoU": 0.2503, "fpr": 2.39},
    "Exp MC1_ghostnet": {"sAlpha": 0.8376, "ePhi": 0.8590, "fBetaW": 0.7078, "mae": 0.0451, "boundaryIoU": 0.2495, "fpr": 2.48},
    "Exp MC1_shufflenet": {"sAlpha": 0.8294, "ePhi": 0.8515, "fBetaW": 0.6929, "mae": 0.0472, "boundaryIoU": 0.2503, "fpr": 2.39},
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
    "MC1 GhostNet": "Exp_MC1_ghostnet",
    "MC1 ShuffleNet": "Exp_MC1_shufflenet",
    "Exp MC1_ghostnet": "Exp_MC1_ghostnet",
    "Exp MC1_shufflenet": "Exp_MC1_shufflenet",
    "ghostnet": "Exp_MC1_ghostnet",
    "shufflenet": "Exp_MC1_shufflenet",
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
        if "ghostnet" in model_key.lower():
            from lib.Network_GhostNet_GRA_NCD import Network as GhostNetNetwork
            net = GhostNetNetwork(imagenet_pretrained=False)
        elif "shufflenet" in model_key.lower():
            from lib.Network_ShuffleNetV2_GRA_NCD import Network as ShuffleNetNetwork
            net = ShuffleNetNetwork(imagenet_pretrained=False)
        else:
            net = Network(imagenet_pretrained=False)
        net.load_state_dict(torch.load(str(pth_path), map_location="cpu", weights_only=False))
        net.to(DEVICE)
        net.eval()
        _model_cache[model_key] = net
        return net
    except Exception as e:
        print(f"[ai_service] Failed to load checkpoint for {model_key}: {e}")
        _model_cache[model_key] = None
        return None


def run_pytorch_inference(net, img_bgr: np.ndarray) -> np.ndarray:
    """Runs real SINet-V2 inference.

    Returns the RAW sigmoid probability map (float32 in [0, 1]) at original
    resolution. It is deliberately NOT min-max rescaled here: rescaling forces
    max->1 and min->0 on every image, which would destroy the calibration the
    confidence metric depends on. Normalisation happens later, for display only.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = PREPROCESS(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, _, _, s3_pred = net(tensor)
        res = F.interpolate(s3_pred, size=(h, w), mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()

    return res.astype(np.float32)


def run_synthetic_inference(img_bgr: np.ndarray, model_key: str) -> np.ndarray:
    """Synthetic probability map used when no trained checkpoint is available.
    Mask fidelity is scaled by the model's real Table 6 weighted F-measure so weaker
    training regimes visibly produce noisier, less precise masks.
    Returns float32 in [0, 1], matching run_pytorch_inference."""
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
    return (mask.astype(np.float32) / 255.0)


def to_display_mask(prob: np.ndarray) -> np.ndarray:
    """Min-max normalised uint8 view of the probability map, for rendering only."""
    lo, hi = float(prob.min()), float(prob.max())
    stretched = (prob - lo) / (hi - lo + 1e-8)
    return (stretched * 255).astype(np.uint8)


def significant_contours(binary_u8: np.ndarray):
    """External contours large enough to count as detections, plus the frame area."""
    h, w = binary_u8.shape[:2]
    frame_area = float(h * w)
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_REGION_FRACTION * frame_area
    return [c for c in contours if cv2.contourArea(c) >= min_area], frame_area


def compute_telemetry(prob: np.ndarray, binary: np.ndarray, elapsed_ms: float, engine: str) -> dict:
    """Per-image telemetry derived entirely from this image's prediction.

    No ground truth exists for a user upload, so nothing here is a dataset-level
    score — every value is a property of the probability map we just produced.
    """
    h, w = prob.shape[:2]
    frame_area = float(h * w)
    binary_u8 = (binary.astype(np.uint8) * 255)

    detected = binary.any()
    confidence = float(prob[binary].mean() * 100) if detected else 0.0
    peak_confidence = float(prob.max() * 100)
    camouflage_area = float(binary.sum() / frame_area * 100)

    contours, _ = significant_contours(binary_u8)
    largest_area = max((cv2.contourArea(c) for c in contours), default=0.0)
    largest_region = float(largest_area / frame_area * 100)

    # Boundary crispness: mean gradient of the probability map sampled on a band
    # around the predicted contour. A hard 0->1 step saturates at 1.0, a soft or
    # smeared boundary reads lower. Stands in for Boundary IoU, which needs GT.
    edge_definition = 0.0
    if contours:
        # A one-pixel band: widening it would average in flat interior/exterior
        # pixels and squash the score into a narrow, unreadable range.
        band = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(band, contours, -1, 255, 1)
        if np.any(band > 0):
            gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
            # /4.0 undoes the ksize=3 Sobel kernel gain so a unit step maps to ~1.0
            grad = np.sqrt(gx * gx + gy * gy) / 4.0
            edge_definition = float(np.clip(grad[band > 0].mean(), 0.0, 1.0))

    # How much of the frame the model could not commit either way.
    ambiguity = float(((prob > 0.3) & (prob < 0.7)).sum() / frame_area * 100)

    if not contours:
        threat_level = "NONE"
    else:
        # Weight area by confidence so a large, hesitant blob is not called CRITICAL.
        score = camouflage_area * (confidence / 100.0)
        if score > 10:
            threat_level = "CRITICAL"
        elif score > 3:
            threat_level = "ELEVATED"
        else:
            threat_level = "LOW"

    return {
        "confidenceScore": round(confidence, 1),
        "peakConfidence": round(peak_confidence, 1),
        "camouflageAreaPercent": round(camouflage_area, 2),
        "detectedRegions": len(contours),
        "largestRegionPercent": round(largest_region, 2),
        "edgeDefinition": round(edge_definition, 3),
        "ambiguityPercent": round(ambiguity, 2),
        "inferenceMs": round(elapsed_ms, 1),
        "threatLevel": threat_level,
        "inferenceEngine": engine,
    }


def generate_tactical_overlays(binary_u8: np.ndarray, display_mask: np.ndarray, color_bgr=(16, 185, 129)):
    """
    Given the thresholded detection mask (uint8 0/255) and a display-normalised
    grayscale map, produces:
    1. combined: RGBA image with semi-transparent fill + sharp glowing contour outline.
    2. outline: RGBA image with ONLY the sharp glowing contour outline.
    3. fill: RGBA image with ONLY the semi-transparent fill.
    4. raw: Grayscale probability map.

    Contours come from the same binary cut used for telemetry, so the drawn
    region matches the reported camouflage area exactly.
    """
    h, w = binary_u8.shape[:2]

    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1. Fill layer (RGBA)
    fill_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    fill_rgba[binary_u8 > 0] = [color_bgr[0], color_bgr[1], color_bgr[2], 160]

    # 2. Outline layer (RGBA)
    outline_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Bright tactical green (50, 255, 120, 255) with 2px contour
    cv2.drawContours(outline_rgba, contours, -1, (50, 255, 120, 255), 2)

    # 3. Combined overlay (fill + outline)
    combined_rgba = fill_rgba.copy()
    cv2.drawContours(combined_rgba, contours, -1, (50, 255, 120, 255), 2)

    def to_b64(img):
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode("utf-8")

    return {
        "combined": to_b64(combined_rgba),
        "outline": to_b64(outline_rgba),
        "fill": to_b64(fill_rgba),
        "raw": to_b64(display_mask)
    }


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

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode uploaded image"}

    net = get_model(model_key)
    engine = "pytorch"
    started = time.perf_counter()
    if net is not None:
        try:
            prob = run_pytorch_inference(net, img)
        except Exception as e:
            print(f"[ai_service] Inference failed for {model_key}, falling back to synthetic: {e}")
            prob = run_synthetic_inference(img, model_key)
            engine = "synthetic"
    else:
        prob = run_synthetic_inference(img, model_key)
        engine = "synthetic"
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    binary = prob > DETECTION_THRESHOLD
    binary_u8 = (binary.astype(np.uint8) * 255)

    overlays = generate_tactical_overlays(binary_u8, to_display_mask(prob))
    telemetry = compute_telemetry(prob, binary, elapsed_ms, engine)
    telemetry["model"] = model_key

    return {
        "maskBase64": overlays["combined"],
        "overlayBase64": overlays["combined"],
        "outlineBase64": overlays["outline"],
        "fillBase64": overlays["fill"],
        "rawMaskBase64": overlays["raw"],
        "telemetry": telemetry,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
