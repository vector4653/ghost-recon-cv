import os
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\ritvi\Downloads\Apps\Datsets-CV"
REPORT_DIR = os.path.join(BASE_DIR, "dataset_analysis_report")
os.makedirs(REPORT_DIR, exist_ok=True)

datasets = {
    "Military_Dataset": os.path.join(BASE_DIR, "Military_Dataset"),
    "CAMO": os.path.join(BASE_DIR, "CAMO"),
    "COD10K": os.path.join(BASE_DIR, "COD10K")
}

stats = {}

for ds_name, ds_path in datasets.items():
    print(f"Analyzing {ds_name}...")
    widths = []
    heights = []
    aspect_ratios = []
    object_ratios = []
    foreground_counts = []
    
    for split in ["TrainValDataset", "TestDataset"]:
        img_dir = os.path.join(ds_path, split, "Images")
        gt_dir = os.path.join(ds_path, split, "GT")
        
        for f in os.listdir(img_dir):
            stem, _ = os.path.splitext(f)
            img_p = os.path.join(img_dir, f)
            
            # Find matching GT
            gt_p = os.path.join(gt_dir, stem + ".png")
            if not os.path.exists(gt_p):
                gt_p = os.path.join(gt_dir, stem + ".jpg")
            
            # Read image resolution
            with Image.open(img_p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / max(h, 1))
            
            # Read mask and compute object size ratio
            with Image.open(gt_p) as mask:
                mask_np = np.array(mask)
                # Binarize if necessary
                if mask_np.ndim == 3:
                    mask_np = mask_np[:, :, 0]
                fg_pixels = np.count_nonzero(mask_np > 127)
                total_pixels = w * h
                ratio = fg_pixels / max(total_pixels, 1)
                object_ratios.append(ratio)
                if fg_pixels > 0:
                    foreground_counts.append(1)
                else:
                    foreground_counts.append(0)

    stats[ds_name] = {
        "total_images": len(widths),
        "mean_width": float(np.mean(widths)),
        "mean_height": float(np.mean(heights)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        "mean_object_ratio": float(np.mean(object_ratios)),
        "median_object_ratio": float(np.median(object_ratios)),
        "min_object_ratio": float(np.min(object_ratios)),
        "max_object_ratio": float(np.max(object_ratios)),
        "std_object_ratio": float(np.std(object_ratios)),
        "foreground_rate": float(np.mean(foreground_counts)),
        "widths": widths,
        "heights": heights,
        "object_ratios": object_ratios
    }
    print(f"  {ds_name} processed: {len(widths)} images. Mean obj ratio: {stats[ds_name]['mean_object_ratio']:.4f}")

# Save JSON summary (excluding raw lists)
summary_json = {
    k: {sub_k: v for sub_k, v in stats[k].items() if sub_k not in ["widths", "heights", "object_ratios"]}
    for k in stats
}
with open(os.path.join(REPORT_DIR, "dataset_statistics.json"), "w") as fp:
    json.dump(summary_json, fp, indent=4)

# Plot Figure: Object-size Ratios Distribution (Mirroring COD10K Fig 7 / 8)
plt.figure(figsize=(10, 6))
colors = {"Military_Dataset": "tab:blue", "CAMO": "tab:orange", "COD10K": "tab:green"}
for ds_name in datasets:
    ratios = [r for r in stats[ds_name]["object_ratios"] if r > 0]
    plt.hist(ratios, bins=50, alpha=0.5, density=True, label=f"{ds_name} (mean: {np.mean(ratios):.3f})", color=colors[ds_name])

plt.title("Object-Size Ratio Distribution ($\sqrt{Area_{target} / Area_{image}}$ or Pixel Ratio)")
plt.xlabel("Object Size Ratio (Target Area / Image Area)")
plt.ylabel("Density")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "figure_object_size_ratios.png"), dpi=300)
plt.close()

# Plot Figure: Resolution Spread (Width vs Height Scatter)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, ds_name in enumerate(datasets):
    ax = axes[idx]
    w = stats[ds_name]["widths"]
    h = stats[ds_name]["heights"]
    ax.scatter(w, h, alpha=0.3, s=10, color=colors[ds_name])
    ax.set_title(f"{ds_name} Resolution Spread\n(N={len(w)})")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_xlim(0, max(2500, max(w)+100))
    ax.set_ylim(0, max(2500, max(h)+100))
    ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "figure_resolution_spread.png"), dpi=300)
plt.close()

print("Analysis and Figures successfully generated in:", REPORT_DIR)
