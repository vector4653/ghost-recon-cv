import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec


def get_image_path(img_dir, stem):
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def calculate_boundary_iou(gt_mask, pred_mask, dilation_radius=4):
    """
    Computes Boundary IoU within a narrow band around the ground truth boundary.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
    gt_dilated = cv2.dilate(gt_mask, kernel)
    gt_eroded = cv2.erode(gt_mask, kernel)
    gt_boundary = gt_dilated - gt_eroded

    pred_dilated = cv2.dilate(pred_mask, kernel)
    pred_eroded = cv2.erode(pred_mask, kernel)
    pred_boundary = pred_dilated - pred_eroded

    intersection = np.logical_and(gt_boundary > 0, pred_boundary > 0).sum()
    union = np.logical_or(gt_boundary > 0, pred_boundary > 0).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def run_qualitative_analysis():
    print("=" * 70)
    print("  Ghost-Recon-CV: Phase 3 Qualitative Analysis & Visualization")
    print("=" * 70)

    out_dir = "./qualitative_analysis"
    os.makedirs(out_dir, exist_ok=True)

    test_img_dir = "./Dataset/TestDataset/Images"
    test_gt_dir = "./Dataset/TestDataset/GT"

    models = {
        'Baseline (Off-the-shelf)': './res/SINet_V2',
        'Exp M (Military)': './res/Exp_M',
        'Exp C1 (CAMO)': './res/Exp_C1',
        'Exp C2 (COD10K)': './res/Exp_C2',
        'Exp MC1 (Military + CAMO - Winner)': './res/Exp_MC1',
        'Exp MC2 (Military + COD10K)': './res/Exp_MC2',
        'Exp MCC (Tri-Domain Fusion)': './res/Exp_MCC',
    }

    # Verify model result dirs
    for name, path in models.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing predictions folder: {path}")

    gt_files = sorted(os.listdir(test_gt_dir))
    print(f"[+] Found {len(gt_files)} ground truth masks for evaluation.")

    # -------------------------------------------------------------------------
    # 1. Compute Quantitative Boundary and False Positive Diagnostics
    # -------------------------------------------------------------------------
    print("\n[Step 1/3] Computing Boundary Sharpness and False Positive Diagnostics...")
    
    boundary_metrics = {m: [] for m in models}
    fpr_metrics = {m: [] for m in models}
    fp_area_metrics = {m: [] for m in models}
    bce_background_metrics = {m: [] for m in models}

    for idx, gt_file in enumerate(gt_files):
        stem, _ = os.path.splitext(gt_file)
        gt_path = os.path.join(test_gt_dir, gt_file)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            continue
        h, w = gt_img.shape
        gt_bin = (gt_img > 128).astype(np.uint8)
        bg_mask = (gt_bin == 0)
        num_bg_pixels = bg_mask.sum()

        for m_name, m_dir in models.items():
            pred_file = os.path.join(m_dir, stem + ".png")
            if not os.path.exists(pred_file):
                pred_file = os.path.join(m_dir, stem + ".jpg")
            pred_img = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
            if pred_img is None:
                continue
            if pred_img.shape != (h, w):
                pred_img = cv2.resize(pred_img, (w, h))

            pred_bin = (pred_img > 128).astype(np.uint8)

            # Boundary IoU
            b_iou = calculate_boundary_iou(gt_bin, pred_bin, dilation_radius=4)
            boundary_metrics[m_name].append(b_iou)

            # False Positives on tactical background
            fp_pixels = (pred_bin[bg_mask] > 0).sum()
            fpr = fp_pixels / max(num_bg_pixels, 1)
            fpr_metrics[m_name].append(fpr)
            fp_area_metrics[m_name].append(float(fp_pixels / (w * h)))

            # Mean background leakage intensity (0 to 1)
            bg_intensity = float(pred_img[bg_mask].mean() / 255.0) if num_bg_pixels > 0 else 0.0
            bce_background_metrics[m_name].append(bg_intensity)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(gt_files):
            print(f"  Processed {idx + 1}/{len(gt_files)} images...")

    summary_stats = {}
    print("\n" + "=" * 75)
    print(f"{'Model':<35} | {'Boundary IoU':<12} | {'FPR (Tactical BG)':<18} | {'BG Leakage':<10}")
    print("-" * 75)
    for m_name in models:
        m_biou = float(np.mean(boundary_metrics[m_name]))
        m_fpr = float(np.mean(fpr_metrics[m_name]))
        m_leak = float(np.mean(bce_background_metrics[m_name]))
        summary_stats[m_name] = {
            'mean_boundary_iou': m_biou,
            'mean_fpr': m_fpr,
            'mean_bg_leakage': m_leak
        }
        print(f"{m_name:<35} | {m_biou:<12.4f} | {m_fpr*100:<17.2f}% | {m_leak:<10.4f}")
    print("=" * 75)

    with open(os.path.join(out_dir, "diagnostics_summary.json"), "w") as f:
        json.dump(summary_stats, f, indent=2)

    # -------------------------------------------------------------------------
    # 2. Generate Boundary Sharpness & False Positive Diagnostic Plots
    # -------------------------------------------------------------------------
    print("\n[Step 2/3] Generating Boundary Sharpness & Diagnostic Figures...")

    # Plot 1: Boundary IoU Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    labels = [
        'Baseline\n(Off-shelf)', 'Exp M\n(Military)', 'Exp C1\n(CAMO)', 
        'Exp C2\n(COD10K)', 'Exp MC1\n(M+CAMO)*', 'Exp MC2\n(M+COD)', 'Exp MCC\n(Tri-Domain)'
    ]
    b_values = [summary_stats[m]['mean_boundary_iou'] for m in models]
    colors = ['#888888', '#2ca02c', '#1f77b4', '#d62728', '#ff7f0e', '#9467bd', '#8c564b']
    
    bars = ax.bar(labels, b_values, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    ax.set_ylabel('Boundary IoU (Dilation = 4px)', fontsize=12, fontweight='bold')
    ax.set_title('Contour & Boundary Delineation Accuracy across Models', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0.0, max(b_values) * 1.18)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f'{h:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    b_plot_path = os.path.join(out_dir, "boundary_sharpness_analysis.png")
    plt.savefig(b_plot_path)
    plt.close()
    print(f"  -> Saved {b_plot_path}")

    # Plot 2: Background Distractor False Alarm Analysis (Tactical OODA Loop Context)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fpr_values = [summary_stats[m]['mean_fpr'] * 100 for m in models]
    leak_values = [summary_stats[m]['mean_bg_leakage'] * 100 for m in models]

    b1 = ax1.bar(labels, fpr_values, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    ax1.set_ylabel('False Positive Rate in Background (%)', fontsize=11, fontweight='bold')
    ax1.set_title('False Alarm Rate on Tactical Background Pixels (FPR ↓)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.1, f'{h:.2f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    b2 = ax2.bar(labels, leak_values, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    ax2.set_ylabel('Mean Background Leakage Intensity (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Background Uncertainty & Noise Leakage (Lower is cleaner)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.1, f'{h:.2f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fa_plot_path = os.path.join(out_dir, "false_alarm_background_diagnostics.png")
    plt.savefig(fa_plot_path)
    plt.close()
    print(f"  -> Saved {fa_plot_path}")

    # -------------------------------------------------------------------------
    # 3. Generate Multi-Scene Visual Comparison Grid & Overlay Panels
    # -------------------------------------------------------------------------
    print("\n[Step 3/3] Generating High-Resolution Qualitative Comparison Grid...")

    # Curate diverse challenging scenes
    # 1. Distant concealed sniper: image974
    # 2. Forest soldier in brush: dataset01_09_00014463
    # 3. Heavy brush prone stance: dataset18_11_00016470
    # 4. Multi-camo tactical operator: image1005
    # 5. Woodland pattern operator: image721
    # 6. High-contrast uniform: image697
    curated_samples = [
        ("image974", "Concealed Long-Range Sniper"),
        ("dataset01_09_00014463", "Woodland Brush Operator"),
        ("dataset18_11_00016470", "Prone Position in Dense Foliage"),
        ("image1005", "Multi-Terrain Tactical Operator"),
        ("image721", "Urban/Woodland Camouflage"),
        ("image697", "Close-Range Tactical Stance")
    ]

    grid_models = [
        ('Original RGB', 'RGB'),
        ('Ground Truth', 'GT'),
        ('Baseline (COD)', './res/SINet_V2'),
        ('Exp M (Military)', './res/Exp_M'),
        ('Exp C1 (CAMO)', './res/Exp_C1'),
        ('Exp C2 (COD10K)', './res/Exp_C2'),
        ('Exp MC1 (Winner)', './res/Exp_MC1'),
        ('Exp MCC (Tri-Domain)', './res/Exp_MCC'),
    ]

    n_rows = len(curated_samples)
    n_cols = len(grid_models)

    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.0), dpi=120)
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.04, hspace=0.12)

    for row_idx, (stem, label) in enumerate(curated_samples):
        # Load RGB
        img_path = get_image_path(test_img_dir, stem)
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Load GT
        gt_path = os.path.join(test_gt_dir, stem + ".png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(test_gt_dir, stem + ".jpg")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        for col_idx, (col_title, col_source) in enumerate(grid_models):
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

            # Column headers on top row
            if row_idx == 0:
                ax.set_title(col_title, fontsize=11, fontweight='bold', pad=8)

            # Row label on first column
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, fontweight='bold', rotation=90, labelpad=8)

    plt.suptitle("Qualitative Camouflaged Soldier Segmentation Across Multi-Domain Models", 
                 fontsize=16, fontweight='bold', y=0.995)

    grid_path = os.path.join(out_dir, "visual_comparison_grid.png")
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {grid_path}")

    # -------------------------------------------------------------------------
    # 4. Generate Tactical Overlay Panel (RGB + Green GT + Red Error)
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Generating Tactical Color Mask Overlay Comparison...")
    fig, axes = plt.subplots(len(curated_samples), 4, figsize=(16, len(curated_samples) * 3.2), dpi=120)

    for i, (stem, label) in enumerate(curated_samples):
        img_path = get_image_path(test_img_dir, stem)
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        gt_path = os.path.join(test_gt_dir, stem + ".png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(test_gt_dir, stem + ".jpg")
        gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 128)

        # Baseline overlay
        p_base = cv2.resize(cv2.imread(os.path.join('./res/SINet_V2', stem + ".png"), 0), (w, h)) > 128
        # Exp M overlay
        p_m = cv2.resize(cv2.imread(os.path.join('./res/Exp_M', stem + ".png"), 0), (w, h)) > 128
        # Exp MC1 overlay
        p_mc1 = cv2.resize(cv2.imread(os.path.join('./res/Exp_MC1', stem + ".png"), 0), (w, h)) > 128

        def make_overlay(img, pred, gt):
            overlay = img.copy()
            # True positive: Yellow [255, 255, 0]
            tp = np.logical_and(pred, gt)
            # False positive: Red [255, 0, 0]
            fp = np.logical_and(pred, ~gt)
            # False negative: Blue [0, 150, 255]
            fn = np.logical_and(~pred, gt)

            overlay[tp] = (overlay[tp] * 0.4 + np.array([255, 220, 0]) * 0.6).astype(np.uint8)
            overlay[fp] = (overlay[fp] * 0.4 + np.array([255, 30, 30]) * 0.6).astype(np.uint8)
            overlay[fn] = (overlay[fn] * 0.4 + np.array([0, 140, 255]) * 0.6).astype(np.uint8)
            return overlay

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Target: {label}" if i == 0 else "")
        axes[i, 0].set_ylabel(label, fontsize=9, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(make_overlay(rgb, p_base, gt))
        if i == 0: axes[i, 1].set_title("Zero-Shot Baseline (Error Overlay)", fontsize=10, fontweight='bold')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        axes[i, 2].imshow(make_overlay(rgb, p_m, gt))
        if i == 0: axes[i, 2].set_title("Exp M: Military Only (Error Overlay)", fontsize=10, fontweight='bold')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        axes[i, 3].imshow(make_overlay(rgb, p_mc1, gt))
        if i == 0: axes[i, 3].set_title("Exp MC1: Best Fusion (Error Overlay)", fontsize=10, fontweight='bold')
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])

    plt.suptitle("Tactical Diagnostics Overlay: Yellow = True Positive | Red = False Alarm (FP) | Blue = Missed (FN)",
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    overlay_path = os.path.join(out_dir, "tactical_overlay_comparison.png")
    plt.savefig(overlay_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {overlay_path}")

    print("\n[+] Phase 3 Qualitative Analysis Completed Successfully!")


if __name__ == '__main__':
    run_qualitative_analysis()
