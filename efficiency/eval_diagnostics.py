import os
import sys
import json
import numpy as np
import cv2
from tqdm import tqdm

def calculate_boundary_iou(gt_mask, pred_mask, dilation_radius=4):
    """
    Computes Boundary IoU within a narrow band around the ground truth boundary.
    Exact implementation from generate_qualitative_analysis.py.
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


def compute_diagnostics_for_dir(pred_dir, gt_dir="./Dataset/TestDataset/GT"):
    gt_files = sorted(os.listdir(gt_dir))
    boundary_list = []
    fpr_list = []
    leakage_list = []

    for gt_file in tqdm(gt_files, desc=f"Diagnostics for {os.path.basename(pred_dir)}"):
        stem, _ = os.path.splitext(gt_file)
        gt_path = os.path.join(gt_dir, gt_file)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            continue
        h, w = gt_img.shape
        gt_bin = (gt_img > 128).astype(np.uint8)
        bg_mask = (gt_bin == 0)
        num_bg_pixels = bg_mask.sum()

        pred_file = os.path.join(pred_dir, stem + ".png")
        if not os.path.exists(pred_file):
            pred_file = os.path.join(pred_dir, stem + ".jpg")
        if not os.path.exists(pred_file):
            continue

        pred_img = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        if pred_img is None:
            continue
        if pred_img.shape != (h, w):
            pred_img = cv2.resize(pred_img, (w, h))

        pred_bin = (pred_img > 128).astype(np.uint8)

        # Boundary IoU
        b_iou = calculate_boundary_iou(gt_bin, pred_bin, dilation_radius=4)
        boundary_list.append(b_iou)

        # FPR on tactical background
        fp_pixels = (pred_bin[bg_mask] > 0).sum()
        fpr = fp_pixels / max(num_bg_pixels, 1)
        fpr_list.append(fpr)

        # Mean background leakage intensity (0 to 1)
        bg_intensity = float(pred_img[bg_mask].mean() / 255.0) if num_bg_pixels > 0 else 0.0
        leakage_list.append(bg_intensity)

    mean_b_iou = float(np.mean(boundary_list)) if boundary_list else 0.0
    mean_fpr = float(np.mean(fpr_list)) if fpr_list else 0.0
    mean_leakage = float(np.mean(leakage_list)) if leakage_list else 0.0

    return {
        'boundary_iou_4px': round(mean_b_iou, 4),
        'fpr_background': round(mean_fpr, 4),
        'background_leakage': round(mean_leakage, 4),
        'num_samples': len(boundary_list)
    }


def update_model_json(json_path, eval_dict):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if 'evaluation' not in data:
        data['evaluation'] = {}
    data['evaluation'].update(eval_dict)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[+] Updated {json_path} with evaluation metrics.")


if __name__ == '__main__':
    models = {
        'res2net': './res/Exp_MC1',
        'ghostnet': './res/Exp_MC1_ghostnet',
        'shufflenet': './res/Exp_MC1_shufflenet'
    }

    results = {}
    for key, pdir in models.items():
        if os.path.exists(pdir):
            diag = compute_diagnostics_for_dir(pdir)
            results[key] = diag
            print(f"{key}: Boundary IoU={diag['boundary_iou_4px']}, FPR={diag['fpr_background']*100:.2f}%, Leakage={diag['background_leakage']}")
            json_file = f"efficiency/results/{key}.json"
            if os.path.exists(json_file):
                update_model_json(json_file, diag)
        else:
            print(f"[-] Directory {pdir} not found, skipping.")
