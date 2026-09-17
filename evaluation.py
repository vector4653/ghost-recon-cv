import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import py_sod_metrics

def main():
    parser = argparse.ArgumentParser(description="PySODMetrics Evaluation for Military Camouflaged Object Detection")
    parser.add_argument('--gt_dir', type=str, default="./Dataset/TestDataset/GT/", help='Ground truth directory')
    parser.add_argument('--pred_dir', type=str, default="./res/SINet_V2/", help='Predicted masks directory')
    parser.add_argument('--title', type=str, default="Baseline (Zero-Shot on Military Domain)", help='Evaluation title')
    opt = parser.parse_args()

    gt_dir = opt.gt_dir
    pred_dir = opt.pred_dir

    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    WFM = py_sod_metrics.WeightedFmeasure()
    MAE = py_sod_metrics.MAE()

    if not os.path.exists(pred_dir):
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")

    mask_name_list = sorted(os.listdir(pred_dir))
    print(f"Found {len(mask_name_list)} predictions in {pred_dir} to evaluate against {gt_dir}...")

    evaluated_count = 0
    for mask_name in tqdm(mask_name_list):
        pred_path = os.path.join(pred_dir, mask_name)
        gt_path = os.path.join(gt_dir, mask_name)
        
        if not os.path.exists(gt_path):
            stem, _ = os.path.splitext(mask_name)
            for ext in ['.png', '.jpg', '.jpeg']:
                cand = os.path.join(gt_dir, stem + ext)
                if os.path.exists(cand):
                    gt_path = cand
                    break
            if not os.path.exists(gt_path):
                continue
            
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if pred_mask is None or gt_mask is None:
            continue

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        SM.step(pred=pred_mask, gt=gt_mask)
        EM.step(pred=pred_mask, gt=gt_mask)
        WFM.step(pred=pred_mask, gt=gt_mask)
        MAE.step(pred=pred_mask, gt=gt_mask)
        evaluated_count += 1

    sm = SM.get_results()['sm']
    em = EM.get_results()['em']['adp']
    wfm = WFM.get_results()['wfm']
    mae = MAE.get_results()['mae']

    print(f"\n=== Evaluation Results: {opt.title} ({evaluated_count} images) ===")
    print(f"S-measure (S_alpha):      {sm:.4f}")
    print(f"E-measure (E_phi adp):    {em:.4f}")
    print(f"Weighted F-measure (F_w): {wfm:.4f}")
    print(f"MAE:                      {mae:.4f}")
    print("=======================================================")

    return {'sm': sm, 'em': em, 'wfm': wfm, 'mae': mae}

if __name__ == '__main__':
    main()