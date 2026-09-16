import os
import cv2
import numpy as np
from tqdm import tqdm
import py_sod_metrics

# 1. Point to your Military Ground Truth and your fresh predictions
gt_dir = "./Dataset/TestDataset/GT/"
pred_dir = "./res/SINet_V2/"

# 2. Initialize the PySODMetrics calculators
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
WFM = py_sod_metrics.WeightedFmeasure()
MAE = py_sod_metrics.MAE()

mask_name_list = sorted(os.listdir(pred_dir))
print(f"Found {len(mask_name_list)} predictions to evaluate...")

for mask_name in tqdm(mask_name_list):
    pred_path = os.path.join(pred_dir, mask_name)
    gt_path = os.path.join(gt_dir, mask_name)
    
    # Fallback check in case a GT mask is missing
    if not os.path.exists(gt_path):
        continue
        
    # Load masks in grayscale
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize prediction to exactly match GT dimensions if they differ
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # 3. Update the metrics per image
    SM.step(pred=pred_mask, gt=gt_mask)
    EM.step(pred=pred_mask, gt=gt_mask)
    WFM.step(pred=pred_mask, gt=gt_mask)
    MAE.step(pred=pred_mask, gt=gt_mask)

# 4. Extract and print the final baseline scores
print("\n=== Baseline Results (Zero-Shot on Military Domain) ===")
print(f"S-measure:          {SM.get_results()['sm']:.4f}")
print(f"E-measure (adp):    {EM.get_results()['em']['adp']:.4f}")
print(f"Weighted F-measure: {WFM.get_results()['wfm']:.4f}")
print(f"MAE:                {MAE.get_results()['mae']:.4f}")
print("=======================================================")