import os
import shutil
import random

BASE_DIR = r"C:\Users\ritvi\Downloads\Apps\Datsets-CV"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def standardize_camo():
    print("Standardizing CAMO...")
    src_train_img = os.path.join(BASE_DIR, "CAMO-V.1.0", "Images", "Train")
    src_test_img = os.path.join(BASE_DIR, "CAMO-V.1.0", "Images", "Test")
    src_gt = os.path.join(BASE_DIR, "CAMO-V.1.0", "GT")

    dst_train_img = os.path.join(BASE_DIR, "CAMO", "TrainValDataset", "Images")
    dst_train_gt = os.path.join(BASE_DIR, "CAMO", "TrainValDataset", "GT")
    dst_test_img = os.path.join(BASE_DIR, "CAMO", "TestDataset", "Images")
    dst_test_gt = os.path.join(BASE_DIR, "CAMO", "TestDataset", "GT")

    for d in [dst_train_img, dst_train_gt, dst_test_img, dst_test_gt]:
        ensure_dir(d)

    train_files = sorted(os.listdir(src_train_img))
    for f in train_files:
        stem, _ = os.path.splitext(f)
        shutil.copy2(os.path.join(src_train_img, f), os.path.join(dst_train_img, f))
        gt_file = stem + ".png"
        shutil.copy2(os.path.join(src_gt, gt_file), os.path.join(dst_train_gt, gt_file))
    print(f"  CAMO TrainValDataset: {len(train_files)} pairs copied.")

    test_files = sorted(os.listdir(src_test_img))
    for f in test_files:
        stem, _ = os.path.splitext(f)
        shutil.copy2(os.path.join(src_test_img, f), os.path.join(dst_test_img, f))
        gt_file = stem + ".png"
        shutil.copy2(os.path.join(src_gt, gt_file), os.path.join(dst_test_gt, gt_file))
    print(f"  CAMO TestDataset: {len(test_files)} pairs copied.")

def standardize_cod10k():
    print("Standardizing COD10K...")
    src_train_img = os.path.join(BASE_DIR, "COD10K-v3", "Train", "Image")
    src_train_gt = os.path.join(BASE_DIR, "COD10K-v3", "Train", "GT_Object")
    src_test_img = os.path.join(BASE_DIR, "COD10K-v3", "Test", "Image")
    src_test_gt = os.path.join(BASE_DIR, "COD10K-v3", "Test", "GT_Object")

    dst_train_img = os.path.join(BASE_DIR, "COD10K", "TrainValDataset", "Images")
    dst_train_gt = os.path.join(BASE_DIR, "COD10K", "TrainValDataset", "GT")
    dst_test_img = os.path.join(BASE_DIR, "COD10K", "TestDataset", "Images")
    dst_test_gt = os.path.join(BASE_DIR, "COD10K", "TestDataset", "GT")

    for d in [dst_train_img, dst_train_gt, dst_test_img, dst_test_gt]:
        ensure_dir(d)

    train_files = sorted(os.listdir(src_train_img))
    for f in train_files:
        stem, _ = os.path.splitext(f)
        shutil.copy2(os.path.join(src_train_img, f), os.path.join(dst_train_img, f))
        gt_file = stem + ".png"
        shutil.copy2(os.path.join(src_train_gt, gt_file), os.path.join(dst_train_gt, gt_file))
    print(f"  COD10K TrainValDataset: {len(train_files)} pairs copied.")

    test_files = sorted(os.listdir(src_test_img))
    for f in test_files:
        stem, _ = os.path.splitext(f)
        shutil.copy2(os.path.join(src_test_img, f), os.path.join(dst_test_img, f))
        gt_file = stem + ".png"
        shutil.copy2(os.path.join(src_test_gt, gt_file), os.path.join(dst_test_gt, gt_file))
    print(f"  COD10K TestDataset: {len(test_files)} pairs copied.")

def standardize_military(seed=42):
    print("Standardizing Military_Dataset...")
    random.seed(seed)

    dst_train_img = os.path.join(BASE_DIR, "Military_Dataset", "TrainValDataset", "Images")
    dst_train_gt = os.path.join(BASE_DIR, "Military_Dataset", "TrainValDataset", "GT")
    dst_test_img = os.path.join(BASE_DIR, "Military_Dataset", "TestDataset", "Images")
    dst_test_gt = os.path.join(BASE_DIR, "Military_Dataset", "TestDataset", "GT")

    for d in [dst_train_img, dst_train_gt, dst_test_img, dst_test_gt]:
        ensure_dir(d)

    # 1. ACD1K
    acd_train_img_dir = os.path.join(BASE_DIR, "ACD1K", "Training", "images")
    acd_train_gt_dir = os.path.join(BASE_DIR, "ACD1K", "Training", "GT")
    acd_test_img_dir = os.path.join(BASE_DIR, "ACD1K", "Testing", "images")
    acd_test_gt_dir = os.path.join(BASE_DIR, "ACD1K", "Testing", "GT")

    for f in sorted(os.listdir(acd_train_img_dir)):
        stem, _ = os.path.splitext(f)
        shutil.copy2(os.path.join(acd_train_img_dir, f), os.path.join(dst_train_img, f))
        gt_cand = os.path.join(acd_train_gt_dir, stem + ".jpg")
        if not os.path.exists(gt_cand):
            gt_cand = os.path.join(acd_train_gt_dir, stem + ".png")
        shutil.copy2(gt_cand, os.path.join(dst_train_gt, os.path.basename(gt_cand)))

    for f in sorted(os.listdir(acd_test_img_dir)):
        stem, _ = os.path.splitext(f)
        shutil.copy2(os.path.join(acd_test_img_dir, f), os.path.join(dst_test_img, f))
        gt_cand = os.path.join(acd_test_gt_dir, stem + ".png")
        if not os.path.exists(gt_cand):
            gt_cand = os.path.join(acd_test_gt_dir, stem + ".jpg")
        shutil.copy2(gt_cand, os.path.join(dst_test_gt, os.path.basename(gt_cand)))

    print("  ACD1K copied: 748 train, 330 test.")

    # 2. CPD1K
    cpd_img_dir = os.path.join(BASE_DIR, "CPD1K", "img")
    cpd_gt_dir = os.path.join(BASE_DIR, "CPD1K", "gt")
    all_cpd = sorted([os.path.splitext(f)[0] for f in os.listdir(cpd_img_dir)])

    train_stems = set(random.sample(all_cpd, 750))
    test_stems = set(all_cpd) - train_stems

    for stem in train_stems:
        shutil.copy2(os.path.join(cpd_img_dir, stem + ".jpg"), os.path.join(dst_train_img, stem + ".jpg"))
        shutil.copy2(os.path.join(cpd_gt_dir, stem + ".png"), os.path.join(dst_train_gt, stem + ".png"))

    for stem in test_stems:
        shutil.copy2(os.path.join(cpd_img_dir, stem + ".jpg"), os.path.join(dst_test_img, stem + ".jpg"))
        shutil.copy2(os.path.join(cpd_gt_dir, stem + ".png"), os.path.join(dst_test_gt, stem + ".png"))

    print(f"  CPD1K 3:1 split: 750 train, 250 test.")
    print(f"  Total Military_Dataset TrainValDataset: {len(os.listdir(dst_train_img))} images")
    print(f"  Total Military_Dataset TestDataset: {len(os.listdir(dst_test_img))} images")

if __name__ == "__main__":
    standardize_military()
    standardize_camo()
    standardize_cod10k()
    print("Data standardization successfully complete!")
