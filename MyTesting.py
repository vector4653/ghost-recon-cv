import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/SINet_V2/Net_epoch_best.pth')
opt = parser.parse_args()

# 1. Initialize the model
model = Network(imagenet_pretrained=False)

# 2. Map NVIDIA-trained weights to CPU first to prevent the Pickle error
model.load_state_dict(torch.load(opt.pth_path, map_location='cpu'))

# 3. Select device (CUDA on Windows RTX 4050, MPS on Apple Silicon, or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")
model.to(device)
model.eval()

# 4. Point directly to your standardized Military split
image_root = './Dataset/TestDataset/Images/'
gt_root = './Dataset/TestDataset/GT/'
save_path = './res/SINet_V2/'

os.makedirs(save_path, exist_ok=True)

test_loader = test_dataset(image_root, gt_root, opt.testsize)
print(f"Found {test_loader.size} military test images. Starting inference...")

for i in range(test_loader.size):
    image, gt, name, _ = test_loader.load_data()
    
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    
    # 5. Move the image tensors to device
    image = image.to(device)
    
    # Forward pass: Unpack the 4 multi-scale outputs
    res5, res4, res3, res2 = model(image)
    
    # 6. Pass ONLY the final prediction map (res2) to interpolate
    res = F.interpolate(res2, size=gt.shape, mode='bilinear', align_corners=False)
        
    res = res.sigmoid().data.cpu().numpy().squeeze()
    
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    
    # 7. Fix the scipy.misc.imsave bug using OpenCV
    cv2.imwrite(save_path + name, (res * 255).astype(np.uint8))
    
    if i % 50 == 0:
        print(f"Processed {i}/{test_loader.size}: {name}")

print("Inference complete! Masks saved to ./res/SINet_V2/")