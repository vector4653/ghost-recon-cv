import os
import random
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


DATASET_PATHS = {
    'Military': os.path.abspath('Formated Datsets/Military_Dataset/TrainValDataset'),
    'CAMO': os.path.abspath('Formated Datsets/CAMO/TrainValDataset'),
    'COD10K': os.path.abspath('Formated Datsets/COD10K/TrainValDataset'),
    'Military_Test': os.path.abspath('Formated Datsets/Military_Dataset/TestDataset'),
}

EXPERIMENT_MAPPING = {
    'M': ['Military'],
    'C1': ['CAMO'],
    'C2': ['COD10K'],
    'MC1': ['Military', 'CAMO'],
    'MC2': ['Military', 'COD10K'],
    'MCC': ['Military', 'CAMO', 'COD10K'],
    'MC1_GHOSTNET': ['Military', 'CAMO'],
    'MC1_SHUFFLENET': ['Military', 'CAMO'],
}


class CamouflageDataset(data.Dataset):
    """
    Multi-domain Camouflaged Object Detection Dataset loader.
    Pairs images and GT masks reliably across single or multiple combined datasets.
    """
    def __init__(self, dataset_dirs, trainsize=352, is_train=True):
        self.trainsize = trainsize
        self.is_train = is_train
        self.pairs = []

        if isinstance(dataset_dirs, str):
            dataset_dirs = [dataset_dirs]

        for d in dataset_dirs:
            img_dir = os.path.join(d, 'Images')
            gt_dir = os.path.join(d, 'GT')
            if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
                raise FileNotFoundError(f"Missing Images or GT in {d}")

            gt_files = os.listdir(gt_dir)
            gt_lookup = {os.path.splitext(f)[0]: os.path.join(gt_dir, f) for f in gt_files}

            for img_name in sorted(os.listdir(img_dir)):
                stem, ext = os.path.splitext(img_name)
                if ext.lower() in ['.jpg', '.png', '.jpeg']:
                    if stem in gt_lookup:
                        self.pairs.append((os.path.join(img_dir, img_name), gt_lookup[stem]))
                    else:
                        print(f"Warning: No GT match found for image {img_name} in {d}")

        self.size = len(self.pairs)
        self.img_norm = transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img_path, gt_path = self.pairs[index]
        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(gt_path, 'rb') as f:
            gt = Image.open(f).convert('L')

        # Joint augmentations
        if self.is_train:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize
        image = image.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        gt = gt.resize((self.trainsize, self.trainsize), Image.NEAREST)

        # To Tensor
        img_tensor = transforms.ToTensor()(image)
        img_tensor = self.img_norm(img_tensor)

        gt_tensor = transforms.ToTensor()(gt)
        gt_tensor = (gt_tensor > 0.5).float()

        return img_tensor, gt_tensor


def get_composite_loader(exp_name_or_list, batchsize=8, trainsize=352, shuffle=True,
                         num_workers=2, pin_memory=True):
    """
    Factory to retrieve a DataLoader for any of the 6 combination experiments:
    M, C1, C2, MC1, MC2, MCC (or a custom list of dataset keys/paths).
    """
    if isinstance(exp_name_or_list, str) and exp_name_or_list.upper() in EXPERIMENT_MAPPING:
        keys = EXPERIMENT_MAPPING[exp_name_or_list.upper()]
        dirs = [DATASET_PATHS[k] for k in keys]
    elif isinstance(exp_name_or_list, list):
        dirs = [DATASET_PATHS.get(k, k) for k in exp_name_or_list]
    else:
        raise ValueError(f"Unknown experiment configuration: {exp_name_or_list}")

    dataset = CamouflageDataset(dirs, trainsize=trainsize, is_train=True)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    return loader, dataset


if __name__ == '__main__':
    for exp in ['M', 'C1', 'C2', 'MC1', 'MC2', 'MCC']:
        _, ds = get_composite_loader(exp, batchsize=8, num_workers=0)
        print(f"Exp {exp:4s}: {len(ds)} training samples loaded successfully.")
