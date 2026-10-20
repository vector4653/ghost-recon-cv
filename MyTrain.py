import os
import time
import json
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib.Network_Res2Net_GRA_NCD import Network
from utils.composite_dataloader import get_composite_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter


def structure_loss(pred, mask):
    """
    Multiscale structure loss combining weighted BCE and weighted IoU (F3Net-style).
    Emphasizes object contours and camouflaged boundaries.
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, opt, loss_meter):
    model.train()
    loss_record = AvgMeter()
    epoch_step = len(train_loader)
    
    current_lr = adjust_lr(optimizer, opt.lr, epoch, decay_rate=opt.decay_rate, decay_epoch=opt.decay_epoch)
    
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{opt.epoch}] LR: {current_lr:.1e}", leave=False)
    for i, (images, gts) in enumerate(pbar):
        images = images.cuda()
        gts = gts.cuda()

        optimizer.zero_grad()
        # Forward pass: 4 multiscale predictions
        S_g, S_5, S_4, S_3 = model(images)

        # Multiscale structure loss: L = L_g + L_5 + L_4 + L_3
        loss_g = structure_loss(S_g, gts)
        loss_5 = structure_loss(S_5, gts)
        loss_4 = structure_loss(S_4, gts)
        loss_3 = structure_loss(S_3, gts)
        loss = loss_g + loss_5 + loss_4 + loss_3

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        loss_record.update(loss.item(), images.size(0))
        pbar.set_postfix({
            'L_tot': f"{loss.item():.4f}",
            'Avg': f"{loss_record.avg:.4f}",
            'L3': f"{loss_3.item():.3f}"
        })

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch:02d}/{opt.epoch:02d} | Train Loss: {loss_record.avg:.4f} | LR: {current_lr:.1e}")
    return loss_record.avg


def main():
    parser = argparse.ArgumentParser(description="SINet-V2 Multi-Domain Training (Hwang & Ma 2024 Replication)")
    parser.add_argument('--exp', type=str, default='M', choices=['M', 'C1', 'C2', 'MC1', 'MC2', 'MCC', 'MC1_ghostnet', 'MC1_shufflenet'],
                        help='Experiment identifier (Table 6)')
    parser.add_argument('--epoch', type=int, default=30, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch size for training')
    parser.add_argument('--trainsize', type=int, default=352, help='Input image resolution')
    parser.add_argument('--clip', type=float, default=0.5, help='Gradient clipping bound')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='LR decay factor')
    parser.add_argument('--decay_epoch', type=int, default=15, help='Epoch interval for LR decay')
    parser.add_argument('--load', type=str, default='./snapshot/SINet_V2/Net_epoch_best.pth',
                        help='Checkpoint to initialize/fine-tune from (set to empty for ImageNet scratch)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--backbone', type=str, default='res2net', choices=['res2net', 'ghostnet', 'shufflenet'], help='Backbone architecture')
    opt = parser.parse_args()

    if opt.save_dir is None:
        opt.save_dir = f"./snapshot/Exp_{opt.exp}/"
    os.makedirs(opt.save_dir, exist_ok=True)

    print("=================================================================")
    print(f"  Ghost-Recon-CV: Training Experiment '{opt.exp}'")
    print(f"  Target Checkpoints : {opt.save_dir}")
    print(f"  Batch Size / Epochs: {opt.batchsize} / {opt.epoch}")
    print(f"  Initial LR / Decay : {opt.lr} (decay {opt.decay_rate} at epoch {opt.decay_epoch})")
    print("=================================================================")

    # Auto-infer backbone if experiment name implies it
    if 'ghostnet' in opt.exp.lower():
        opt.backbone = 'ghostnet'
    elif 'shufflenet' in opt.exp.lower():
        opt.backbone = 'shufflenet'

    # If lightweight backbone and load was left as default Res2Net checkpoint, reset to None
    if opt.backbone != 'res2net' and opt.load == './snapshot/SINet_V2/Net_epoch_best.pth':
        opt.load = None

    # 1. Initialize Network
    if opt.backbone == 'res2net':
        from lib.Network_Res2Net_GRA_NCD import Network
        model = Network(imagenet_pretrained=False)
    elif opt.backbone == 'ghostnet':
        from lib.Network_GhostNet_GRA_NCD import Network
        model = Network(imagenet_pretrained=True if not opt.load else False)
    elif opt.backbone == 'shufflenet':
        from lib.Network_ShuffleNetV2_GRA_NCD import Network
        model = Network(imagenet_pretrained=True if not opt.load else False)

    if opt.load and os.path.exists(opt.load):
        print(f"Loading pretrained weights from: {opt.load}")
        state_dict = torch.load(opt.load, map_location='cpu')
        model.load_state_dict(state_dict)
        print("[+] Pretrained weights successfully loaded.")
    else:
        print(f"Initializing {opt.backbone} model (pretrained on ImageNet).")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model placed on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Build Multi-Domain Composite DataLoader
    train_loader, train_dataset = get_composite_loader(
        exp_name_or_list=opt.exp,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    print(f"[+] Composite dataset ready: {len(train_dataset)} training image/mask pairs ({len(train_loader)} batches/epoch).")

    # 3. Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # 4. Training Loop
    best_loss = float('inf')
    loss_history = []
    loss_meter = AvgMeter()

    start_time = time.time()
    for epoch in range(1, opt.epoch + 1):
        epoch_loss = train(train_loader, model, optimizer, epoch, opt, loss_meter)
        loss_history.append({'epoch': epoch, 'loss': float(epoch_loss)})

        # Save checkpoint periodically
        if epoch % 5 == 0 or epoch == opt.epoch:
            ckpt_path = os.path.join(opt.save_dir, f"Net_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

        # Track best epoch
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_ckpt = os.path.join(opt.save_dir, "Net_epoch_best.pth")
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [*] New best model saved (loss: {best_loss:.4f})")

        # Save loss history json & csv
        with open(os.path.join(opt.save_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f, indent=2)
        import csv
        with open(os.path.join(opt.save_dir, "loss_history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for h in loss_history:
                writer.writerow([h['epoch'], h['loss']])

    total_time = time.time() - start_time
    print("=================================================================")
    print(f"Training Complete for Exp '{opt.exp}' in {total_time/60:.2f} minutes.")
    print(f"Best Loss: {best_loss:.4f} | Best weights saved to: {os.path.join(opt.save_dir, 'Net_epoch_best.pth')}")
    print("=================================================================")


if __name__ == '__main__':
    main()
