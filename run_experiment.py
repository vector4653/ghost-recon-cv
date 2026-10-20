import os
import sys
import argparse
import subprocess
import json

def run_cmd(cmd):
    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run complete Train -> Test -> Evaluate pipeline for an experiment.")
    parser.add_argument('--exp', type=str, required=True, choices=['M', 'C1', 'C2', 'MC1', 'MC2', 'MCC', 'MC1_ghostnet', 'MC1_shufflenet'],
                        help='Experiment identifier (Table 6)')
    parser.add_argument('--epoch', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--skip_train', action='store_true', help='Skip training if checkpoint already exists')
    parser.add_argument('--backbone', type=str, default='res2net', choices=['res2net', 'ghostnet', 'shufflenet'], help='Backbone architecture')
    opt = parser.parse_args()

    if 'ghostnet' in opt.exp.lower():
        opt.backbone = 'ghostnet'
    elif 'shufflenet' in opt.exp.lower():
        opt.backbone = 'shufflenet'

    python_bin = sys.executable
    ckpt_dir = f"./snapshot/Exp_{opt.exp}/"
    best_pth = os.path.join(ckpt_dir, "Net_epoch_best.pth")
    res_dir = f"./res/Exp_{opt.exp}/"

    # Step 1: Train
    if opt.skip_train and os.path.exists(best_pth):
        print(f"[+] Found existing checkpoint at {best_pth}, skipping training.")
    else:
        train_cmd = [
            python_bin, "MyTrain.py",
            "--exp", opt.exp,
            "--backbone", opt.backbone,
            "--epoch", str(opt.epoch),
            "--batchsize", str(opt.batchsize),
            "--lr", str(opt.lr),
            "--save_dir", ckpt_dir
        ]
        run_cmd(train_cmd)

    # Step 2: Inference on Military Test split (580 images)
    test_cmd = [
        python_bin, "MyTesting.py",
        "--backbone", opt.backbone,
        "--pth_path", best_pth,
        "--save_path", res_dir
    ]
    run_cmd(test_cmd)

    # Step 3: Metric Evaluation
    eval_cmd = [
        python_bin, "evaluation.py",
        "--gt_dir", "./Dataset/TestDataset/GT/",
        "--pred_dir", res_dir,
        "--title", f"Exp {opt.exp} (Military Test Split)"
    ]
    run_cmd(eval_cmd)

    print(f"\n[+] Pipeline completed successfully for Exp {opt.exp}!")

if __name__ == '__main__':
    main()
