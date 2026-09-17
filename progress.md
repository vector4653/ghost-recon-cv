# Ghost-Recon-CV: Project Progress & Execution Roadmap
**Replication & Extension of Hwang & Ma (2024) — Camouflaged Object Detection (COD) with SINet-V2**

---

## 📌 Executive Summary
This project investigates military camouflaged target segmentation using deep learning. We adapt **SINet-V2** to evaluate domain transfer between natural camouflage (animals/insects in **CAMO** and **COD10K**) and artificial military camouflage (**ACD1K** and **Military Personnel Dataset**). The ultimate goal is replicating and extending the 6-experiment combination methodology (Table 6 in Hwang & Ma, 2024) to assess how multi-domain camouflage training impacts soldier detection in tactical environments.

---

## 🏆 Completed Milestones

### ✅ Week 1: Data Audit & Standardization
- [x] **Data Auditing**: Verified binary pixel-level ground truth masks for **ACD1K** and **Military Personnel Dataset** (confirming true image segmentation; no SAM pseudo-labeling fallback needed).
- [x] **Standardization**: Structured all datasets into standard `TrainValDataset/` and `TestDataset/` pairs with `Images/` and `GT/` subdirectories.
- [x] **3:1 Split**: Partitioned military dataset into 1,498 training pairs (75%) and 580 test pairs (25%).
- [x] **Benchmark Acquisition**: Staged CAMO (1,000 train / 250 test) and COD10K (6,000 train / 4,000 test).
- [x] **Statistical EDA**: Generated dataset analysis reports mirroring Figures 6–8 of the baseline paper (object size distributions, foreground/negative sample ratios, image resolutions).

### ✅ Week 2: Zero-Shot Baseline Evaluation
- [x] **Pretrained Baseline**: Evaluated off-the-shelf SINet-V2 (trained on generic COD datasets) directly on the unseen Military Test split (580 images).
- [x] **Metric Extraction**: Benchmarked via PySODMetrics:
  - **Structure Measure ($S_\alpha$):** `0.7647` (Strong global layout recognition)
  - **Enhanced-Alignment Measure ($E_\phi$):** `0.8061` (Solid foreground/background separation)
  - **Weighted F-measure ($F_\beta^w$):** `0.6036` (Identifies the domain gap: blurry boundaries on uniforms)
  - **Mean Absolute Error (MAE):** `0.0593` (Low pixel discrepancy supported by negative samples)
- [x] **Code & Pipeline**: Committed [`MyTesting.py`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/MyTesting.py) and [`evaluation.py`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/evaluation.py) to repository.

---

## 🚀 Remaining Project Phases

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Environment & Codebase Alignment (RTX 4050 Laptop) │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Week 3 - The 6 Combination Experiments (Table 6)   │
│ (Exp M ➔ Exp C1 ➔ Exp C2 ➔ Exp MC1 ➔ Exp MC2 ➔ Exp MCC)     │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Qualitative Visualizations & Error Analysis        │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Week 4 - Report, Presentation & Interactive Demo   │
└─────────────────────────────────────────────────────────────┘
```

---

### 🔹 Phase 1: Environment & Codebase Alignment (RTX 4050 Setup)
**Objective**: Prepare the Windows 11 / RTX 4050 (6GB VRAM) environment for GPU-accelerated training.

- [x] **1.1 Python & CUDA Setup**
  - Installed supported Python version (Python 3.11.9).
  - Created isolated `.venv` environment.
  - Installed PyTorch 2.5.1 with native CUDA 12.1 support (`cu121`).
  - Installed required libraries: `torchvision`, `pysodmetrics`, `opencv-python`, `pillow`, `matplotlib`, `scipy`, `timm`, `tqdm`.
  - Verified CUDA detection: **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)** active.
- [x] **1.2 SINet-V2 Core Modules**
  - Integrated `lib/` (`Network_Res2Net_GRA_NCD.py`, `Res2Net_v1b.py`) and `utils/` (`data_val.py`, `utils.py`) into the workspace.
  - Created zero-copy NTFS directory junctions for `Dataset/TestDataset` and `Dataset/TrainValDataset`.
- [x] **1.3 Pretrained Checkpoint Acquisition**
  - Staged valid `Net_epoch_best.pth` (103.31 MB, 880 parameter tensors) into [`./snapshot/SINet_V2/`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/snapshot/SINet_V2).
- [x] **1.4 CUDA Sanity Check**
  - Updated device mappings in [`MyTesting.py`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/MyTesting.py) to dynamic CUDA detection (`cuda` on RTX 4050).
  - Executed full 580-image inference on the RTX 4050 GPU in seconds with zero OOM errors.
  - Verified PySODMetrics baseline reproduction ($S_\alpha$: `0.7647`, $E_\phi$: `0.8061`, $F_\beta^w$: `0.6037`, MAE: `0.0593`).

---

### 🔹 Phase 2: Week 3 — The 6 Combination Experiments
**Objective**: Replicate Table 6 of Hwang & Ma (2024). Train models across 6 distinct dataset splits and evaluate every model on the standardized **Military Test split (580 images)**.

| Experiment | Training Dataset(s) | Image Pairs | Device / Batch | Epochs | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Off-the-shelf SINet-V2 | N/A | Apple MPS / 1 | 0 | ✅ **Done** |
| **Exp M** | Military Only (`TrainValDataset`) | 1,498 | RTX 4050 / 8 | 30 | ✅ **Done** |
| **Exp C1** | CAMO Only (`TrainValDataset`) | 1,000 | RTX 4050 / 8 | 30 | ✅ **Done** |
| **Exp C2** | COD10K Only (`TrainValDataset`) | 6,000 | RTX 4050 / 8 | 30 | ✅ **Done** |
| **Exp MC1** | Military + CAMO | 2,498 | RTX 4050 / 8 | 30 | ✅ **Done** |
| **Exp MC2** | Military + COD10K | 7,498 | RTX 4050 / 8 | 30 | ✅ **Done** |
| **Exp MCC** | Military + CAMO + COD10K | 8,498 | RTX 4050 / 8 | 30 | ✅ **Done** |

#### Action Items:
- [x] **2.1 Build Combined Dataset Loaders / Directories**
  - Created [`utils/composite_dataloader.py`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/utils/composite_dataloader.py) with dynamic multi-domain loading for all 6 experiment splits (`M`, `C1`, `C2`, `MC1`, `MC2`, `MCC`).
- [x] **2.2 Create CUDA Training Script (`MyTrain.py`)**
  - Implemented multiscale structure loss ($L = L_g + L_5 + L_4 + L_3$, weighted BCE + weighted IoU).
  - Configured Adam optimizer ($lr=1e-4$, decay by 0.1 at epoch 15), gradient clipping (`0.5`), periodic checkpointing every 5 epochs, and best checkpoint retention.
  - Verified end-to-end execution on NVIDIA GeForce RTX 4050 Laptop GPU (~48s/epoch).
  - Created automated orchestrator [`run_experiment.py`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/run_experiment.py) linking training, inference, and PySODMetrics evaluation.
- [x] **2.3 Run Experiment M (Military Only Fine-Tuning)**
  - Validated training loop and loss convergence (loss dropped from 7.68 to 1.76).
  - Executed inference on Military Test set (580 images) saved to [`./res/Exp_M/`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/res/Exp_M).
  - Benchmarked with PySODMetrics: $S_\alpha = 0.8638$, $E_\phi = 0.9046$, $F_\beta^w = 0.7718$, $\text{MAE} = 0.0348$.
- [x] **2.4 Run Nature Camouflage Baselines (Exp C1 & C2)**
  - [x] **Exp C1 (CAMO Only)**: Trained 30 epochs on 1,000 CAMO pairs. Results on Military Test (580): $S_\alpha = 0.7572$, $E_\phi = 0.8045$, $F_\beta^w = 0.5886$, $\text{MAE} = 0.0538$.
  - [x] **Exp C2 (COD10K Only)**: Trained 30 epochs on 6,000 pairs. Results on Military Test (580): $S_\alpha = 0.6223$, $E_\phi = 0.6616$, $F_\beta^w = 0.3541$, $\text{MAE} = 0.1007$.
- [x] **2.5 Run Fusion Experiments (Exp MC1, MC2, MCC)**
  - [x] **Exp MC1 (Military + CAMO)**: Trained 30 epochs on 2,498 pairs. Results on Military Test (580): $S_\alpha = 0.8675$, $E_\phi = 0.9095$, $F_\beta^w = 0.7773$, $\text{MAE} = 0.0322$ (*Top performing model across all 4 metrics!*).
  - [x] **Exp MC2 (Military + COD10K)**: Trained 30 epochs on 7,498 pairs. Results on Military Test (580): $S_\alpha = 0.8460$, $E_\phi = 0.8765$, $F_\beta^w = 0.7276$, $\text{MAE} = 0.0423$.
  - [x] **Exp MCC (Military + CAMO + COD10K)**: Trained 30 epochs on 8,498 pairs. Results on Military Test (580): $S_\alpha = 0.8518$, $E_\phi = 0.8802$, $F_\beta^w = 0.7379$, $\text{MAE} = 0.0408$.
- [x] **2.6 Compile Metric Results Matrix**
  - Populated the full Table 6 comparison:
    $$S_\alpha \uparrow \quad|\quad E_\phi \uparrow \quad|\quad F_\beta^w \uparrow \quad|\quad \text{MAE} \downarrow$$

---

### 🔹 Phase 3: Qualitative Analysis & Visualization
**Objective**: Produce presentation-ready visual evidence and error case diagnostics.

- [x] **3.1 Visual Comparison Grid**
  - Generated high-resolution side-by-side comparative panels across 6 challenging operational scenarios (snipers, woodland foliage, prone operators, multi-camo):
    `[Original RGB] | [Ground Truth Mask] | [Zero-Shot Baseline] | [Exp M] | [Exp C1] | [Exp C2] | [Exp MC1] | [Exp MCC]`.
  - Saved to [`qualitative_analysis/visual_comparison_grid.png`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/qualitative_analysis/visual_comparison_grid.png) and [`qualitative_analysis/tactical_overlay_comparison.png`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/qualitative_analysis/tactical_overlay_comparison.png).
- [x] **3.2 Boundary Sharpness & Contour Analysis**
  - Computed Boundary IoU across all 580 test images within a narrow 4-pixel contour band.
  - Results: **Exp MC1 achieved highest boundary precision (0.3437)** compared to Exp M (0.3372), Exp MCC (0.3199), and Baseline (0.2332).
  - Saved visualization to [`qualitative_analysis/boundary_sharpness_analysis.png`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/qualitative_analysis/boundary_sharpness_analysis.png).
- [x] **3.3 Negative Distractor & False Positive Diagnostics**
  - Quantified False Positive Rate (FPR) and mean leakage intensity across ~90.3% non-target tactical background pixels across all 580 test images.
  - Multi-domain fusion (**Exp MC1**) regularized the network, dropping false alarms by ~23% compared to military-only fine-tuning (**1.55% vs 2.02% FPR**; background leakage **0.0167 vs 0.0213**).
  - Saved diagnostic figures to [`qualitative_analysis/false_alarm_background_diagnostics.png`](file:///C:/Users/ritvi/Downloads/Apps/ghost-recon-cv/qualitative_analysis/false_alarm_background_diagnostics.png).

---

### 🔹 Phase 4: Week 4 — Report, Presentation & Interactive Demo
**Objective**: Final synthesis, documentation, and interactive presentation tools.

- [ ] **4.1 Final Research Paper / Report**
  - Structure according to IEEE / Hwang & Ma format:
    1. *Introduction & Motivation* (OODA loop, tactical vision)
    2. *Related Work* (COD evolution, SINet, MCAM)
    3. *Dataset Architecture & Statistical Analysis* (ACD1K, Military Personnel, CAMO, COD10K)
    4. *Methodology* (Res2Net backbone, Neighbor Connection Decoder, Group Reversal Attention)
    5. *Experimental Setup & 6-Model Combination Ablation*
    6. *Results & Discussion* (Comparative tables, cross-domain transfer analysis)
    7. *Conclusion & Future Work*
- [ ] **4.2 Interactive Inference Demo (Streamlit / Gradio)**
  - Build a lightweight local web app to upload an arbitrary tactical image and see real-time camouflage mask overlays from the fine-tuned model.
- [ ] **4.3 Presentation Slide Deck**
  - Visual slides summarizing dataset statistics, combination ablation tables, and qualitative prediction overlays.

---

## 📊 Results Tracking Table

| Experiment | Training Dataset | Test Dataset | $S_\alpha \uparrow$ | $E_\phi \uparrow$ | $F_\beta^w \uparrow$ | $\text{MAE} \downarrow$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Off-the-shelf COD | Military Test (580) | **0.7647** | **0.8061** | **0.6036** | **0.0593** |
| **Exp M** | Military Train (1,498) | Military Test (580) | **0.8638** | **0.9046** | **0.7718** | **0.0348** |
| **Exp C1** | CAMO Train (1,000) | Military Test (580) | **0.7572** | **0.8045** | **0.5886** | **0.0538** |
| **Exp C2** | COD10K Train (6,000) | Military Test (580) | **0.6223** | **0.6616** | **0.3541** | **0.1007** |
| **Exp MC1** | Military + CAMO (2,498) | Military Test (580) | **0.8675** | **0.9095** | **0.7773** | **0.0322** |
| **Exp MC2** | Military + COD10K (7,498) | Military Test (580) | **0.8460** | **0.8765** | **0.7276** | **0.0423** |
| **Exp MCC** | Military + CAMO + COD10K (8,498) | Military Test (580) | **0.8518** | **0.8802** | **0.7379** | **0.0408** |
