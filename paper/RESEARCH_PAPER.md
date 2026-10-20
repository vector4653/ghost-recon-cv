# Camouflaged Object Detection (COD) in Tactical & Military Settings
## Deep Learning Case Study & Replication of Hwang & Ma (2024)

### Abstract
This paper investigates military camouflaged target segmentation using deep learning. We adapt SINet-V2 (Search and Identification Network V2) to evaluate domain transfer between natural camouflage (animals and insects in CAMO and COD10K datasets) and artificial military camouflage. By replicating and extending the 6-experiment combination methodology (Table 6 in Hwang & Ma, 2024), we assess how multi-domain camouflage training impacts soldier detection in tactical environments, specifically addressing the OODA (Observe-Orient-Decide-Act) loop constraints in military operations.

---

### I. Introduction
The OODA loop (Observe-Orient-Decide-Act) is a critical decision-making model in tactical environments. The "Observe" phase is significantly hindered by modern military camouflage, which aims to break down the soldier's silhouette and blend seamlessly into the background. False alarms cost cognitive overload, while missed targets are lethal. 

Recent advancements in computer vision, particularly Camouflaged Object Detection (COD), offer potential solutions. However, most COD models are trained on natural camouflage. This study explores the domain gap between natural and military camouflage, utilizing the SINet-V2 architecture to determine the optimal training regimen for tactical deployment.

---

### II. Related Work
Camouflaged Object Detection has evolved significantly with the introduction of deep learning. SINet-V2, proposed by Fan et al., represents a state-of-the-art approach utilizing a Res2Net-50 backbone. It employs a Neighbor Connection Decoder (NCD) for coarse search and localization, followed by Group Reversal Attention (GRA) for fine boundary identification. This search-and-identify mechanism mimics human visual perception when actively seeking hidden threats.

---

### III. Dataset Architecture & Statistical Analysis
We utilized four distinct datasets:
1. **ACD1K & Military Personnel Dataset**: 1,498 training pairs (75%) and 580 test pairs (25%). These contain artificial tactical camouflage (uniforms, ghillie suits).
2. **CAMO**: 1,000 training pairs featuring natural camouflage (animals in natural habitats).
3. **COD10K**: 6,000 training pairs of highly diverse natural camouflage.

---

### IV. Methodology
Our implementation leverages the SINet-V2 architecture. The model extracts multi-scale features using Res2Net-50. The NCD module aggregates high-level features for initial localization, while the GRA module refines boundaries using a multi-scale structural loss (weighted BCE + weighted IoU) to ensure sharp contour delineation critical for identifying soldier silhouettes.

---

### V. Experimental Setup & Table 6 Ablation
We trained six distinct models (30 epochs each on an NVIDIA RTX 4050) to replicate Table 6 from Hwang & Ma (2024), evaluating on the standardized Military Test split (580 images).

| Experiment | Training Dataset(s) | Pairs | $S_\alpha \uparrow$ | $E_\phi \uparrow$ | $F_\beta^w \uparrow$ | $\text{MAE} \downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Off-the-shelf COD | N/A | 0.7647 | 0.8061 | 0.6036 | 0.0593 |
| **Exp M** | Military Train | 1,498 | 0.8638 | 0.9046 | 0.7718 | 0.0348 |
| **Exp C1** | CAMO | 1,000 | 0.7572 | 0.8045 | 0.5886 | 0.0538 |
| **Exp C2** | COD10K | 6,000 | 0.6223 | 0.6616 | 0.3541 | 0.1007 |
| **Exp MC1**| Military + CAMO | 2,498 | **0.8675** | **0.9095** | **0.7773** | **0.0322** |
| **Exp MC2**| Military + COD10K | 7,498 | 0.8460 | 0.8765 | 0.7276 | 0.0423 |
| **Exp MCC**| Military + CAMO + COD10K | 8,498 | 0.8518 | 0.8802 | 0.7379 | 0.0408 |

---

### VI. Qualitative Diagnostics & Results
**Boundary Sharpness**: We computed the Boundary IoU within a narrow 4-pixel contour band. Exp MC1 achieved the highest boundary precision (0.3437) compared to Exp M (0.3372) and Baseline (0.2332).

**False Alarm Analysis**: In tactical scenarios, false positive rates (FPR) on non-target backgrounds are critical. Multi-domain fusion in Exp MC1 regularized the network, dropping false alarms by ~23% compared to military-only fine-tuning (1.55% vs 2.02% FPR; background leakage 0.0167 vs 0.0213).

**Negative Transfer**: Exp C2 (COD10K) suffered severe negative transfer due to the extreme divergence between wildlife textures and soldier silhouettes, introducing significant noise when fused (Exp MC2).

---

---

### VII. Edge Efficiency & Lightweight Backbone Analysis
In tactical environments (e.g., micro-drones, soldier-worn HUDs), computational constraints impose strict power and memory bounds. To assess the operational trade-off between architectural complexity and camouflage segmentation fidelity, we implemented and evaluated lightweight variants of SINet-V2 using **GhostNet** (`ghostnet_100`) and **ShuffleNetV2** (`shufflenet_v2_x1_0`), keeping the decoder (RFB, NCD, GRA) identical. All models were trained on the winning MC1 regime (2,498 pairs, 30 epochs) and profiled with an input resolution of $1\times 3 \times 352 \times 352$ on an NVIDIA GeForce RTX 4050 Laptop GPU and 14-thread CPU.

| Model | Params (M) | MACs (G) | FLOPs (G) | Size (MB) | GPU FP32 | GPU FP16 | FPS | CPU (ms) | $S_\alpha \uparrow$ | $E_\phi \uparrow$ | $F_\beta^w \uparrow$ | MAE $\downarrow$ | B-IoU $\uparrow$ | FPR $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Res2Net-50 (MC1)** | 26.98 | 12.31 | 24.63 | 103.4 | 28.72ms | 19.99ms | 34.8 | 128.9ms | 0.8675 | 0.9095 | 0.7773 | 0.0322 | 0.3437 | 1.55% |
| **GhostNet-SINet (MC1)** | 5.92 | 2.26 | 4.51 | 23.1 | 13.55ms | 16.90ms | 73.8 | 57.3ms | 0.8376 | 0.8590 | 0.7078 | 0.0451 | 0.2495 | 2.48% |
| **ShuffleNetV2-SINet (MC1)** | 1.59 | 1.19 | 2.37 | 6.4 | 10.56ms | 13.37ms | 94.7 | 50.0ms | 0.8294 | 0.8515 | 0.6929 | 0.0472 | 0.2503 | 2.39% |

#### Key Empirical Findings:
1. **Parameter & Compute Compression**:
   - GhostNet achieves a **4.56× parameter reduction** and **5.46× MAC reduction**, dropping checkpoint size from 103.4 MB to 23.1 MB while achieving 73.8 FPS (2.12× speedup).
   - ShuffleNetV2 achieves an extreme **16.96× parameter reduction** (only 1.59M parameters total) and **10.38× MAC reduction** (1.19 GMACs), fitting within a 6.36 MB binary footprint and running at 94.7 FPS (2.72× speedup).
2. **FLOPs vs. Latency Reality**:
   - Although ShuffleNetV2 has 1.90× fewer MACs than GhostNet, its GPU latency improves by only 1.28× (10.56 ms vs 13.55 ms). This discrepancy is attributed to depthwise convolutions and channel shuffle operations exhibiting low arithmetic intensity and memory-access-cost bottlenecks, whereas GhostNet's point-wise linear operators maintain superior GPU cache utilization.
3. **Detection Fidelity & Boundary Degradation**:
   - Against our objective lightweight certification threshold ($\ge 4\times$ params & MAC reduction with $\ge 95\%$ $F_\beta^w$ retained), neither model certified formally: GhostNet retained 91.06% of $F_\beta^w$ and ShuffleNetV2 retained 89.14%.
   - Camouflaged soldier silhouettes depend on fine-grained contextual correlations to resolve human contours against natural background textures. The compressed backbones exhibited a drop in 4px Boundary IoU from 0.3437 to ~0.250 and higher background leakage (~0.028 vs 0.0167). However, for SWaP-constrained (Size, Weight, and Power) tactical edge systems, ShuffleNetV2 provides an exceptional trade-off.

---

### VIII. Conclusion & Future Work
We demonstrated that fusing natural camouflage (CAMO) with military datasets (Exp MC1) provides optimal regularization, yielding the sharpest target boundaries and lowest false alarm rates. This cross-domain synergy enhances the "Observe" phase of the OODA loop. Future work will focus on deploying this model via Edge AI for real-time UAV integration and tactical C2 dashboards.
