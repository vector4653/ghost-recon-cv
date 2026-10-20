# Tactical Camouflage Edge Efficiency & Lightweight Backbone Analysis

| Model | Params (M) | MACs (G) | FLOPs (G) | Size (MB) | GPU FP32 | GPU FP16 | FPS | CPU (ms) | S_α | E_φ | F_β^w | MAE | B-IoU | FPR | Param Red. | MACs Red. | % F_β^w Retained | Lightweight? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SINet-V2 Res2Net-50 (MC1)** | 26.98 | 12.31 | 24.63 | 103.4 | 28.72ms | 19.99ms | 34.8 | 128.9ms | 0.8675 | 0.9095 | 0.7773 | 0.0322 | 0.3437 | 1.55% | **1.0×** | **1.0×** | **100.0%** | **Baseline Reference** |
| **GhostNet-SINet (MC1)** | 5.92 | 2.26 | 4.51 | 23.1 | 13.55ms | 16.90ms | 73.8 | 57.3ms | 0.8376 | 0.8590 | 0.7078 | 0.0451 | 0.2495 | 2.48% | **4.56×** | **5.46×** | **91.06%** | **NO** |
| **ShuffleNetV2-SINet (MC1)** | 1.59 | 1.19 | 2.37 | 6.4 | 10.56ms | 13.37ms | 94.7 | 50.0ms | 0.8294 | 0.8515 | 0.6929 | 0.0472 | 0.2503 | 2.39% | **16.96×** | **10.38×** | **89.14%** | **NO** |


> 🔍 **Lightweight Criterion Definition:** A model is classified as **Lightweight** if and only if it achieves $\ge 4.0\times$ reduction in parameters AND $\ge 4.0\times$ reduction in MACs compared to the Res2Net-50 baseline while retaining $\ge 95.0\%$ of weighted F-measure ($F_\beta^w$).
