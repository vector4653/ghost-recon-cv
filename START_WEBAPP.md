# Quick Start Guide: Running the Ghost-Recon-CV Tactical Web App

This guide contains step-by-step instructions to set up and run the **Ghost-Recon-CV** tactical visualization platform on **Windows** and **macOS** (Apple Silicon M-series or Intel), including GPU acceleration, checkpoint detection, and the interactive soldier reveal slider.

---

## Prerequisites

| Tool | Windows | macOS |
| :--- | :--- | :--- |
| **Python** | 3.11 ([python.org](https://www.python.org/downloads/)) — tick *"Add to PATH"* | 3.11 (`brew install python@3.11`) |
| **Node.js** | v18+ LTS ([nodejs.org](https://nodejs.org)) | v18+ LTS (`brew install node`) |
| **Git** | [git-scm.com](https://git-scm.com) | `xcode-select --install` or `brew install git` |
| **Acceleration** | NVIDIA CUDA (RTX GPUs) | Apple Silicon Metal (MPS: M1/M2/M3/M4) |

> [!NOTE]
> **Why port 5050?**
> On macOS, port 5000 is occupied by default by the system **AirPlay Receiver** (`ControlCenter`). The server is therefore configured to run on port **5050** on all platforms so it works seamlessly on both Mac and Windows.

---

## Initial One-Time Setup

Clone the repository and install the dependencies for your operating system:

````carousel
<!-- slide -->
### Windows (PowerShell)

Open PowerShell in the folder where you want the project:

```powershell
# 1. Clone the project
git clone https://github.com/vector4653/ghost-recon-cv.git
cd ghost-recon-cv

# 2. Set server port to 5050
Set-Content server\.env "PORT=5050"

# 3. Create Python virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate
pip install -r ai_service/requirements.txt

# 4. Install Node.js gateway and client dependencies
cd server ; npm install ; cd ..
cd client ; npm install ; cd ..
```

<!-- slide -->
### macOS (Terminal / zsh)

Open Terminal in the folder where you want the project:

```bash
# 1. Clone the project
git clone https://github.com/vector4653/ghost-recon-cv.git
cd ghost-recon-cv

# 2. Set server port to 5050 (avoids macOS AirPlay port conflict)
echo "PORT=5050" > server/.env

# 3. Create Python virtual environment and install dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r ai_service/requirements.txt

# 4. Install Node.js gateway and client dependencies
cd server && npm install && cd ..
cd client && npm install && cd ..
```
````

---

## Starting the Application (3 Terminal Windows)

Open **3 separate terminal windows** (PowerShell on Windows, Terminal / iTerm2 on macOS) in the `ghost-recon-cv` project folder. Keep all three windows running.

### Window 1: AI Model Service (FastAPI)
Loads the SINet-V2 Res2Net-50 camouflage detection models with CUDA on Windows or MPS (Metal Performance Shaders) on Apple Silicon:

- **Windows (PowerShell):**
  ```powershell
  cd C:\Users\ritvi\Downloads\Apps\ghost-recon-cv
  .\.venv\Scripts\python.exe -m uvicorn ai_service.app:app --port 8000
  ```
- **macOS (Terminal):**
  ```bash
  cd path/to/ghost-recon-cv
  source .venv/bin/activate
  python -m uvicorn ai_service.app:app --port 8000
  ```
> **Expected Output:**  
> `INFO: Application startup complete.`  
> `INFO: Uvicorn running on http://127.0.0.1:8000`  
> *(On Windows, device will show `cuda`. On Apple Silicon Mac, device will show `mps`).*

---

### Window 2: API Gateway (Node.js / Express)
Routes requests between the frontend and model service, and serves preset military test split images:

- **Windows (PowerShell):**
  ```powershell
  cd C:\Users\ritvi\Downloads\Apps\ghost-recon-cv\server
  node server.js
  ```
- **macOS (Terminal):**
  ```bash
  cd path/to/ghost-recon-cv/server
  node server.js
  ```
> **Expected Output:**  
> `Server running on port 5050`  
> `MongoDB connection failed. Running in fallback mode without persistence.` *(Normal; MongoDB is optional).*

---

### Window 3: Web App (React + Vite)
Compiles and serves the interactive tactical HUD user interface:

- **Windows (PowerShell):**
  ```powershell
  cd C:\Users\ritvi\Downloads\Apps\ghost-recon-cv\client
  npm run dev
  ```
- **macOS (Terminal):**
  ```bash
  cd path/to/ghost-recon-cv/client
  npm run dev
  ```
> **Expected Output:**  
> `VITE v8.x ready in ... ms`  
> `➜ Local: http://localhost:5173/`

---

## Using the Tactical Viewport

1. Open your browser and navigate to **[http://localhost:5173](http://localhost:5173)**.
2. Confirm the two status dots in the top right are green:
   - **Gateway**: `online`
   - **Model**: `online`
3. Load an image:
   - Click one of the **Sample images** chips (e.g., `Concealed Sniper`, `Woodland Foliage`).
   - Or click **Upload image** to run detection on any custom photo.
4. **Interactive Curtain Slider**:
   - Drag the white curtain handle from right to left across the picture.
   - **Left of slider**: Original camouflaged image.
   - **Right of slider**: Revealed soldier with tactical overlay.
5. **Visualization Layer Selector** (in the sidebar):
   - **Outline + Fill**: Semi-transparent emerald highlight with a sharp 2px glowing green contour tracing the soldier's silhouette.
   - **Outline Only**: Tactical wireframe silhouette showing only the boundary contour.
   - **Fill Only**: Semi-transparent mask overlay without the outer stroke.
6. **Mask Opacity**: Adjust the slider (0% to 100%) to control overlay transparency.
7. **Model Selector**: Switch between training regimes (`Exp MC1 (Winner)`, `Exp M`, `Baseline`, etc.).

---

## Stopping the Application

To stop all services:
- Press <kbd>Ctrl</kbd> + <kbd>C</kbd> (or <kbd>Cmd</kbd> + <kbd>C</kbd> on macOS) in each of the 3 terminal windows.
