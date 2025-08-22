# AI Vision Assist (Learning Project)

This project explores **real-time computer vision and AI-assisted input** as a learning exercise.  
It uses a **fine-tuned YOLOv8 points model** with **ONNX Runtime + DirectML**, combined with **multithreaded video capture and inference** for low-latency performance.

⚠️ **Disclaimer**  
This project is for **educational purposes only**.  
It was tested in **sandbox environments (e.g., R5Reloaded)** not connected to live servers.  
Do **not** use it to cheat in Apex Legends or any other multiplayer game.

---

## ✨ Features

-   Fine-tuned YOLOv8 points model for fast object detection
-   ONNX Runtime with DirectML backend for any GPU acceleration (on windows)
-   Multithreaded video capture & AI inference for real-time performance

---

## 🚀 Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/ai-vision-assist.git
cd ai-vision-assist
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

3. install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your fine-tuned YOLOv8 .onnx model in the main directory.

2. You can use my export_onnx.py script to export finetuned .pt to .onnx.

3. Adjust settings (input source, capture resolution, thresholds, ...) in the main file in the CONFIG section.

4. Run the main script:

```bash
python main.py
```

## 📚 Learning Goals

-   Experimenting with YOLOv8 model deployment

-   Exploring ONNX + DirectML integration

-   Building multithreaded pipelines for real-time AI

-   Understanding trade-offs in latency vs accuracy
