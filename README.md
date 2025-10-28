## Driver Drowsiness Detection (Lightweight‐CPU)

### Overview

This project is a lightweight real-time **driver drowsiness detection system** using:

* **MediaPipe Face Detection** (fast CPU-based face detector)
* **Quantized MobileNetV3-Small** classifier (binary: Awake / Drowsy)
* **Live Webcam Input**
* Automatic alert warning when drowsiness symptoms are detected

It is optimized to run on **low-spec laptops with CPU only** - no GPU required

> The included model is pretrained only on **synthetic placeholder images**
> ➜ Replace them with real images and retrain the model for actual use.

---

### Project Structure

```
project/
│
├── data/
│   ├── awake/        # synthetic images (replace later)
│   └── drowsy/       # synthetic images (replace later)
│
├── drowsiness_model.py       # train + quantize MobileNetV3
├── run_detection.py          # main webcam detection script
├── generate_synthetic_images.py  # creates synthetic placeholder dataset
│
├── drowsiness_quantized.pth  # sample model (not real performance)
└── README.md
```

---

### 🛠 Requirements

| Library               | Purpose                   |
| --------------------- | ------------------------- |
| OpenCV                | Web-camera feed + display |
| MediaPipe             | Face detection            |
| PyTorch + TorchVision | Model + quantization      |
| PIL                   | Image Format Handling     |

Install with:

```bash
pip install opencv-python mediapipe torch torchvision pillow
```

---

### How to Use

#### 1. Generate synthetic dataset (optional)

Only if you want to regenerate the placeholders:

```bash
python generate_synthetic_images.py
```

#### 2. Train & Quantize the Model

```bash
python drowsiness_model.py
```

This creates:

```
drowsiness_quantized.pth
```

#### 3. Run Real-Time Webcam Detection

```bash
python run_detection.py
```

---

### Replacing the Fake Dataset

To make the tool useful:

1. Delete synthetic images
2. Add real labeled images:

```
data/awake/    <- eyes open, alert faces
data/drowsy/   <- eyes closed, yawning, head drooping
```

3. Retrain:

```bash
python drowsiness_model.py
```

---

### ⚠️ Disclaimer

This project is **not medically accurate** in its current state.
It should not be used for **vehicle safety** without rigorous evaluation.

---

### Roadmap for Improvement

- ✔ Eye landmark-based blink detection
- ✔ Head-pose estimation for nodding
- ✔ Audio alert system
- ✔ Better dataset & trained model
- ✔ ONNX export for mobile deployment

---