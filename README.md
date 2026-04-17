---
license: cc-by-4.0
language:
  - es
tags:
  - license-plate-recognition
  - ocr
  - alpr
  - costa-rica
  - latin-america
  - onnx
  - ctc
  - crnn
library_name: onnxruntime
pipeline_tag: image-to-text
datasets: []
metrics:
  - accuracy
model-index:
  - name: wink-lpr-ocr-cr
    results:
      - task:
          type: image-to-text
          name: License Plate OCR
        metrics:
          - type: accuracy
            value: 98.4
            name: Plate Accuracy
          - type: accuracy
            value: 97.8
            name: Character Accuracy
---

<div align="center">
<img src="https://www.wink.co/img/winklogo.png" width="200" alt="WINK Streaming">

# WINK LPR — Costa Rica Plate OCR

**98.4% plate accuracy** · 4.4MB ONNX · CTC architecture · Real-time inference

[![WINK Streaming](https://img.shields.io/badge/WINK-Streaming-blue?style=flat-square)](https://www.wink.co)
[![License](https://img.shields.io/badge/License-CC--BY--4.0-green?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)
[![ONNX](https://img.shields.io/badge/Format-ONNX-orange?style=flat-square)](https://onnx.ai/)

</div>

## Overview

A high-accuracy OCR model for reading Costa Rica license plates, developed by [WINK Streaming](https://www.wink.co) as part of a production LPR (License Plate Recognition) system.

The model reads cropped plate images and outputs the plate text. It handles both daytime color and nighttime IR/grayscale camera feeds.

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | CNN (5 conv blocks) + BiLSTM (128 hidden × 2 layers) + CTC decoder |
| **Parameters** | 1,117,170 |
| **Input** | `[batch, 96, 192, 1]` — grayscale, NHWC, uint8 (0-255) |
| **Output** | `[batch, 48, 38]` — 48 CTC timesteps, 38 classes |
| **Alphabet** | `0-9`, `A-Z`, `_` (pad) + CTC blank (index 37) |
| **ONNX size** | 4.4 MB |
| **ONNX opset** | 18 (IR version 8) |
| **Inference** | ~5ms CPU, ~2ms GPU |

## Performance

| Metric | Score |
|--------|-------|
| **Plate accuracy** (exact match) | **98.4%** |
| **Character accuracy** | **97.8%** |
| Validation set | 9,074 crops |
| Training set | 81,666 crops from 9,727 labeled detections |

Evaluated on held-out 10% split of production data from Axis & Hikvision cameras (2304×1296, color day / IR night).

## Supported Plate Formats

- **Standard:** 3 letters + 3 digits (e.g. `AAP096`)
- **Numeric:** All digits (e.g. `672625`)
- **Government:** 2-3 letters + 2 digits
- **Motorcycle:** All digits, 2-row format
- **Diplomat/Electric:** Various formats
- All 36 alphanumeric characters are valid (including `O`)

## Quick Start

```python
import numpy as np
import cv2
import onnxruntime as ort

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
BLANK = 37  # CTC blank token

def load_and_preprocess(image_path, h=96, w=192):
    """Load image, convert to grayscale, resize with aspect-preserving padding."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    src_h, src_w = img.shape[:2]
    scale = min(h / src_h, w / src_w)
    new_h, new_w = int(src_h * scale), int(src_w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w), 128, dtype=np.uint8)
    y_off, x_off = (h - new_h) // 2, (w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def ctc_decode(logits):
    """Greedy CTC decode: collapse repeats, remove blanks."""
    indices = np.argmax(logits, axis=-1)
    chars = []
    prev = -1
    for idx in indices:
        if idx != prev and idx != BLANK and idx < len(ALPHABET):
            chars.append(ALPHABET[idx])
        prev = idx
    return ''.join(chars)

def predict(session, image_path):
    """Run OCR on a plate crop image."""
    gray = load_and_preprocess(image_path)
    blob = gray.reshape(1, 96, 192, 1).astype(np.uint8)
    logits = session.run(None, {"input": blob})[0][0]  # [48, 38]
    return ctc_decode(logits)

# Usage
session = ort.InferenceSession("wink-lpr-ocr-cr.onnx")
plate_text = predict(session, "plate_crop.jpg")
print(f"Plate: {plate_text}")
```

## Input Preprocessing

1. Convert to grayscale (single channel)
2. Resize with aspect-preserving padding to 96×192
3. Pad with gray (value 128)
4. Input as uint8 `[batch, 96, 192, 1]` NHWC format — no normalization needed, the model handles it internally

## Output Decoding

The model outputs `[batch, 48, 38]` logits — 48 CTC timesteps over 38 classes:
- Indices 0-9: digits `0-9`
- Indices 10-35: letters `A-Z`
- Index 36: pad/underscore `_`
- Index 37: CTC blank token

**Greedy CTC decode:** Take argmax per timestep, collapse consecutive identical indices, remove blank tokens.

## Training

- **Architecture:** CNN feature extractor (5 blocks: 1→48→96→128→128 channels) with max-pooling that preserves width for CTC alignment, followed by BiLSTM (128 hidden, 2 layers, bidirectional) and linear classifier
- **Loss:** CTC loss + confusion-pair auxiliary penalty for commonly confused characters in IR mode (3/7, 8/0, 1/7, etc.)
- **Data:** 9,727 human-verified labels from production deployment (dual cameras, day + night)
- **Augmentation:** Light (±3.5° rotation, ±25 brightness, 0.85-1.15× contrast)
- **Training:** 200 epochs, batch 64, AdamW with cosine LR schedule, early stopping (patience 40)
- **Hardware:** NVIDIA RTX 3060 12GB

The model was trained from scratch on hard labels (no knowledge distillation). Training data includes both positive examples (verified plate crops) and negative examples (non-plate crops like logos, signs) to teach the model to abstain on non-plate input.

## Deployment Notes

- Compatible with ONNX Runtime (CPU and GPU providers)
- NHWC input format — no channel-first conversion needed
- Works with .NET OnnxRuntime (IR version 8 for compatibility)
- Designed for cropped plate images — pair with a plate detector (e.g. YOLO) for end-to-end ALPR
- Handles both color (daytime) and IR/grayscale (nighttime) crops

## Limitations

- Trained specifically on Costa Rica plates — accuracy on other countries will be lower
- Best performance on crops from fixed surveillance cameras (similar to training distribution)
- Character confusion pairs in IR/night mode: 3↔7, 8↔0, 1↔7, 0↔D, 5↔S
- Requires a separate plate detection model to crop plates from full frames

## Citation

If you use this model, please cite:

```bibtex
@misc{wink-lpr-ocr-cr-2026,
  title={WINK LPR OCR — Costa Rica License Plate Recognition},
  author={WINK Streaming},
  year={2026},
  url={https://www.wink.co},
  note={98.4\% plate accuracy CTC-CRNN model}
}
```

## License

CC-BY-4.0 — free for commercial and non-commercial use with attribution.

## About WINK Streaming

[WINK Streaming](https://www.wink.co) builds intelligent video infrastructure — from camera ingestion and AI-powered analytics to archival and playback. This model is part of our production LPR system.

---

<div align="center">
<sub>Built by <a href="https://www.wink.co">WINK Streaming</a></sub>
</div>
