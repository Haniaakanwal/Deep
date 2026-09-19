# Receipt Border Detection

Detects and crops individual receipts from a single photo containing multiple receipts — works across different backgrounds, lighting conditions, and receipt templates without any custom training.


## Approach

### 1. Classical image processing — tried, rejected
Grayscale → threshold (Otsu / adaptive) → morphological closing → contour detection → rotated bounding box → perspective warp.

**Why it failed:** the test photo has non-uniform lighting — background brightness ranges from ~107 (corners) to ~236 (center glare), overlapping the receipts' own paper brightness (~197–240). No single global threshold could separate paper from background everywhere. Depending on parameters it either merged everything into one blob or fragmented single receipts into pieces.

### 2. Deep neural network — final solution
Uses **Segment Anything Model (SAM)**, ViT-B checkpoint, for zero-shot instance segmentation — no labelled training data needed, generalizes to unseen photos out of the box.

## Pipeline

1. **Mask proposal** — `SamAutomaticMaskGenerator`, 32×32 point grid, IoU/stability thresholds 0.90 / 0.95
2. **Size filter** — keep masks between 1.5%–35% of image area
3. **Deduplication** — drop masks with IoU ≥ 0.5 vs. an already-kept mask, or masks sitting >85% inside one (removes duplicates and internal sub-elements like stamp boxes)
4. **Outlier filter** — drop masks smaller than 35% of the median receipt area
5. **Content check** — drop crops with grayscale std < 15 (blank background slivers)
6. **Shape filter** — drop boxes with aspect ratio < 0.15 (thin edge shadows/folds)
7. **Output** — save each surviving crop + an annotated overview image

## Results

| Test photo | Result |
|---|---|
| Original photo — 5 receipts, uneven lighting | 5 / 5 correct |
| Synthetic template photo — 6 receipts, touching pair | 6 / 6 correct |
| Mixed pharmacy/grocery/receipt photo — 4 receipts | 4 / 4 correct |

## Usage (Google Colab)

```python
# 1. Upload photo
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# 2. Install deps
!pip install opencv-python-headless numpy segment-anything -q
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Then run `receipt_detector_colab.py` cells 3–5 (detect, view, download) — see that file for the full pipeline code.

## Limitations

- Receipts touching with zero visible gap may still merge into one mask in rare cases
- Very low-resolution or blurry photos reduce mask quality
- Filter thresholds were tuned on the 3 test photos above; very different receipt counts/scales may need re-tuning

## Requirements

```
opencv-python-headless
numpy
segment-anything
torch
```

GPU (e.g. Colab T4) strongly recommended — SAM is slow on CPU.
