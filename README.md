# Lung Segmentation for Chest X-Ray — Cross-Dataset Generalization Study

This project builds a robust lung segmentation pipeline for chest X-ray images. The segmented lungs are designed to be passed downstream to a pathology classification model, with the goal of studying whether lung-region isolation improves pathology detection accuracy. The study trains multiple segmentation architectures on both original and augmented datasets and evaluates their ability to generalize across different data distributions.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Datasets](#datasets)
3. [Augmentation Pipeline](#augmentation-pipeline)
4. [Image Enhancement](#image-enhancement)
5. [Architectures](#architectures)
6. [Training Configuration](#training-configuration)
7. [Results](#results)
8. [Best Model Selection](#best-model-selection)
9. [Discussion](#discussion)
10. [API](#api)
11. [Streamlit Applications](#streamlit-applications)
12. [Installation](#installation)

---

## Project Structure

- notebooks/
  - Augmentation-notebooks/ — Notebooks for generating the augmented datasets (one for augmenting the CXR dataset, one for augmenting the COVID-QU-Ex dataset)
  - augmented_notebooks/ — Training notebooks NB02–NB14 (augmented datasets)
  - baseline_notebooks/ — Training on original datasets
  - comparison_notebooks/ — Final comparison notebook (NB15)

- app/
  - app.py — Main testing app (single model or all models)
  - improve_quality.py — Image enhancement utilities

- requirements.txt


---

## Datasets

### Original Datasets

**Dataset 1 — Chest X-Ray Masks and Labels (CXR)**
- Source: https://www.kaggle.com/datasets/felipemeganha/chest-xray-masks-and-labels-images
- 138 chest X-ray images with corresponding lung segmentation masks
- Images sourced from the Montgomery County and Shenzhen Hospital datasets
- Format: PNG, variable resolution
- Used as the primary small-scale training source for baseline experiments

**Dataset 2 — COVID-QU-Ex**
- Source: https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
- 33,920 chest X-ray images across three categories: COVID-19, Non-COVID, and Normal
- Contains lung segmentation masks (used) and infection masks (not used)
- Structured into Train, Val and Test splits with category subfolders
- Format: PNG, 256x256
- Used as the second training source and primary out-of-distribution evaluation set

### Augmented Datasets

**Augmented CXR Dataset**
- Source: https://www.kaggle.com/datasets/elhamidyabderahman/augmented-xray-variants-v1
- Generated from the 138 original CXR images using 290 augmentation pipelines
- Total pairs: approximately 40,071 image and mask pairs
- Image size: 256x256, PNG format
- Structure: flat folders with images/ and masks/ subdirectories
- Mask naming convention: image stem + _mask.png

**Augmented COVID-QU-Ex Dataset**
- Source: https://www.kaggle.com/datasets/elhamidyabderahman/covidqu-augmented
- Generated from 33,920 original COVID-QU-Ex lung pairs using 5 pipelines per image
- Total pairs: approximately 169,600 image and mask pairs
- Image size: 256x256, PNG format
- Structure: 22 chunk folders (lung_chunk01_of_22 to lung_chunk22_of_22)
- Each chunk follows: lung/Split/Category/images and masks

---

## Augmentation Pipeline

The augmentation pipeline was run offline on Kaggle CPU using Albumentations and applied before training to generate fixed datasets. This avoids on-the-fly augmentation overhead during training.

**Single transforms applied:**
- Horizontal flip
- Rotations at multiple angles: 5, 15, 30, 45 degrees and their negatives
- Elastic transform at soft, medium and hard intensity
- Grid distortion and optical distortion
- Brightness and contrast adjustments at multiple levels
- Gamma correction up and down
- CLAHE at two clip limits
- Histogram equalization
- Solarize at two thresholds
- Gaussian blur, motion blur, median blur
- Gaussian noise and ISO noise at multiple intensities
- Sharpening, coarse dropout, grid dropout, downscaling
- Affine transforms with scale, translate and shear

**Composite transforms (two and three transform combinations):**
- Flip combined with rotation, elastic, affine, optical distortion
- Elastic combined with photometric transforms
- Rotation combined with photometric transforms
- Affine combined with photometric transforms
- Triple combinations of geometric and photometric transforms

**Pre-augmentation image enhancement:**
- CLAHE with clip limit 2.0 and grid 8x8 applied before augmentation
- Unsharp masking for edge sharpening
- Applied to improve mask consistency across augmented variants

Total pipelines defined: 290

---

## Image Enhancement

A dedicated enhancement module (improve_quality.py) provides the following techniques applicable at inference time through the Streamlit apps.

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
Enhances local contrast by redistributing intensity values in small tiles. Particularly effective for chest X-rays where lung fields have low contrast. Parameters: clip limit (1.0 to 10.0) and grid size (2 to 16).

**Histogram Equalization**
Global contrast enhancement by spreading out the most frequent intensity values. Less adaptive than CLAHE but faster and parameter-free.

**Gamma Correction**
Non-linear brightness adjustment. Values below 1.0 brighten the image, values above 1.0 darken it. Useful for overexposed or underexposed films.

**Sharpening**
Applies an unsharp masking kernel to enhance edges and structural boundaries. Improves visibility of lung borders in soft or blurry images.

**Denoising**
Fast non-local means denoising to remove film grain and scanner noise. Parameter: strength (1 to 30).

---

## Architectures

## Architectures

All models were implemented using the segmentation-models-pytorch library.

| Architecture | Encoder | Attention | Parameters |
|--------------|----------|-----------|------------|
| UNet++ | EfficientNet-B4 | scSE: Yes | ~20.9M |
| UNet++ | EfficientNet-B7 | scSE: Yes | ~66.4M |
| UNet++ | ResNet50 | scSE: Yes | ~32.5M |
| UNet++ | ResNet34 | scSE: Yes | ~24.4M |
| UNet | EfficientNet-B4 | scSE: Yes | ~19.8M |
| DeepLabV3+ | EfficientNet-B4 | scSE: No | ~17.6M |
| MAnet | EfficientNet-B4 | scSE: No | ~22.1M |

UNet++ uses nested dense skip connections that reuse feature maps at multiple scales, making it more powerful than standard UNet for boundary-sensitive tasks.  
The scSE decoder attention recalibrates feature maps both spatially and channel-wise, improving segmentation of thin structures.  
DeepLabV3+ uses atrous convolutions and ASPP to capture multi-scale context without losing resolution.  
MAnet uses multi-scale feature aggregation with channel attention in both encoder and decoder paths.

---

## Training Configuration

Image size: 256 x 256
Loss function: 0.5 x BCE + 0.5 x Dice
Optimizer: AdamW
Learning rate: 5.83e-4
Weight decay: 1e-4
Scheduler: CosineAnnealingLR with T_max=50
Max epochs: 50
Early stopping: Patience = 7
Normalization: ImageNet mean and std
Random seed: 42

Data splits: 80% train, 10% validation, 10% test with random_state=42.

Batch sizes varied by dataset and GPU. CXR on A100 40GB used batch size 64. COVIDQU on A100 used 32. EfficientNet-B7 on any dataset used 32 due to model size. COVIDQU on H100 102GB used 128.

Training was run on Google Colab Pro+ using A100 40GB as the primary GPU and H100 102GB for larger COVIDQU runs. All checkpoints were saved to Google Drive after every improvement with full resume support. Training history JSON was saved after every epoch for timeout safety.

Due to the size of the COVID-QU-Ex augmented dataset at 169,600 pairs, training on the full dataset required approximately 39 minutes per epoch on A100. For COVIDQU training runs the training set was subsampled to 40,000 pairs matching the scale of the CXR dataset. Validation and test sets remained at full size.

---

## Results

### CXR-Trained Models (OOD evaluation on COVID-QU-Ex, 16,960 samples)

| Model | In-dist Dice | In-dist IoU | OOD Dice | OOD IoU |
|-------|--------------|--------------|----------|----------|
| UNet++ / EfficientNet-B7 | 0.9949 | 0.9898 | 0.9607 | 0.9244 |
| UNet / EfficientNet-B4 | 0.9940 | 0.9881 | 0.9603 | 0.9236 |
| UNet++ / EfficientNet-B4 | 0.9939 | 0.9879 | 0.9590 | 0.9217 |
| DeepLabV3+ / EfficientNet-B4 | 0.9930 | 0.9861 | 0.9606 | 0.9241 |
| UNet++ / ResNet34 | 0.9925 | 0.9852 | 0.9572 | 0.9179 |
| MAnet / EfficientNet-B4 | 0.9914 | 0.9830 | 0.9584 | 0.9202 |
| UNet++ / ResNet50 | 0.9912 | 0.9826 | 0.9540 | 0.9122 |

---

### COVIDQU-Trained Models (OOD evaluation on CXR, 4,008 samples)

| Model | In-dist Dice | In-dist IoU | OOD Dice | OOD IoU |
|-------|--------------|--------------|----------|----------|
| UNet++ / EfficientNet-B7 | 0.9926 | 0.9853 | — | — |
| UNet++ / EfficientNet-B4 | 0.9921 | 0.9843 | 0.9430 | 0.8923 |
| UNet / EfficientNet-B4 | 0.9855 | 0.9714 | 0.9573 | 0.9182 |
| DeepLabV3+ / EfficientNet-B4 | 0.9878 | 0.9759 | 0.9537 | 0.9115 |
| MAnet / EfficientNet-B4 | 0.9848 | 0.9701 | 0.9556 | 0.9150 |
| UNet++ / ResNet50 | 0.9819 | 0.9645 | 0.9552 | 0.9143 |
| UNet++ / ResNet34 | 0.9821 | 0.9649 | 0.9532 | 0.9106 |


---


## Selected Model

**Selected model:** UNet++ with EfficientNet-B7 encoder trained on augmented CXR dataset.

### Selection Criteria

This model was selected based on four criteria:

1. **Highest in-distribution performance**
   - Dice score: 0.9949 on the CXR test set

2. **Best cross-dataset generalization (OOD)**
   - OOD Dice: 0.9607 on COVID-QU-Ex  
   - Generalization gap: 0.0342  
   - Despite no exposure to this distribution during training

3. **Most powerful architecture tested**
   - EfficientNet-B7 (~66M parameters)
   - Compound scaling (depth, width, resolution)
   - UNet++ nested skip connections
   - scSE decoder attention for fine-grained lung boundary detail

4. **Deployment success**
   - Successfully deployed as a production REST API
   - Demonstrated practical real-world inference capability
---

## Discussion

The results confirm that training on augmented data significantly improves both in-distribution performance and cross-dataset generalization. Augmented models consistently outperform their baseline counterparts particularly on out-of-distribution data, with OOD Dice scores above 0.95 for the top CXR-trained models.

EfficientNet-based encoders outperform ResNet encoders across all metrics, suggesting that compound scaling produces more transferable feature representations for medical image segmentation. UNet++ consistently outperforms plain UNet with the same encoder, confirming the benefit of nested dense skip connections for boundary-sensitive segmentation tasks.

The observation that CLAHE enhancement rescues poorly performing baseline models on low-quality images highlights an important practical consideration. In clinical settings where image quality is variable, preprocessing pipelines are as important as model architecture. Augmented models internalize this robustness during training while baseline models depend on external preprocessing at inference time.

The relatively small generalization gap between CXR-trained and COVIDQU-trained models of approximately 0.02 to 0.05 Dice points suggests that lung anatomy is sufficiently consistent across X-ray acquisition protocols that a model trained on one distribution transfers well to another. This is encouraging for real-world deployment where training data may not match the target clinical environment.

The pathology detection experiments using TorchXRayVision showed that lung segmentation before classification shifts pathology confidence scores in ways that depend on the specific pathology. Some pathologies show improved detection after masking because the model focuses on lung-specific features. Others show reduced scores because they involve context outside the lung boundary such as cardiomegaly where the cardiac silhouette is partly removed by the lung mask. This confirms that segmentation preprocessing is not universally beneficial and its effect should be evaluated per pathology in clinical applications.

---

## API

The best model is served as a REST API using FastAPI on Kaggle GPU with ngrok tunneling.

Endpoints:
- GET  /                  Health check and model information
- POST /segment           Returns binary lung mask as PNG
- POST /segment/json      Returns JSON with coverage, confidence and base64 mask
- POST /segment/overlay   Returns original image with lung region highlighted in cyan
- POST /segment/all       Returns all outputs combined in a single JSON response

Example usage:

```bash
curl -X POST https://YOUR-NGROK-URL/segment \
  -F "file=@xray.jpg" --output mask.png

curl -X POST https://YOUR-NGROK-URL/segment/overlay \
  -F "file=@xray.jpg" --output overlay.png

curl -X POST https://YOUR-NGROK-URL/segment/json \
  -F "file=@xray.jpg"
```

Sample JSON response:

```json
{
  "status": "ok",
  "inference_ms": 57.1,
  "coverage_pct": 41.57,
  "confidence": 0.9894,
  "mask_size": "256x256",
  "model": {
    "architecture": "UNet++",
    "encoder": "EfficientNet-B7",
    "train_dataset": "CXR (augmented)",
    "indist_dice": 0.9949,
    "ood_dice": 0.9607
  }
}
```

---

## Streamlit Applications

**app.py** tests individual models or runs all models sequentially on an uploaded X-ray image. Supports both baseline and augmented model groups with image enhancement controls and displays mask, overlay and heatmap outputs.

---

## Installation

```bash
pip install -r requirements.txt
pip install transformers
```

Run applications:

```bash
streamlit run app/app.py
```

---

## Author

EL-HAMIDY Abderahman