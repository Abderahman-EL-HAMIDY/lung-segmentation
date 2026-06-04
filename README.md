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
lung-segmentation/
├── notebooks/
│   ├── augmented_notebooks/       Training notebooks NB02-NB14 on augmented datasets
│   ├── baseline_notebooks/        Training on original non-augmented datasets
│   └── comparison_notebooks/      Final results comparison notebook (NB15)
├── app/
│   ├── app.py                   Main testing app (single model or all models)
│   └── improve_quality.py         Image enhancement utilities
└── requirements.txt