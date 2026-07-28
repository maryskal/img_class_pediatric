# Pediatric Chest X-Ray AI Prototype

Prototype code for pediatric chest X-ray image classification experiments, focused on pneumonia versus normal image categories and on the technical challenges of working with limited pediatric imaging data.

This repository is kept as a portfolio example of clinical-technical translation in medical imaging AI: converting a pediatric imaging question into preprocessing, model design, hyperparameter exploration and evaluation code.

## Clinical-Technical Focus

Pediatric AI work often raises questions that are not visible from model code alone:

- Can models trained or pre-trained on broader imaging data be adapted to pediatric tasks?
- How should preprocessing and thoracic masking affect model inputs?
- How can limited pediatric sample size increase overfitting risk?
- Which outputs are exploratory metrics rather than clinically validated evidence?

This repository shows the technical side of that workflow while keeping the clinical limitations explicit.

## Dataset

The original public dataset referenced by the project is:

- Chest X-Ray Images (Pneumonia), Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The raw images and trained model weights are not included in this repository. Scripts expect data and model artifacts to be available locally.

## What Is Included

- Image preprocessing utilities.
- Optional thoracic masking workflow.
- U-Net based representation learning experiments.
- Pediatric classification model code.
- Evaluation and plotting utilities.
- Hyperparameter exploration scripts.

## Repository Structure

```text
.
├── entrenar_modelo_unsupervised.py
├── pediatric.py
├── ht_exe.py
├── funciones_imagenes/
├── funciones_evaluacion/
├── funciones_unsupervised/
├── otras_funciones/
└── imagenes_modelos/
```

## Methods

The workflow includes:

- Grayscale conversion, resizing and z-score normalization.
- Optional CLAHE contrast enhancement.
- Optional mask-based preprocessing to focus on thoracic regions.
- U-Net based representation learning.
- Pediatric classification architectures with attention-style components.
- Evaluation using model outputs and custom metric utilities.

## Limitations

This is an exploratory research prototype. It has not been clinically validated and is not intended for diagnosis, triage, treatment decisions or patient care.

Important limitations:

- Raw image data and trained weights are not included.
- Some scripts expect local paths or previously generated artifacts.
- Results should not be interpreted as clinical performance claims.
- A clinically meaningful evaluation would require external validation, bias analysis, calibration, workflow assessment and prospective clinical study design.

## Portfolio Context

This repository is most relevant as evidence of medical imaging AI literacy, pediatric validation thinking, preprocessing choices and clinical caution around limited datasets.

It supports a Clinical Technology Consultant profile by showing how a clinical question can be translated into data preparation, model experimentation and validation-aware documentation.

## License

No license has been specified yet. Reuse is not granted unless a license is added.
