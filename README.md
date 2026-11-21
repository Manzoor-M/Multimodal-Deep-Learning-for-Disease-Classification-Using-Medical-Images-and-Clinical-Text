# Multimodal Classification of Chest X-ray Findings

A comparative deep-learning study of image-only, clinical-indication-only, and multimodal approaches for multilabel classification of seven chest X-ray findings using the Indiana University Chest X-ray (IU-Xray) dataset.

## 1\. Project Overview

This project investigates whether clinical indication text adds useful information to chest X-ray image features when predicting radiological findings.

Three approaches are compared:

1. **Image-only:** EfficientNet-B0 extracts visual features from chest X-ray images.
2. **Text-only:** BioClinicalBERT encodes the clinical indication associated with the examination.
3. **Multimodal V2:** EfficientNet-B0 and BioClinicalBERT representations are projected into a shared feature space and combined using a trainable fusion network.

The task is **multilabel classification**. An image may be associated with more than one of these seven findings:

* Cardiomegaly
* Atelectasis
* Pleural effusion
* Emphysema
* Pulmonary congestion
* Opacity
* Granulomatous disease

This project studies classification performance on the selected dataset and experimental setup. It is not intended to produce independent clinical diagnoses or replace clinical judgement.

## 2\. Dataset

The experiments use the **Indiana University Chest X-ray (IU-Xray)** dataset, which contains chest radiographs paired with radiology reports.

### Dataset summary

|Item|Count|
|-|-:|
|Original chest X-ray image records|7,470|
|Original XML reports|3,955|
|Unique reports represented in the matched image-record dataset|3,851|
|Image records with missing or empty clinical indications excluded|159|
|Image records retained for indication-based experiments|7,311|
|Training reports|2,695|
|Validation reports|578|
|Test reports|578|
|Training images|5,122|
|Validation images|1,089|
|Test images|1,100|
|Unique reports represented in the test set|565|

The 3,955 count refers to XML reports in the original source collection. The 3,851 count refers to unique reports represented after matching image records to report identifiers.

### Text input and leakage control

Only the **Clinical Indication** field is used as text input. The `Findings` and `Impression` sections are excluded because they may explicitly state the radiological findings that the models are trained to predict, creating a risk of target-label leakage.

The data is split at the report level so images associated with the same report are not distributed across different splits. All three models are evaluated on the same matched test set of 1,100 images.

### Obtaining the dataset

The dataset archives are not included in this source-code package. Obtain them from the IU-Xray collection:

* [IU-Xray PNG images](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz)
* [IU-Xray XML reports](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz)
* [Open-i portal](https://openi.nlm.nih.gov/)

After downloading, extract the archives into the directory structure expected by the preprocessing scripts. The expected locations described by the project scripts are:

```text
IU\_Xray/
├── NLMCXR\_png/
└── NLMCXR\_reports/
    └── ecgen-radiology/
```

Check the individual scripts for their exact expected filenames and paths before running them.

## 3\. Model Architectures

### Image-only model

* Image encoder: EfficientNet-B0 pretrained on ImageNet.
* Input image size: 224 × 224 pixels.
* The original classification layer is replaced with a seven-output classification layer.
* Training: 10 epochs, AdamW optimizer, learning rate `1e-4`, batch size `16`.
* Loss: Weighted Binary Cross-Entropy with Logits Loss.

### Text-only model

* Text encoder: `emilyalsentzer/Bio\_ClinicalBERT`.
* Input: Clinical Indication only.
* Maximum sequence length: 128 tokens.
* Classification head: seven output logits.
* Training: 5 epochs, AdamW optimizer, learning rate `2e-5`, batch size `8`.
* Loss: Weighted Binary Cross-Entropy with Logits Loss.

### Multimodal V2

The final multimodal model combines image and clinical text representations:

1. EfficientNet-B0 produces a 1,280-dimensional image representation.
2. BioClinicalBERT produces a 768-dimensional text representation.
3. Separate projection layers map both representations to 256 dimensions.
4. The projected vectors are concatenated into a 512-dimensional joint representation.
5. A fusion network with fully connected dimensions `512 → 256 → 128` processes the joint representation.
6. The final classification layer produces seven output logits.

Both pretrained encoders are frozen during training. The projection layers, fusion network, and final classification layer are trainable.

Training configuration:

* Optimizer: AdamW
* Learning rate: `1e-4`
* Weight decay: `1e-4`
* Batch size: `8`
* Maximum epochs: `10`
* Loss: Weighted Binary Cross-Entropy with Logits Loss
* Checkpoint selection: Lowest validation loss

## 4\. Experimental Results

All three approaches were evaluated on the same matched test set of 1,100 images.

### Overall performance

|Model|Threshold strategy|Macro F1-score|Macro ROC-AUC|
|-|-|-:|-:|
|Image-only|Fixed (0.5)|0.2203|**0.8001**|
|Text-only|Fixed (0.5)|0.1136|0.6491|
|Multimodal V2|Fixed (0.5)|0.1997|0.7559|
|Multimodal V2|Validation-optimized, class-specific|0.2203|0.7559|

For Multimodal V2, class-specific thresholds were selected using the validation set and then kept unchanged for test evaluation. The selected thresholds ranged from 0.39 to 0.79.

### Class-wise ROC-AUC

|Radiological finding|Image-only|Text-only|Multimodal V2|
|-|-:|-:|-:|
|Cardiomegaly|0.8651|0.6797|0.7890|
|Atelectasis|0.7424|0.6356|0.6795|
|Pleural effusion|0.8550|0.7037|0.7574|
|Emphysema|0.8783|0.6800|0.8300|
|Pulmonary congestion|0.8962|0.6516|0.8680|
|Opacity|0.7350|0.6320|0.7267|
|Granulomatous disease|0.6285|0.5609|0.6409|

### Interpretation

The image-only model achieved the highest macro ROC-AUC (0.8001). The text-only model performed below the image-based approach in this experiment. Multimodal V2 reached the same macro F1-score as the image-only model after validation-based threshold optimization, but its macro ROC-AUC remained lower (0.7559).

Therefore, the results do **not** demonstrate that multimodal fusion outperforms the image-only baseline. They show that threshold selection affects F1-score and that clinical indication text, as used in this experiment, did not improve overall discrimination over image features alone.

## 5\. Repository Contents

The source package contains 24 Python scripts covering preprocessing, training, evaluation, threshold optimization, and a demonstration interface.

### Preprocessing and dataset inspection

* `prepare\_dataset.py`
* `analyze\_dataset.py`
* `build\_multimodal\_dataset.py`
* `inspect\_multimodal\_dataset.py`
* `split\_dataset.py`
* `prepare\_indication\_text.py`
* `prepare\_indication\_splits.py`
* `check\_report\_ids.py`
* `check\_text\_label\_leakage.py`

### Model training

* `train\_image\_baseline.py`
* `train\_text\_baseline.py`
* `train\_indication\_baseline.py`
* `train\_multimodal.py`
* `train\_multimodal\_v2.py`
* `train\_multimodal\_v3.py`

### Evaluation and threshold optimization

* `evaluate\_image\_baseline.py`
* `evaluate\_image\_matched.py`
* `evaluate\_text\_baseline.py`
* `evaluate\_indication\_baseline.py`
* `evaluate\_multimodal.py`
* `evaluate\_multimodal\_v2.py`
* `evaluate\_multimodal\_v3.py`
* `optimize\_thresholds\_v2.py`

### Demonstration

* `demo\_v2.py` — Tkinter-based demonstration interface.

The package also includes project documentation, dependency information, a source manifest, and CSV result files. Refer to the actual files in the repository for the complete directory listing.

## 6\. Environment Setup

Python 3.10 or 3.11 is recommended.

Create and activate a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the listed dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch installation note

PyTorch and torchvision installation can depend on the operating system, hardware, and CUDA version. If the default installation does not match your machine or GPU setup, install a compatible PyTorch/torchvision build using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

The pretrained BioClinicalBERT and EfficientNet-B0 weights may be downloaded by the relevant libraries when the corresponding models are initialized. Internet access may therefore be required on the first run.

## 7\. Running the Project

The scripts cover separate stages of the workflow. A typical high-level order is:

1. Obtain and extract the IU-Xray image and report archives.
2. Run the dataset preparation and inspection scripts.
3. Build the image-report and multimodal dataset.
4. Prepare indication-only text data and report-level train, validation, and test splits.
5. Run the image-only, text-only, and multimodal training scripts.
6. Run the corresponding evaluation scripts.
7. Run threshold optimization for Multimodal V2 when reproducing the validation-optimized F1 result.
8. Run `demo\_v2.py` if its required model checkpoint and data paths are available.

Relevant scripts include:

```text
prepare\_dataset.py
analyze\_dataset.py
build\_multimodal\_dataset.py
inspect\_multimodal\_dataset.py
split\_dataset.py
prepare\_indication\_text.py
prepare\_indication\_splits.py

train\_image\_baseline.py
train\_text\_baseline.py
train\_multimodal\_v2.py

evaluate\_image\_matched.py
evaluate\_text\_baseline.py
evaluate\_multimodal\_v2.py
optimize\_thresholds\_v2.py
```

**Important:** This is a high-level workflow, not a guarantee that every script can be run in this exact sequence without configuration changes. Check each script's input paths, output paths, command-line arguments, and dependencies before execution. Run scripts from the project directory unless their code specifies otherwise.

## 8\. Reproducibility and Limitations

* The dataset archives are not included in the source package and must be obtained separately.
* Trained model checkpoints are not included.
* Pretrained model weights may need to be downloaded when running the scripts.
* Results depend on the dataset files, preprocessing, library versions, hardware, random seeds, and training configuration.
* The clinical indication field may be short or provide limited information about the target findings.
* The current multimodal architecture uses feature-level concatenation; other fusion strategies may behave differently.
* The study uses one dataset and should not be interpreted as evidence of clinical readiness or generalization to other hospitals.

## 9\. Future Work

Potential directions include:

* Exploring alternative feature-fusion and cross-modal attention methods.
* Evaluating on larger and more diverse chest X-ray datasets.
* Studying additional clinical text sources that do not expose target labels.
* Improving probability calibration and threshold selection.
* Conducting more extensive validation and reproducibility experiments.

