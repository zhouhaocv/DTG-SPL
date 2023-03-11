# Towards-Diverse-Temporal-Grounding-under-Single-Positive-Labels
This repository contains the revised pytorch codes in our paper "Towards Diverse Temporal Grounding under Single Positive Labels".
## Installation
**Requirement**
- anaconda
### 1.install dependencies
```bash
conda env create -f environment.yml
conda activate DTG-SPL
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.0.0/en_core_web_lg-3.0.0.tar.gz
pip install en_core_web_lg-3.0.0.tar.gz
```
### 2.download datasets
- download data from [Google Driver](https://drive.google.com/file/d/1HU7aXrEbQtYz_9xxKTXu6Mg-BBJQCvfW/view?usp=sharing) and unzip it to the DTG-SPL/data folder.

### 3.training for Charades-STA
```bash
cd $INSTALL_DIR
cd ablations
python train_charades_i3d.py
```
