# PI-INR: Physics-Geometry Fusion for Uncertainty-Aware Early Warning in Adaptive Radiotherapy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official PyTorch implementation of the framework proposed in the paper:  
**"Physics-Geometry Fusion with Uncertainty-Aware Early Warning for Dosimetric Decay in Adaptive Radiotherapy via Implicit Neural Representations"**  
*(Submitted to Medical Image Analysis, 2026)*

---

## Overview

We propose a novel, zero-dose-computation framework (PI-INR) for real-time dosimetric decay warning in Adaptive Radiotherapy (ART). By bridging Riemannian geometry, Lie derivatives, and Evidential Deep Learning (EDL), this framework provides a highly interpretable Red-Yellow-Green clinical triage system in under 8 seconds per patient, avoiding the heavy computational burden of traditional Deformable Dose Accumulation (DDA).

### Key Features

- Topology-preserving deformable registration with analytical Jacobian constraint
- Riemannian dose manifold for physics-informed regularization
- Lie derivative entropy (ATI) for zero-shot dosimetric risk assessment
- Evidential Deep Learning for calibrated uncertainty estimation
- Traffic-light clinical decision system (Red-Yellow-Green)

---

## Dataset

This code is designed to work with the Pancreatic-CT-CBCT-SEG dataset from The Cancer Imaging Archive (TCIA).

| Item | Details |
|------|---------|
| Dataset Name | Pancreatic-CT-CBCT-SEG |
| Institution | The Cancer Imaging Archive (TCIA) |
| DOI | 10.7937/TCIA.ESHQ-4D90 |
| Patients | 40 |
| Modalities | CT, RTDOSE, RTSTRUCT |

### Download Instructions

1. Visit: https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/
2. Download Version 2 (13.3 GB) using NBIA Data Retriever
3. Place data in: ./data/Pancreatic-CT-CBCT-SEG/

The directory structure should look like:

PI-INR-MedIA/
└── data/
    └── Pancreatic-CT-CBCT-SEG/
        ├── Pancreas-CT-CB_001/
        ├── Pancreas-CT-CB_002/
        └── ...

---

## Installation

Option 1: Using pip

git clone https://github.com/YourUsername/PI-INR-MedIA.git
cd PI-INR-MedIA
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Option 2: Using conda

git clone https://github.com/YourUsername/PI-INR-MedIA.git
cd PI-INR-MedIA
conda env create -f environment.yml
conda activate pi-inr

---

## Usage

Quick Start Demo

Run a quick test on a single patient (100 epochs, ~2-5 minutes):

python demo.py --patient Pancreas-CT-CB_001 --quick

Full Pipeline

Process all patients with full training (1500 epochs, ~3-5 min/patient on CPU):

python run_pipeline.py
python run_statistics.py
python run_visualization.py

Advanced Options

# Process only first 10 patients
python run_pipeline.py --max-patients 10

# Force re-run even if results exist
python run_pipeline.py --force-rerun

# Run on CPU only
python run_pipeline.py --device cpu

---

## Output Structure

results/MedIA_Ultimate_Run_Final/
├── MedIA_Quantitative_Results.csv      # Main results table
├── patient_status.csv                   # Data integrity check
├── batch_log.txt                         # Processing log
├── Pancreas-CT-CB_001/                   # Individual patient results
│   ├── Warped.nii.gz                      # Registered image
│   ├── Warped_BSpline.nii.gz              # B-Spline baseline
│   ├── ATI_Map.nii.gz                     # Risk map
│   ├── Uncertainty.nii.gz                 # Uncertainty map
│   ├── results.npz                        # Numpy archive
│   └── metadata.json                      # Configuration metadata
├── MedIA_Final_Excel/                     # Paper-ready outputs
│   ├── excel_tables/                      # Excel spreadsheets
│   ├── figures/                           # Standard figures
│   └── text/                              # Paper paragraphs
└── MedIA_Paper_Figures/                   # Publication figures
    ├── Figure5_MedIA.pdf/tiff
    ├── FigS1_Flowchart.pdf/tiff
    ├── FigS2_Patient012_Annotated.pdf/tiff
    ├── FigS3_Patient021_Annotated.pdf/tiff
    ├── FigS4_SSIM_Histogram.pdf/tiff
    ├── FigS5_ATI_Log_Dist.pdf/tiff
    └── FigS6_ATI_DDA_Dist.pdf/tiff

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Network frequency (w0) | 20.0 |
| Batch size | 25,000 |
| Training epochs | 1,500 |
| Learning rate | 5e-4 |
| Physical regularization (l_phys) | 0.2 |
| Edge matching (l_edge) | 5.0 |
| Smoothness (l_smooth) | 0.1 |
| Expansion penalty (l_exp) | 0.001 |
| Diffeomorphic penalty (l_fold) | 0.1 -> 5.0 (annealed) |

---

## Citation

If you find this code useful, please cite our paper:

@article{pi_inr_2026,
  title={Physics-Geometry Fusion with Uncertainty-Aware Early Warning for Dosimetric Decay in Adaptive Radiotherapy via Implicit Neural Representations},
  author={Anonymous Authors},
  journal={Medical Image Analysis (Under Review)},
  year={2026}
}

Also cite the dataset:

@article{hong2021breath,
  title={Breath-hold CT and cone-beam CT images with expert manual organ-at-risk segmentations from radiation treatments of locally advanced pancreatic cancer},
  author={Hong, J and Reyngold, M and Crane, C and Cuaron, J and Hajj, C and Mann, J and Zinovoy, M and Yorke, E and LoCastro, E and Apte, AP and Mageras, G},
  journal={The Cancer Imaging Archive},
  year={2021},
  doi={10.7937/TCIA.ESHQ-4D90}
}

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

The Pancreatic-CT-CBCT-SEG dataset is licensed under the CC BY 4.0 license.

---

## Notes

- Python 3.9+, PyTorch 2.0+
- Random seed fixed (42) for reproducibility
- ~3-5 min/patient on CPU, ~30-60 sec on GPU
- Dataset NOT included - download separately from TCIA

---

## Contact

For questions or issues, please open an issue on GitHub.
