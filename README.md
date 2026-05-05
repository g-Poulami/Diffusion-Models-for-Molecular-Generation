Here is a professional and comprehensive `README.md` for your project, specifically designed for your **Diffusion Models for Molecular Generation** repository. It incorporates the biological significance of your work and the technical results you achieved.

---

# Conditional Diffusion for Molecular Property Generation

This repository features a functional implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** applied to **Property-Guided Drug Design**. The model is trained on the **QM9 dataset** to learn the distribution of molecular properties and generate new property vectors conditioned on specific target chemical values.

## Biological Motivation
In pharmaceutical research, identifying molecules that satisfy strict **ADMET profiles** (Absorption, Distribution, Metabolism, Excretion, and Toxicity) is a significant challenge due to the near-infinite size of chemical space. 

This project implements a generative framework that allows a researcher to:
1.  **Define a Target**: Specify desired chemical properties, such as polarizability or dipole moment.
2.  **Reverse Diffusion**: Use the model to "sculpt" a molecular profile from random Gaussian noise that fits the defined target.
3.  **Identify Scaffolds**: Map these generated profiles back to real-world chemical structures.

## Technical Overview
The project utilizes a **Conditional Multi-Layer Perceptron (MLP)** as the noise predictor within a DDPM framework.

*   **Dataset**: QM9, containing quantum chemistry properties of ~134k small organic molecules.
*   **Architecture**: Conditional MLP with Gaussian Diffusion logic.
*   **Workflow**: Forward diffusion adds noise to property vectors; the model learns to reverse this process guided by a target property $y$.

## Verification & Results

### 1. Training Performance
The model successfully learned the underlying distribution of the QM9 dataset, demonstrated by the convergence of the Mean Squared Error (MSE) loss during training.
*   **Deliverable**: `training_curve.png`

### 2. Denoising Progression
Diagnostic plots illustrate the transformation of high-variance Gaussian noise into structured molecular property profiles over 1,000 timesteps.
*   **Deliverable**: `denoising_progression.png`

### 3. Molecular Scaffold Identification
Generated property profiles were verified against the real QM9 dataset. The model successfully identified fundamental chemical scaffolds, proving it captured "meaningful structure":

| Generated Profile | Atomic Numbers [z] | Chemical Identity |
| :--- | :--- | :--- |
| Candidate 1 | [6, 1, 1, 1, 1] | **Methane ($CH_4$)** |
| Candidate 2 | [7, 1, 1, 1] | **Ammonia ($NH_3$)** |
| Candidate 3 | [6, 6, 1, 1, 1, 1, 1, 1] | **Ethane ($C_2H_6$)** |

## How to Run

### Installation
```powershell
pip install torch torch_geometric rdkit matplotlib
```

### Training
```powershell
python training.py
```
*Note: The script automatically handles the QM9 dataset download to the `/data` folder on the first run.*

### Diagnostics & Visualization
```powershell
python diagnostic.py
python nearest_neighbour.py
```

## Repository Structure
*   `models/diffusion.py`: Core DDPM architecture.
*   `training.py`: Training loop and loss visualization.
*   `diagnostic.py`: Denoising progression and distribution analysis.
*   `nearest_neighbour.py`: Mapping generated tensors to real QM9 molecules.

---

Would you like to add a section specifically explaining the mathematical loss function used in the training script?
