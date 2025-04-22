
# Quantum Trojan Detection

This project focuses on the detection of trojan-injected quantum circuits using classical and quantum machine learning models. It forms part of a master’s thesis exploring security threats in quantum computing environments.

## Research Focus

- **Objective**: Detect anomalies and logical tampering in quantum circuits.
- **Models Used**: 
  - Classical: Random Forest
  - Quantum: Quantum Support Vector Machine (QSVM)
- **Circuits Analyzed**:
  - Deutsch–Jozsa (DJ)
  - Grover’s Search
  - Quantum Fourier Transform (QFT)
  - Shor’s Algorithm
  - Bernstein–Vazirani (BV)
  - Quantum Approximate Optimization Algorithm (QAOA)

## Core Features

- Clean and malicious quantum circuit generation (Qiskit)
- Circuit simulation using Qiskit Aer
- Feature extraction from quantum circuits:
  - Depth, gate counts, entropy, structure
- CSV dataset generation for training/testing
- Binary classification: `Clean` vs. `Malicious`
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC

## 📁 Project Structure

```plaintext
📦 quantum-trojan-detection/
 ┣ 📂 data/                      # CSV datasets for each algorithm
 ┣ 📂 notebooks/                # Jupyter notebooks for QSVM and RF experiments
 ┣ 📂 scripts/                  # Python scripts for dataset generation
 ┣ 📂 models/                   # Saved ML/QML models (optional)
 ┣ 📜 dataset_generation.ipynb # Main pipeline for creating labeled data
 ┣ 📜 qsvm_classifier.ipynb     # QSVM training and visualization
 ┣ 📜 rf_classifier.ipynb       # Random Forest training and feature importance
 ┣ 📜 README.md                 # You’re here!
