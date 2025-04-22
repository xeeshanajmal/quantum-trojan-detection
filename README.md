
# Quantum Trojan Detection Dataset

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
 ┣ 📂 datasets/                  # Final CSV datasets (clean + malicious circuits)
 ┣ 📂 models/                    # Saved models (e.g., .pkl, serialized classifiers)
 ┣ 📂 models_with_details/       # Visuals + metrics for each model (confusion, ROC, etc.)
 ┣ 📂 notebooks/                 # Main Jupyter notebooks for QSVM and RF
 ┃ ┣ 📜 rf_classifier.ipynb
 ┃ ┣ 📜 qsvm_classifier.ipynb
 ┃ ┗ 📜 dataset_generation.ipynb
 ┣ 📂 qsvm_outputs/              # Output folders per algorithm for QSVM results
 ┣ 📂 rf_outputs/                # Output folders per algorithm for RF results
 ┣ 📂 six_algorithms_with_details/ # Qiskit code for all six quantum algorithms
 ┃ ┣ 📜 grover.ipnyb
 ┃ ┣ 📜 qaoa.ipnyb
 ┃ ┗ 📜 ...
 ┣ 📜 README.md
 ┣ 📜 .gitignore
 ┗ 📜 requirements.txt
