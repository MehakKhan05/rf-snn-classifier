# RF-SNN Classifier

A spiking neural network (SNN) that classifies RF signal modulation types (e.g., AM, FM, BPSK) using NengoDL.  
This project explores low-power, brain-inspired signal processing for real-time inference in neuromorphic systems.

---

## 🧠 Overview

Traditional modulation classifiers (e.g., CNNs) are computationally expensive and poorly suited for edge deployment.  
This project implements a biologically inspired approach using spiking neural networks, aiming for:

- Real-time classification of modulated RF signals
- Energy-efficient inference with sparse spiking activity
- Compatibility with neuromorphic hardware (e.g., Loihi, FPGA, ODIN)

---

## 📦 Project Structure

```bash
rf-snn-classifier/
│
├── data/             # Raw or preprocessed RF signal datasets
├── nengo/            # NengoDL model scripts
├── models/           # Trained model weights and configurations
├── plots/            # Evaluation plots (confusion matrix, spike activity, etc.)
├── notes/            # Literature review summaries and planning
├── README.md         # This file
├── requirements.txt  # Python dependencies
└── main.py           # Main training/evaluation script

## 🧠 Model Architecture

- **Input**: RF signal samples or engineered features (rate-coded)
- **Ensemble A**: 128 LIF neurons (projects from input)
- **Ensemble B** (optional): 128 LIF neurons, adds depth
- **Output**: 3-node classification layer for AM/FM/BPSK
- **Training**: Supervised learning via surrogate gradients (NengoDL)

The architecture mimics biological signal flow with discrete spiking events, enabling eventual deployment to neuromorphic hardware.

Input (Node, 256-d)
  ↓ (rate-coded current)
LIF Layer (e.g., 512 neurons)
  ↓ (filtered spikes)
Dense Output (4 units)
  ↓
Softmax / Argmax over time-averaged output
