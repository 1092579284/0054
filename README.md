# Milano Network Traffic Prediction - UCL Master's Thesis

This repository contains the implementation and experimental code for a UCL master's thesis on network traffic prediction using deep learning techniques applied to the Milano telecommunications dataset.

## 📋 Project Overview

This project investigates various aspects of network traffic prediction using deep learning models, focusing on:

- **Optimal feature selection** for network traffic prediction
- **Model architecture comparison** through ablation studies
- **Training data volume impact** on prediction performance
- **Transfer learning strategies** with different unfreezing approaches
- **Feature importance analysis** using SHAP and visualization techniques

## 🏗️ Project Structure

```
├── ablation_study/                    # Model architecture ablation experiments
│   ├── models.py                      # Different model architectures
│   ├── ablation_experiment.py         # Main ablation experiment script
│   ├── run_all_experiments.py         # Automated experiment runner
│   └── results/                       # Experiment results and visualizations
├── data/                              # Dataset and related files
│   ├── cleaned_dataset_30/            # Preprocessed Milano dataset
│   └── milano-grid.geojson           # Geographic grid data
├── data_scaling_results/              # Data volume impact experiment results
│   ├── *.png                         # Performance visualization charts
│   ├── *.csv                         # Detailed results data
│   └── spatial_heatmaps/             # Geographic performance heatmaps
├── unfreezing_strategies_3/           # Transfer learning experiments
│   ├── milano_pytorch_unfreezing_comparison.py
│   └── pytorch_unfreezing_results/    # Transfer learning results
├── feature_selection_results/         # Feature analysis results
├── data_scaling_experiment_fixed.py   # Data volume impact experiment
├── feature_analysis_and_visualization.py  # SHAP analysis and t-SNE
└── intelligent_feature_selection.py   # Automated feature selection
```

## 🔬 Research Components

### 1. Ablation Study
- **Purpose**: Compare different neural network architectures for sequence modeling
- **Models tested**: 
  - Unidirectional GRU
  - Pure LSTM
  - GRU+LSTM combinations
  - Baseline model (Bidirectional GRU + LSTM)
- **Location**: `ablation_study/`

### 2. Data Scaling Experiment
- **Purpose**: Investigate the relationship between training data volume and model performance
- **Grid sizes tested**: 50, 100, 150, 200, 250, 300, 400, 500, 600, 800
- **Key findings**: Performance improvement with increased data, with diminishing returns
- **Location**: `data_scaling_experiment_fixed.py` and `data_scaling_results/`

### 3. Transfer Learning Analysis
- **Purpose**: Evaluate different unfreezing strategies for transfer learning
- **Strategies**: No unfreezing, full unfreezing, progressive unfreezing, adaptive unfreezing
- **Location**: `unfreezing_strategies_3/`

### 4. Feature Analysis
- **Purpose**: Understand feature importance using SHAP values and dimensionality reduction
- **Techniques**: SHAP analysis, t-SNE visualization, PCA
- **Location**: `feature_analysis_and_visualization.py`

## 🚀 Quick Start

### Prerequisites

```bash
# Required Python packages
torch>=1.9.0
numpy
pandas
scikit-learn
matplotlib
seaborn
shap
```

### Running Experiments

#### 1. Ablation Study
```bash
cd ablation_study
python run_all_experiments.py
```

#### 2. Data Scaling Experiment
```bash
python data_scaling_experiment_fixed.py
```

#### 3. Transfer Learning Comparison
```bash
cd unfreezing_strategies_3
python milano_pytorch_unfreezing_comparison.py
```

#### 4. Feature Analysis
```bash
python feature_analysis_and_visualization.py
```

