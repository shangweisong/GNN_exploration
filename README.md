# 📊 Exploration of GNN Applications with PyTorch Geometric

This repository explores the application of Graph Neural Networks (GNNs) using the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) library. It focuses on demonstrating how GNNs can be applied to a variety of datasets and problem types, while also providing insight into graph data creation, visualization, and architecture design.

---

## 🔍 Overview

We aim to cover the following key areas:

- How to convert raw data into graph-structured data  
- Techniques for visualizing graphs using popular Python libraries  
- Implementing and experimenting with various GNN architectures:
  - **Vanilla GNNs**
  - **GNNs with edge features**
  - **Spatio-temporal GNNs**

---

## 📂 Datasets

The project utilizes three diverse datasets:

1. **Cora Citation Dataset**  
   - Classic benchmark for node classification in citation networks.

2. **MLSP 2014 Schizophrenia Dataset**  
   - Brain connectome data used to explore GNNs for neuroimaging applications.

3. **Singapore Rainfall Dataset**  
   - A geospatial-temporal dataset used to demonstrate spatio-temporal GNN models.

---

## Features

- Graph construction from tabular/raw data  
- Visualization using `NetworkX`, `matplotlib`,`plotly`,and others  
- End-to-end training pipelines using PyTorch Geometric  
- Modular and extensible architecture for experimenting with various GNN models

---

## 📁 Project Structure

```

```

---

## Getting Started


### Installation

```bash
# Clone the repository
git clone https://github.com/shangweisong/GNN_exploration
cd GNN_exploration

# (Recommended) Create a virtual environment
python -m venv venv
uv sync-all-groups
source venv/bin/activate  # or venv\Scripts\activate on Windows

```

---

##  Visualization

The project includes several utilities to visualize graph structures and training progress using:

- `networkx`
- `matplotlib`
-  `plotly`
---

## Future Work

- Incorporation of attention-based GNNs (e.g., GAT, Graph Transformers)  
- Integration with more real-world temporal datasets  
- Hyperparameter tuning utilities  
---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.