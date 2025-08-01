# 🧠 Schizophrenia Classification Using Graph Neural Networks

This project builds a Graph Neural Network (GNN) to classify whether a subject is a Schizophrenic Patient (`1`) or a Healthy Control (`0`) using functional and structural brain imaging data. The model integrates graph-based features from fMRI (FNC) and scalar features from sMRI (SBM).

---

## Dataset Description

We use multimodal neuroimaging data consisting of:

### 1. Functional Network Connectivity (FNC)
- Correlation features between timecourses of 28 ICA brain components.
- Represented as a graph:
  - **Nodes**: Brain regions (28 total)
  - **Edges**: Correlation between region pairs
  - **Edge Attributes**: Strength of functional connectivity
  - **Node Features**: 3D spatial coordinates (x, y, z)

### 2. Source-Based Morphometry (SBM)
- 32 ICA-derived components representing gray matter concentration.
- Treated as a graph-level feature vector for each subject.

---

## Files Used

- `train_FNC.csv`: FNC correlation features (edge attributes)
- `train_labels.csv`: Binary labels (0 or 1)
- `rs_fMRI_FNC_mapping.csv`: Mapping from FNC columns to (nodeA, nodeB)
- `train_node.csv`: Node coordinates (x, y, z) for ICA components
- `train_SBM.csv`: SBM ICA loadings per subject (graph-level features)

---

##  Graph Construction

Each subject is modeled as a graph using:

```python
Data(
  x=[28, 3],                # Node coordinates
  edge_index=[2, 378],      # Graph structure
  edge_attr=[378, 1],       # FNC edge weights
  sbm=[32],                 # SBM features
  y=[1],                    # Label
  subject_id=xxxxx
)

class GNNWithEdgeAttrsAndSBM(nn.Module):
    def __init__(self, in_channels=3, edge_dim=1, hidden_dim=256):
        ...
    def forward(self, data):
        # Apply NNConv layers
        # Global pooling over nodes
        # Concatenate SBM features
        # Classify with final linear layer