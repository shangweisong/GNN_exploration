# Application of Graph Neural Networks on the Cora Dataset for Node Classification

## Introduction

The **Cora dataset** is a popular benchmark in graph machine learning, widely used to test graph neural network (GNN) models. It consists of scientific papers connected by citation links. The main task is **node classification**—predicting the research topic of each paper by leveraging both its features and the graph structure.

---

## Graph Data: Understanding Cora

* **Nodes:** Each node represents a scientific publication.
* **Features:** Nodes are described by 1,433-dimensional binary bag-of-words vectors indicating word presence in the papers.
* **Edges:** Citation links connect papers, forming around 2,700 nodes and 5,400 edges. For GNNs, these edges are often treated as undirected to simplify aggregation.
* **Labels:** Each node belongs to one of seven research topic classes (e.g., Neural Networks, Reinforcement Learning).
* **Data Preparation:**

  * Features are normalized to improve training stability.
  * Self-loops are added to allow nodes to consider their own features during message passing.
  * The dataset is split into training, validation, and test sets, usually with only a small portion labeled for semi-supervised learning.

---

## Graph Model Architecture for Node Classification

* **Input:** The model ingests the node feature matrix and the graph connectivity information.
* **Graph Convolutional Layers:**
  These layers perform **message passing**, where each node updates its representation by aggregating feature information from neighboring nodes and itself. This process helps capture local graph structure and feature patterns.
* **Non-linearities:**
  Activation functions like ReLU introduce non-linearity, and dropout helps prevent overfitting.
* **Output Layer:**
  Produces class probabilities for each node, predicting the research topic label.
* **Training:**
  Uses cross-entropy loss calculated on the labeled nodes only, enabling the model to generalize in a semi-supervised manner.
* **Evaluation:**
  Model F1-score are computed on unseen test nodes to assess performance.

---

## Summary

The Cora dataset exemplifies a real-world application of graph neural networks, showing how combining node features with graph topology improves classification tasks. The GNN’s message passing framework effectively leverages local and global graph information, making it a powerful tool for node classification in citation networks and beyond.

