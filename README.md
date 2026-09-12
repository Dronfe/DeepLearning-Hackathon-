<<<<<<< HEAD
# ROS2-22AIE442-project-
# ROS2-22AIE442-project-
=======
# Track 3: Activity-Cliff Aware GNN for Tox21

## Overview
This project implements a multi-head **GATv2** Graph Neural Network enhanced with **ToxCliffLoss** (Combined BCE + Triplet Contrastive Loss) to penalize activity cliffs on the MoleculeNet Tox21 dataset.

## Key Results
* **Baseline 2-Layer GCN Test Macro ROC-AUC:** 0.7391
* **ToxGATv2 + ToxCliffLoss Test Macro ROC-AUC:** 0.7986
* **Absolute Improvement:** +5.95%

## Model Architecture
* **Backbone:** Multi-Head Dynamic Graph Attention (`GATv2Conv`)
* **Pooling:** Dual Mean + Max Global Graph Pooling
* **Explainability:** `GNNExplainer` for localized toxicophore attribution
>>>>>>> 7fa49f3 (Initial commit)
