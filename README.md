# COAT-GNN: Cooperative Attribute Learning and Topological Optimization for Protein Binding Sites Prediction


## I. Appendix Overview
This appendix.pdf provides detailed methodological, experimental, and analytical information supporting the main paper “COAT-GNN: Cooperative Attribute Learning and Topological Optimization for Protein Binding Site Prediction.”
It is organized as follows:

### A. Graph Encoder

Describes how a protein is represented as a residue-level undirected graph and how the adjacency matrix is constructed from 3D coordinates or ESMFold-predicted structures. It also explains the implementation of Graph Attention (GAT) layers used to initialize node embeddings.

### B. Amino Acid Features

Provides the detailed composition of residue-level input features (total 66 dimensions). Subsections explain:

B.1–B.2: Evolutionary features (PSSM, HHM) and their normalization.

B.3: Secondary structure (DSSP) and backbone torsion features.

B.4: Pseudo-position embeddings based on residue spatial relations.

B.5: Atom-level physicochemical features aggregated per residue.

### C. Experiment Details

Outlines dataset sources, preprocessing, and experimental settings:

C.1: Describes four benchmark datasets (PPBS, GraphSet, DELPHI_Set, BCE) and their splits.

C.2: Lists implementation details, training environment, hyperparameter tuning, and settings for all baseline models compared.

### D. Statistical Analysis

Reports the standard deviation of AUPRC/AUROC across three random seeds for all models, and presents t-test results showing the statistical significance of COAT-GNN’s improvements over the strongest baselines.

### E. Computational Complexity

Analyzes the theoretical complexity of COAT-GNN compared with standard GNN and Transformer architectures, and explains that its dynamic coordinate update mechanism introduces only marginal overhead while enhancing conformational adaptability.



## II. Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## III. Data
You can download all the data mentioned in the paper from https://drive.google.com/drive/folders/1X2dRFDo5togPRORZkorK6ToB2no9KOK-?usp=drive_link (already anonymized) and put them in data/unity_data/.

We describe in detail how the features used for proteins are obtained in data/feature_process_example/.

## IV. Training

To train the model in the paper, run this command:
```train
python train.py
```
then you can choose the training of the specified dataset according to the prompts.

![example](https://github.com/user-attachments/assets/005c363d-29fd-483b-bef4-cc6dcdaa73f0)
## V. Evaluation

To evaluate model's performence on different dataset, run:

```eval
python evaluation.py
```
then you can choose the specified dataset according to the prompts, too.

## VI. Results

Our model achieves the following performance:

| Dataset        | AUPRC  | AUROC|
| ------------------ |---------------- | -------------- |
| PPBS-T_70   |     0.752        |      0.907       |
| PPBS-T_homo   |     0.740         |      0.913       |
| PPBS-T_topo   |    0.770         |      0.922       |
| PPBS-T_none   |     0.645         |      0.879       |
| GraphSet-T_60   |     0.614         |      0.879       |
| GraphSet-T_287   |     0.593         |      0.887      |
| DELPHI_Set   |    0.578         |      0.593       |
| BCE   |     0.220         |      0.705       |

