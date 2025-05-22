# COAT-GNN: Cooperative Attribute Learning and Topological Optimization for Protein Binding Sites Prediction

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Data
You can download all the data mentioned in the paper from https://drive.google.com/drive/folders/1X2dRFDo5togPRORZkorK6ToB2no9KOK-?usp=drive_link (already anonymized) and put them in data/unity_data/.

We describe in detail how the features used for proteins are obtained in data/feature_process_example/.

## Training

To train the model in the paper, run this command:
```train
python train.py
```
then you can choose the training of the specified dataset according to the prompts.

![example](https://github.com/user-attachments/assets/005c363d-29fd-483b-bef4-cc6dcdaa73f0)
## Evaluation

To evaluate model's performence on different dataset, run:

```eval
python evaluation.py
```
then you can choose the specified dataset according to the prompts, too.

## Results

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

