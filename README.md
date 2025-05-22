# COAT-GNN: Cooperative Attribute Learning and Topological Optimization for Protein Binding Sites Prediction

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Data
You can download all the data mentioned in the paper from https://drive.google.com/drive/folders/1X2dRFDo5togPRORZkorK6ToB2no9KOK-?usp=drive_link (already anonymized) and put them in data/unity_data/.

## Training

To train the model in the paper, run this command:

```train
python train_1.py
python train_2.py
```

then you can choose the training of the specified dataset according to the prompts. train_1.py is used to train on the datasets PPBS and BCE, and train_2.py is used to train on the datasets GraphSet and DELPHI.

![QQ截图20250522175815](https://github.com/user-attachments/assets/8446b5a7-253a-4e3b-a8ba-6d08fd8c2294)

## Evaluation

To evaluate model's performence on different dataset, run:

```eval
python evaluation.py
```
then you can choose the specified dataset according to the prompts, too.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
