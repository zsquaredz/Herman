# Herman
This repository contains code for the EMNLP 2020 Findings paper [Reducing Quantity Hallucinations in Abstractive Summarization](https://aclanthology.org/2020.findings-emnlp.203/).

### Required packages
The code uses `python 3.6` with the following packages:

- `pytorch 1.2.0`
- `torchtext 0.4.0` 
- `spacy` (only needed for preprocessing data)
- `allennlp`
- `sklearn`

### Get the dataset
The dataset is in `csv` format with `id`, `article`, `summary`, `label`, `label_binary`, `quantity_label` fields

- `article`: the original article
- `summary`: the summary of the article
- `label`: the tag sequence `Y` described in the paper Sec 3.1
- `label_binary`: the binary summary-level label `z` described in the paper Sec 3.1
- `quantity_label`: the sequence of binary labels `M` described in the paper Sec 3.1

You can download the dataset [here](https://drive.google.com/file/d/1N5271KSvY6U2vt-spipPhlvQDtFPUpdL/view?usp=sharing). Unzip the data using the following command:
```
tar xvzf herman_data.tar.gz 
```

### How to run
Use the following command to train the Herman system 
```
python run.py --task=train --checkpoint=checkpoint_name --file_path=dataset_path --file_train=train_data_filename
```
The code can be trained on 1 Nvidia GTX1080Ti GPU with 11GB memory with `batch_size` 32. 

To use the trained checkpoint and make label prediction and calculate scores in order to re-rank summaries, use the following code:
```
python predict.py --task=test --checkpoint=checkpoint_path --file_path=dataset_path --file_output=output_file_path --file_test=test_data_filename
``` 
Note that the test set should be beams of summaries generated from any summarization system. The test set need to be in `csv` format with `article`, `summary`, `quantity_label` fields.

After running the code, you will get three output files, one with the predicted labels, one with the global scores, and one with the local scores. Both the global score and local score can be used to re-rank summaries. 
