# TE4SRec

This repository is the public version of the research project about Temporal Encoding for Sequential Recommendations (
TE4SRec).

All the model networks and training pipelines are based on PyTorch framework.

## Quick Start

Start training TE4SRec-b with the following command on Steam dataset.

```
python main.py --dataset_name steam --is_bert True --max_seq_len 50 --num_worker 1 --train_batch_size 512 --pred_batch_size 128
```

## Usage

To run main.py, there are parameters to be considered. The types, descriptions and default values of these parameters
are listed as follows.
To change the parameters use --{param_name} {param_value} after main.py.

| Parameter        | Default Value      | Type  | Description                                                               |
|------------------|--------------------|-------|---------------------------------------------------------------------------|
| seed             | 42                 | int   | Random seed for reproducibility.                                          |
| output_path      | './output/results' | str   | Path to save the output files.                                            |
| data_dir         | './data'           | str   | Directory path for dataset storage.                                       |
| dataset_name     | 'yelp'             | str   | Name of the dataset to be used.                                           |
| pred_threshold   | 3                  | int   | Threshold of sequence length for prediction.                              |
| is_bert          | True               | bool  | Flag to determine whether BERT or left-to-right attention is used.        |
| model_name       | 'TE4SRec'          | str   | Name of the model to be loaded.                                           |
| load_model       | False              | bool  | Flag to determine if a pre-trained model should be loaded.                |
| max_seq_len      | 50                 | int   | Maximum sequence length for the model.                                    |
| alpha            | 0.5                | float | Temporal scaler.                                                          |
| d_temporal       | 50                 | int   | Dimension of temporal features.                                           |
| d_model          | 50                 | int   | Dimension of the model.                                                   |
| num_block        | 2                  | int   | Number of Transformer blocks in the model architecture.                   |
| num_heads        | 1                  | int   | Number of attention heads in the model.                                   |
| is_gelu          | True               | bool  | Flag to determine whether GELU or ReLU to be used as activation function. |
| dropout_rate     | 0.2                | float | Dropout rate for regularization.                                          |
| mask_ratio       | 0.2                | float | Ratio of masking for BERT training.                                       |
| pretrain_epoch   | 0                  | int   | Number of epochs for pre-training without evaluation metrics.             |
| stop_patience    | 10                 | int   | Patience for early stopping during training.                              |
| train_batch_size | 512                | int   | Batch size for training.                                                  |
| num_worker       | 1                  | int   | Number of workers for data loading in parallel (Torch standard).          |
| lr               | 0.001              | float | Learning rate for the optimizer.                                          |
| adam_beta1       | 0.9                | float | Beta1 parameter for Adam optimizer.                                       |
| adam_beta2       | 0.98               | float | Beta2 parameter for Adam optimizer.                                       |
| is_sampling      | False              | bool  | Flag to enable sampling in evaluation (according to SASRec).              |
| neg_num          | 100                | int   | Number of negative samples for evaluation (only for is_sampling=True).    |
| pred_batch_size  | 128                | int   | Batch size for prediction.                                                |

## Dataset Collection

In the project's directory structure, the raw data files are systematically organized under ./data-raw/{dataset_name}.
Following preprocessing, the refined data is stored in ./data/{dataset_name}/data.csv. All datasets have been sourced
from publicly available resources.

- [Steam](https://cseweb.ucsd.edu/~jmcauley/datasets.html)
- [Yelp](https://www.yelp.com/dataset/download)
- [Goodreads](https://mengtingwan.github.io/data/goodreads#overview)
- [RateBeer](https://cseweb.ucsd.edu/~jmcauley/datasets.html)

## Data Preprocessing

We preprocess datasets with two core parameters about sampling in this project. The users and items can be sampled
randomly to reduce the numbers.

- sample types: ['sample_item', 'sample_user']
- sample ratios

```
python preprocess.py
```

Once preprocessing is complete, execute the command below to visualize metadata along with distributions of user-based,
item-based, and temporal information. The results are stored in the directory ./output/data/{dataset_name} as default,
so make sure the directory existing.

```
python data_view.py
```

## Project Structure

```
│   .gitignore
│   dataset.py
│   data_view.py
│   LICENSE
│   main.py
│   preprocess.py
│   README.md
│   result_view.py
│   trainer.py
│   utils.py
│
├───model
│   │   attention.py
│   │   feedforward.py
│   │   TE4SRec.py
│   │   temporal_encoding.py
│
├───data
│   ├───steam
│   │       data.csv
│   └───yelp
│
├───data-raw
│   ├───steam
│   │       steam_new.json
│   └───yelp
│
├───output
│   ├───data
│   │   ├───steam
│   │   │       basic.txt
│   │   │       item_day_gap.png
│   │   │       item_freq_count.png
│   │   │       month_action.png
│   │   │       seq_day_gap.png
│   │   │       seq_len_count.png
│   │
│   └───results
│       ├───steam
│       │   ├───TE4SRec-b
│       │   │       config
│       │   │       eval_valid_line.png
│       │   │       model_checkpoint.pt
│       │   │       temporal_encoding.png
│       │   │       test_metrics.csv
│       │   │       top_k_tables.csv
│       │   │       valid_metrics.csv
```
