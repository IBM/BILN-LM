# BILN_LM
BILN Language Model for describing modified and non-modified peptides

## 0. Installation

You will need to install the following packages:

```bash
pip install transformers[torch] datasets tokenizers mapchiral molfeat rdkit scipy scikit-learn tqdm optuna typer tensorboard lightgbm xgboost
```

```bash
pip install SmilesPE omegaconf
conda install dgl 
```

## 1. Download data

Execute the `download_data.py` script to download both the pretraining and benchmarking datasets. `data_dir_path` refers to the directory where you want
to save the files.

Both collections:

```bash
python code/download_data.py data_dir_path 
```

Only the pretraining data:

```bash
python code/download_data.py data_dir_path --collection pretraining
```

Only the downstream data:

```bash
python code/download_data.py data_dir_path --collection pretraining
```


## 2. Pretrain the model


Execute the `train.py` script. `log_dir` refers to the directory where the training logs will be saved. `--overwrite` flag can be used if you want to overwrite the `log_dir`.

```bash
python code/run_hpo.py log_dir `data_dir_path`
```

## 3. Evaluate model or reproduce baselines

To evaluate a pretrained model, execute the `fingerprint_evaluation.py`.

```bash
python code/fingerprint_evaluation.py data_dir_path BILN-LM:log_dir
```

