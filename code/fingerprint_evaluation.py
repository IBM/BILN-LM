import json
import math
import os

from typing import Dict, List

import optuna
import torch
import typer

import datamol as dm
import numpy as np
import pandas as pd
import transformers as hf

from datasets import Dataset
from mapchiral.mapchiral import encode
from molfeat.trans.fp import FPVecTransformer
from rdkit.Chem import MolFromSmiles
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor,
                              GradientBoostingClassifier)
from sklearn.metrics import (root_mean_squared_error, matthews_corrcoef,
                             accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from tqdm import tqdm


REGRESSION_TASKS = ['c-binding', 'nc-binding', 'nc-cpp']
CLASSIFICATION_TASKS = ['c-sol', 'c-cpp']
TASKS = REGRESSION_TASKS + CLASSIFICATION_TASKS
MULTI_INSTANCE_TASKS = ['c-binding', 'nc-binding']


def represent_peptides(ds: Dict[str, Dataset], fingerprint: str,
                       device: str) -> Dict[str, list]:
    fps = {}

    if 'map4c' in fingerprint:
        radius = int(fingerprint.split('-')[0].split(':')[1]) // 2
        n_bits = int(fingerprint.split('-')[1])
        # print(radius, n_bits)
        for key, dataset in ds.items():
            fps[key] = [encode(MolFromSmiles(x), max_radius=radius,
                               n_permutations=n_bits, mapping=False)
                        for x in tqdm(dataset['SMILES'])]
    elif ('ecfp' in fingerprint or 'maccs' in fingerprint or
          'rdkit' in fingerprint):
        if '-' in fingerprint and 'count' not in fingerprint:
            calc = FPVecTransformer(fingerprint.split('-')[0],
                                    int(fingerprint.split('-')[1]),
                                    n_jobs=4, dtype=np.int8,)
                                    # parallel_kwargs={"progress": True})
        elif '-count' in fingerprint:
            calc = FPVecTransformer('-'.join(fingerprint.split('-')[:2]),
                                    int(fingerprint.split('-')[2]),
                                    n_jobs=4, dtype=np.int8,)
        else:
            calc = FPVecTransformer(fingerprint, n_jobs=4, dtype=np.int8,)
                                    # parallel_kwargs={"progress": True})

        for key, dataset in ds.items():
            with dm.without_rdkit_log():
                fps[key] = calc(dataset['SMILES'])

    elif 'BILN-LM' in fingerprint:
        log_dir = fingerprint.split(':')[1]
        params_path = os.path.join(log_dir, 'best_model_hparams.json')
        tokenizer_path = os.path.join(log_dir, 'best_tokenizer.json')
        state_dict_path = os.path.join(log_dir, 'best_model_st_dict.pt')
        config = json.load(open(params_path))
        del config['learning_rate']
        del config['model_size']
        tokenizer = hf.PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path, padding=True
        )
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]", 'mask_token': "[MASK]"}
        )
        model_config = hf.EsmConfig(
            vocab_size=2026, vocab_list=tokenizer.vocab,
            pad_token_id=3, **config
        )
        model = hf.AutoModel.from_config(model_config)
        base_state_dict = model.state_dict()
        state_dict = torch.load(state_dict_path)
        new_state_dict = state_dict.copy()
        for key, value in state_dict.items():
            if 'esm' in key:
                del new_state_dict[key]
                new_state_dict['.'.join(key.split('.')[1:])] = value
            elif key not in base_state_dict:
                del new_state_dict[key]

        for key, value in base_state_dict.items():
            if key not in new_state_dict:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict)
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params / 1e6 < 1e3:
            print(f'Number of model parameters are: {n_params/1e6:.1f} M')
        else:
            print(f'Number of model parameters are: {n_params/1e9:.1f} B')
        batch_size = 8

        for key, dataset in ds.items():
            smiles = dataset['SMILES']
            batched = [smiles[i:i+batch_size] for i in
                       range(0, len(smiles), batch_size)]
            fps[key] = []
            for batch in tqdm(batched):
                input_ids = tokenizer(batch, return_tensors='pt',
                                      padding='longest').to(device)
                del input_ids['token_type_ids']
                with torch.no_grad():
                    vector = model(**input_ids).last_hidden_state
                    mask = input_ids['attention_mask']
                    for i in range(mask.shape[0]):
                        length = mask[i].sum()
                        fps[key].append(vector[i, :length].mean(0).detach().cpu().tolist())
        model = model.cpu()
        del model, tokenizer, input_ids, state_dict, new_state_dict
    elif 'MolFormer' in fingerprint:
        tokenizer = hf.AutoTokenizer.from_pretrained(
            'ibm/MoLFormer-XL-both-10pct', trust_remote_code=True
        )
        model = hf.AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params / 1e6 < 1e3:
            print(f'Number of model parameters are: {n_params/1e6:.1f} M')
        else:
            print(f'Number of model parameters are: {n_params/1e9:.1f} B')
        batch_size = 32

        for key, dataset in ds.items():
            smiles = dataset['SMILES']
            batched = [smiles[i:i+batch_size] for i in
                       range(0, len(smiles), batch_size)]
            fps[key] = []
            for batch in tqdm(batched):
                input_ids = tokenizer(batch, return_tensors='pt',
                                      padding='longest').to(device)
                # del input_ids['token_type_ids']
                with torch.no_grad():
                    vector = model(**input_ids).last_hidden_state
                    mask = input_ids['attention_mask']
                    for i in range(mask.shape[0]):
                        length = mask[i].sum()
                        fps[key].append(vector[i, :length].mean(0).detach().cpu().tolist())
        model = model.cpu()
        del model, tokenizer, input_ids
    return fps


def represent_proteins(sequences: List[str], data_path: str, task: str,
                       device: str) -> List[np.ndarray]:
    save_file = os.path.join(data_path, f'prot_{task}.json')
    if os.path.exists(save_file):
        print('Loading representations...')
        reprs = json.load(open(save_file))
        return reprs

    print('Computing protein representations...')
    batch_size = 16
    model_name = 'facebook/esm2_t12_35M_UR50D'
    model = hf.AutoModel.from_pretrained(model_name).to(device)
    tokenizer = hf.AutoTokenizer.from_pretrained(model_name)
    batched = [sequences[i:i+batch_size] for i in
               range(0, len(sequences), batch_size)]
    input_ids = [tokenizer(batch, return_tensors='pt',
                           padding='longest').to(device) for batch in batched]
    reprs = []
    for batch, input_id in tqdm(zip(batched, input_ids), total=len(batched)):
        with torch.no_grad():
            vector = model(**input_id).last_hidden_state
            for i, seq in enumerate(batch):
                reprs.append(vector[i, :len(seq)].mean(0).detach().cpu().tolist())
    json.dump(reprs, open(save_file, 'w'))
    del model, tokenizer
    return reprs


def run_hpo(model_algorithm: str, task: str, fps, ds, log_dir: str):
    global counter, prev_loss
    counter, prev_loss = 0, math.inf
    study = optuna.create_study(direction='minimize')

    def optim_objective(trial: optuna.Trial) -> float:
        global counter, prev_loss

        if model_algorithm == 'svm':
            config = {
                # 'kernel': trial.suggest_categorical(
                    # 'kernel', ['linear', 'poly', 'rbf', 'sigmoid']
                # ),
                'kernel': 'linear',
                'C': trial.suggest_float('C', low=1e-3, high=1e1, log=True),
                'epsilon': trial.suggest_float('epsilon', low=1e-3, high=1e3, log=True),
                'max_iter': 1_000_000
            }
        elif model_algorithm == 'rf':
            config = {
                'n_estimators': trial.suggest_int(
                    'n_estimators', 1, 1e3
                ),
                'max_depth': trial.suggest_int(
                    'max_depth', 1, 1e2
                ),
                'min_impurity_decrease': trial.suggest_float(
                    'min_impurity_decrease', 1e-7, 1, log=True
                ),
                'warm_start': trial.suggest_categorical('warm_start', [False, True]),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-12, 1, log=True),
                'n_jobs': 8
            }
        elif model_algorithm == 'xgboost':
            config = {
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-7, 1, log=True
                ),
                'n_estimators': trial.suggest_int('n_estimators', 1, 1e3),
                'subsample': trial.suggest_float('subsample', 0.1, 1),
                'max_depth': trial.suggest_int(
                    'max_depth', 1, 1e2
                ),
                'min_impurity_decrease': trial.suggest_float(
                    'min_impurity_decrease', 1e-7, 1, log=True
                ),
                'warm_start': trial.suggest_categorical('warm_start', [False, True]),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-12, 1, log=True)
            }
        elif model_algorithm == 'knn':
            config = {
                'n_neighbors': trial.suggest_int('n_estimators', 1, len(ds['train'])//10),
                'weights': trial.suggest_categorical('weights', ['distance', 'uniform']),
                'p': 1,
                'n_jobs': 8
            }

        if task in REGRESSION_TASKS:
            if model_algorithm == 'svm':
                model = SVR(**config)
            elif model_algorithm == 'rf':
                model = RandomForestRegressor(**config)
            elif model_algorithm == 'xgboost':
                model = GradientBoostingRegressor(**config)
            elif model_algorithm == 'knn':
                model = KNeighborsRegressor(**config)
            else:
                raise ValueError(f'Model architecture: {model_algorithm} is not currently supported')
        else:
            if model_algorithm == 'svm':
                del config['epsilon']
                model = SVC(**config)
            elif model_algorithm == 'rf':
                model = RandomForestClassifier(**config)
            elif model_algorithm == 'xgboost':
                model = GradientBoostingClassifier(**config)
            elif model_algorithm == 'knn':
                model = KNeighborsClassifier(**config)
            else:
                raise ValueError(f'Model architecture: {model_algorithm} is not currently supported')

        counter += 1
        model.fit(fps['train'], ds['train']['labels'])
        preds = model.predict(fps['valid'])
        if task in REGRESSION_TASKS:
            loss = root_mean_squared_error(ds['valid']['labels'], preds)
        else:
            loss = - matthews_corrcoef(ds['valid']['labels'], preds)
        if loss < prev_loss:
            prev_loss = loss
            config.update({'model': model_algorithm})
            json.dump(config, open(os.path.join(log_dir, 'best_model.json'), 'w'),
                      indent=2)
        return loss

    study.optimize(optim_objective, n_trials=10, show_progress_bar=True,
                   gc_after_trial=True)


def load_data(data_path: str, task: str, seed: int, device: str) -> Dict[str, Dataset]:
    filepath = os.path.join(data_path, f'{task}.csv')
    df = pd.read_csv(filepath)

    if task in MULTI_INSTANCE_TASKS:
        df['protein'] = represent_proteins(df['seq1'].tolist(), data_path,
                                           task, device)

    if task in CLASSIFICATION_TASKS:
        train, test = train_test_split(df, test_size=0.2, random_state=seed,
                                       stratify=df['labels'])
        train, valid = train_test_split(train, test_size=0.1,
                                        random_state=seed)
    else:
        train, test = train_test_split(df, test_size=0.2, random_state=seed)
        train, valid = train_test_split(train, test_size=0.1,
                                        random_state=seed)

    ds = {
        'train': Dataset.from_pandas(train),
        'valid': Dataset.from_pandas(valid),
        'test': Dataset.from_pandas(test)
    }
    if task in CLASSIFICATION_TASKS:
        for key, value in ds.items():
            ds[key] = value.class_encode_column('labels')
    return ds


def run_experiment(
    data_path: str,
    fingerprint: str,
    model_algorithm: str,
    log_dir: str,
    device: str = 'cpu',
    seed: int = 1
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    results = []
    outputfile = os.path.join(log_dir, 'results.csv')
    # TASKS = TASKS.pop(-1)
    for task in TASKS:
        print(f'Currently evaluating on {task}...')
        ds = load_data(data_path, task, seed, device)
        fps = represent_peptides(ds, fingerprint, device)
        if task in MULTI_INSTANCE_TASKS:
            for key, dataset in ds.items():
                fps[key] = np.concatenate([fps[key], np.array(dataset['protein'])],
                                          axis=1)

        run_hpo(model_algorithm, task, fps, ds, log_dir)
        params = json.load(open(os.path.join(log_dir, 'best_model.json')))
        del params['model']
        if task in REGRESSION_TASKS:
            if model_algorithm == 'svm':
                model = SVR(**params)
            elif model_algorithm == 'rf':
                model = RandomForestRegressor(**params)
            elif model_algorithm == 'xgboost':
                model = GradientBoostingRegressor(**params)
            elif model_algorithm == 'knn':
                model = KNeighborsRegressor(**params)
            else:
                raise ValueError(f'Model architecture: {model_algorithm} is not currently supported')
        else:
            if model_algorithm == 'svm':
                model = SVC(**params)
            elif model_algorithm == 'rf':
                model = RandomForestClassifier(**params)
            elif model_algorithm == 'xgboost':
                model = GradientBoostingClassifier(**params)
            elif model_algorithm == 'knn':
                model = KNeighborsClassifier(**params)
            else:
                raise ValueError(f'Model architecture: {model_algorithm} is not currently supported')

        model.fit(fps['train'], ds['train']['labels'])
        preds = model.predict(fps['test'])
        metrics = {'task': task, 'fingerprint': fingerprint,
                   'model': model_algorithm}

        if task in REGRESSION_TASKS:
            rmse = root_mean_squared_error(ds['test']['labels'], preds)
            pcc = pearsonr(ds['test']['labels'], preds)[0]
            spcc = spearmanr(ds['test']['labels'], preds)[0]
            metrics.update({'rmse': rmse, 'pcc': pcc, 'spcc': spcc})
        else:
            acc = accuracy_score(ds['test']['labels'], preds)
            mcc = matthews_corrcoef(ds['test']['labels'], preds)
            f1 = f1_score(ds['test']['labels'], preds)
            metrics.update({'acc': acc, 'mcc': mcc, 'f1': f1})
        results = [metrics]

        results = pd.DataFrame(results)

        if os.path.exists(outputfile):
            prev_results = pd.read_csv(outputfile)
            results = pd.concat([prev_results, results])

        results.to_csv(outputfile, index=False)


if __name__ == '__main__':
    typer.run(run_experiment)
