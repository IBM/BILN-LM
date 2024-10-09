import json
# import math
import operator
import os
from typing import Callable, Dict

import lightgbm
import numpy as np
import optuna
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gauss
import sklearn.neighbors as knn
import sklearn.svm as svm
import typer
import xgboost
import yaml
import warnings

from hestia.dataset_generator import (HestiaDatasetGenerator,
                                      SimilarityArguments)
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (matthews_corrcoef, root_mean_squared_error,
                             accuracy_score, f1_score,
                             precision_score, recall_score, mean_squared_error,
                             mean_absolute_error)
from tqdm import tqdm


def _pcc(preds, truths):
    return pearsonr(preds, truths)[0]


def _spcc(preds, truths):
    return spearmanr(preds, truths)[0]


def _f1_weighted(preds, truths):
    return f1_score(preds, truths, average='weighted')


def _recall(preds, truths):
    return recall_score(preds, truths, zero_division=True)


CLASSIFICATION_METRICS = {
    'mcc': matthews_corrcoef,
    'acc': accuracy_score,
    'f1': f1_score,
    'f1_weighted': _f1_weighted,
    'precision': precision_score,
    'recall': _recall
}

REGRESSION_METRICS = {
    'rmse': root_mean_squared_error,
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'pcc': _pcc,
    'spcc': _spcc
}

CLASSIFICATION_MODELS = {
    'svm': svm.SVC,
    'knn': knn.KNeighborsClassifier,
    'gauss': gauss.GaussianProcessClassifier,
    'rf': ensemble.RandomForestClassifier,
    'xgboost': xgboost.XGBClassifier,
    'lightgbm': lightgbm.LGBMClassifier
}

REGRESSION_MODELS = {
    'svm': svm.SVR,
    'knn': knn.KNeighborsRegressor,
    'gauss': gauss.GaussianProcessRegressor,
    'rf': ensemble.RandomForestRegressor,
    'xgboost': xgboost.XGBRegressor,
    'lightgbm': lightgbm.LGBMRegressor
}


REGRESSION_TASKS = ['c-binding', 'nc-binding', 'nc-cpp']
CLASSIFICATION_TASKS = ['c-cpp', 'c-sol']
TASKS = CLASSIFICATION_TASKS + REGRESSION_TASKS


class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""
    def __init__(self, early_stopping_rounds: int,
                 direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


def evaluate(preds, truth, pred_task) -> Dict[str, float]:
    result = {}
    if pred_task == 'reg':
        metrics = REGRESSION_METRICS
    else:
        metrics = CLASSIFICATION_METRICS

    for key, value in metrics.items():
        result[key] = value(preds, truth)
    return result


def define_hpspace(model: Callable, pred_task: str,
                   trial: optuna.Trial) -> dict:
    cwd = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(cwd, 'h_param_search',
                        f'{model}_{pred_task}.yml')
    hpspace = {}
    config = yaml.load(open(path), yaml.Loader)

    for key, item in config.items():
        if 'fixed' in item['type']:
            if item['type'].split('-')[1] == 'float':
                hpspace[key] = float(item['value'])
            elif item['type'].split('-')[1] == 'int':
                hpspace[key] = int(item['value'])
            else:
                hpspace[key] = item['value']
        elif item['type'] == 'bool':
            hpspace[key] = trial.suggest_categorical(
                key, choices=[True, False]
            )
        elif item['type'] == 'float':
            hpspace[key] = trial.suggest_float(
                key, low=float(item['min']),
                high=float(item['max']),
                log=bool(item['log'])
            )
        elif item['type'] == 'int':
            hpspace[key] = trial.suggest_int(
                key, low=int(item['min']),
                high=int(item['max']),
                log=bool(item['log'])
            )
        elif item['type'] == 'categorical':
            hpspace[key] = trial.suggest_categorical(
                key, choices=item['values']
            )
            if key == 'kernel':
                for name, value in item['extra_parameters'].items():
                    if name == hpspace[key]:
                        for subkey, subitem in value.items():
                            if subitem['type'] == 'fixed':
                                if subitem['type'].split('-')[1] == 'float':
                                    hpspace[subkey] = float(subitem['value'])
                                elif subitem['type'].split('-')[1] == 'int':
                                    hpspace[subkey] = int(subitem['value'])
                                else:
                                    hpspace[subkey] = subitem['value']
                            elif subitem['type'] == 'float':
                                hpspace[subkey] = trial.suggest_float(
                                    subkey, low=float(subitem['min']),
                                    high=float(subitem['max']),
                                    log=bool(subitem['log'])
                                )
                            elif subitem['type'] == 'int':
                                hpspace[subkey] = trial.suggest_int(
                                    subkey, low=int(subitem['min']),
                                    high=int(subitem['max']),
                                    log=bool(subitem['log'])
                                )
                            elif subitem['type'] == 'categorical':
                                hpspace[subkey] = trial.suggest_categorical(
                                    subkey, choices=subitem['values']
                                )
                            else:
                                raise ValueError("Subitem type: " +
                                                 f"{subitem['type']} " +
                                                 "does not exit.")

        else:
            raise ValueError(f"Item type: {item['type']} does not exit.")
    return hpspace


def hpo(pred_task: str, learning_algorithm: Callable,
        model_name: str,
        study: optuna.Study, train_x: np.ndarray,
        train_y: np.ndarray, valid_x: np.ndarray,
        valid_y: np.ndarray, n_trials: int,
        seed: int) -> dict:
    global best_model
    if pred_task == 'class':
        best_model = {'result': {'mcc': float('-inf')}}
    else:
        best_model = {'result': {'mse': float('inf')}}

    def hpo_objective(trial: optuna.Trial) -> float:
        global best_model
        hpspace = define_hpspace(model_name, pred_task,
                                 trial)
        if (not (model_name == 'svm' and pred_task == 'reg') and
           model_name != 'knn'):
            hpspace['random_state'] = seed
        model = learning_algorithm(**hpspace)
        model.fit(train_x, train_y)
        preds = model.predict(valid_x)
        result = evaluate(preds, valid_y, pred_task)
        if pred_task == 'class':
            if result['mcc'] > best_model['result']['mcc']:
                best_model = {
                    'model': model,
                    'config': hpspace,
                    'result': result
                }
            return result['mcc']
        else:
            if result['mse'] < best_model['result']['mse']:
                best_model = {
                    'model': model,
                    'config': hpspace,
                    'result': result
                }
            return -result['mse']
    study.optimize(hpo_objective, n_trials,
                   callbacks=[EarlyStoppingCallback(50, direction='maximize')],
                   show_progress_bar=True, gc_after_trial=False, n_jobs=10)
    return best_model


def define_hestia_generator(
    df: pd.DataFrame,
    similarity_metric: str,
    fp: str
) -> HestiaDatasetGenerator:
    sim_args = SimilarityArguments(
        data_type='small molecule', similarity_metric=fp,
        field_name='SMILES', min_threshold=0.,
        threads=8, bits=2_048, radius=2, distance=similarity_metric
    )
    hdg = HestiaDatasetGenerator(df)
    hdg.calculate_similarity(sim_args)
    hdg.calculate_partitions(
        label_name='labels', min_threshold=0., similarity_args=sim_args
    )
    return hdg


def experiment(dataset: str, model: str, similarity_metric: str,
               fp: str, representation: str,
               n_trials: int = 100, seed: int = 1):
    global best_model
    part_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'partitions'
    )
    data_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'downstream_data'
    )
    part_path = os.path.join(
        part_dir, f"{dataset}_{similarity_metric}_{fp}.gz")
    os.makedirs(part_dir, exist_ok=True)

    np.random.seed(seed)

    if dataset in REGRESSION_TASKS:
        pred_task = 'reg'
        learning_algorithm = REGRESSION_MODELS[model]
    elif dataset in CLASSIFICATION_TASKS:
        pred_task = 'class'
        learning_algorithm = CLASSIFICATION_MODELS[model]
    else:
        raise ValueError(
            f"Dataset: {dataset} not in tasks: {', '.join(TASKS)}")

    df = pd.read_csv(os.path.join(data_path, f'{dataset}.csv'))
    x = np.array(json.load(open(
        os.path.join('reps', f'{representation}_{dataset}.json'))))
    y = df.labels.to_numpy()
    results = []
    if os.path.exists(part_path):
        hdg = HestiaDatasetGenerator(df)
        hdg.from_precalculated(part_path)
    else:
        hdg = define_hestia_generator(df, similarity_metric, fp)
        hdg.save_precalculated(part_path)
    for th, partitions in hdg.get_partitions(filter=0.185):
        train_idx = partitions['train']
        valid_idx = partitions['valid']
        test_idx = partitions['test']
        if len(test_idx) < 0.18 * x.shape[0]:
            continue
        train_x, train_y = x[train_idx], y[train_idx]
        valid_x, valid_y = x[valid_idx], y[valid_idx]
        test_x, test_y = x[test_idx], y[test_idx]

        study = optuna.create_study(direction='maximize')
        best_model = hpo(pred_task, learning_algorithm, model, study,
                         train_x, train_y, valid_x, valid_y, n_trials,
                         seed)
        preds = best_model['model'].predict(test_x)
        result = evaluate(preds, test_y, pred_task)
        result.update({f"{key}_val": val for key, val in
                       best_model['result'].items()})
        result.update({'threshold': th, 'seed': seed})
        results.append(result)
    result_df = pd.DataFrame(results)
    return result_df


def main(dataset: str, model: str, similarity_metric: str,
         fp: str, representation: str,
         n_trials: int = 200, n_seeds: int = 5):
    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    results_dir = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'results_distance'
    )
    results_path = os.path.join(
        results_dir,
        f'{dataset}_{model}_{similarity_metric}_{fp}_{representation}.csv'
    )
    os.makedirs(results_dir, exist_ok=True)

    results_df = pd.DataFrame()
    for i in tqdm(range(n_seeds)):
        # print(f'Experiment seed: {i}')
        result_df = experiment(
            dataset, model, similarity_metric,
            fp, representation, n_trials, i
        )
        results_df = pd.concat([results_df, result_df])
    results_df.to_csv(results_path, index=False)


if __name__ == '__main__':
    typer.run(main)
