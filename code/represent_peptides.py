import json
import os

import numpy as np
import pandas as pd
import torch
import transformers as hf
import typer

from pepfunn.similarity import monomerFP
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from utils.pepclm_tokenizer import SMILES_SPE_Tokenizer
from utils.pepland_inference.inference_pepland import Pepland


def calculate_ecfp(dataset: str):
    out_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps', f'ecfp_{dataset}.json'
    )
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    if os.path.exists(out_path):
        return json.load(open(out_path))
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=8, fpSize=2_048
    )

    def _get_fp(smile: str):
        mol = Chem.MolFromSmiles(smile)
        fp = fpgen.GetFingerprintAsNumPy(mol).astype(np.int8)
        return fp

    fps = thread_map(
        _get_fp, df['SMILES'], max_workers=8
    )
    fps = np.stack(fps).tolist()
    json.dump(fps, open(os.path.join(out_path), 'w'))


def calculate_molformer(dataset: str):
    device = 'cpu'
    batch_size = 32
    out_path = os.path.join(os.path.dirname(__file__),
        '..', '..', 'reps', f'molformer_{dataset}.json')
    if os.path.exists(out_path):
        return json.load(open(out_path))
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    tokenizer = hf.AutoTokenizer.from_pretrained(
        'ibm/MoLFormer-XL-both-10pct', trust_remote_code=True
    )
    model = hf.AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct',
                                         trust_remote_code=True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params / 1e6 < 1e3:
        print(f'Number of model parameters are: {n_params/1e6:.1f} M')
    else:
        print(f'Number of model parameters are: {n_params/1e9:.1f} B')
    smiles = df['SMILES'].tolist()
    batched = [smiles[i:i+batch_size] for i in
               range(0, len(smiles), batch_size)]
    fps = []
    for batch in tqdm(batched):
        input_ids = tokenizer(batch, return_tensors='pt',
                              padding='longest').to(device)
        with torch.no_grad():
            vector = model(**input_ids).last_hidden_state
            mask = input_ids['attention_mask']
            for i in range(mask.shape[0]):
                length = mask[i].sum()
                fps.append(vector[i, :length].mean(0).detach().cpu().tolist())
    json.dump(fps, open(os.path.join(out_path), 'w'))


def calculate_bilnlm(dataset: str):
    device = 'cpu'
    batch_size = 8
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    out_path = os.path.join(os.path.dirname(__file__),
        '..', '..', 'reps', f'bilnlm_{dataset}.json')
    if os.path.exists(out_path):
        return json.load(open(out_path))

    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))

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
    smiles = df['SMILES'].tolist()
    batched = [smiles[i:i+batch_size] for i in
               range(0, len(smiles), batch_size)]
    fps = []
    for batch in tqdm(batched):
        input_ids = tokenizer(batch, return_tensors='pt',
                              padding='longest').to(device)
        del input_ids['token_type_ids']
        with torch.no_grad():
            vector = model(**input_ids).last_hidden_state
            mask = input_ids['attention_mask']
            for i in range(mask.shape[0]):
                length = mask[i].sum()
                fps.append(vector[i, :length].mean(0).detach().cpu().tolist())
    json.dump(fps, open(out_path, 'w'))


def calculate_pepclm(dataset: str):
    device = 'cpu'
    batch_size = 8
    out_path = os.path.join(os.path.dirname(__file__),
        '..', '..', 'reps', f'pepclm_{dataset}.json')
    if os.path.exists(out_path):
        return json.load(open(out_path))
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    tokenizer = SMILES_SPE_Tokenizer(
        os.path.join(os.path.dirname(__file__), 'utils',
                     'tokenizer', 'new_vocab.txt'),
        os.path.join(os.path.dirname(__file__), 'utils',
                     'tokenizer', 'new_splits.txt')
    )
    model = hf.AutoModel.from_pretrained('aaronfeller/PeptideCLM-23M-all',
                                         trust_remote_code=True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params / 1e6 < 1e3:
        print(f'Number of model parameters are: {n_params/1e6:.1f} M')
    else:
        print(f'Number of model parameters are: {n_params/1e9:.1f} B')
    smiles = df['SMILES'].tolist()
    batched = [smiles[i:i+batch_size] for i in
               range(0, len(smiles), batch_size)]
    fps = []
    for batch in tqdm(batched):
        input_ids = tokenizer(batch, return_tensors='pt',
                              padding='longest').to(device)
        with torch.no_grad():
            vector = model(**input_ids).last_hidden_state
            mask = input_ids['attention_mask']
            for i in range(mask.shape[0]):
                length = mask[i].sum()
                fps.append(vector[i, :length].mean(0).detach().cpu().tolist())
    json.dump(fps, open(os.path.join(out_path), 'w'))


def calculate_pepfunnfp(dataset: str):
    out_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps', f'pepfunn_{dataset}.json'
    )
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    if os.path.exists(out_path):
        return json.load(open(out_path))
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))

    def _get_fp(smile: str):
        fp = monomerFP(smile, radius=2, nBits=2_048)
        return fp

    fps = thread_map(
        _get_fp, df['BILN'], max_workers=8
    )
    fps = np.stack(fps).tolist()
    json.dump(fps, open(os.path.join(out_path), 'w'))


def calculate_pepland(dataset: str):
    out_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps', f'pepland_{dataset}.json'
    )
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    if os.path.exists(out_path):
        return json.load(open(out_path))
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    pepland = Pepland()
    fps = pepland.get_embeddings(df.SMILES.tolist())
    fps = np.stack(fps).tolist()
    json.dump(fps, open(os.path.join(out_path), 'w'))


def main(dataset: str, rep: str):
    if rep == 'ecfp':
        print('Calculating ECFP representations...')
        calculate_ecfp(dataset)
    elif rep == 'molformer':
        print('Calculating MolFormer-XL representations...')
        calculate_molformer(dataset)
    elif rep == 'bilnlm':
        print('Calculating BILN-LM representations...')
        calculate_bilnlm(dataset)
    elif rep == 'pepclm':
        print('Calculating PeptideCLM representations...')
        calculate_pepclm(dataset)
    elif rep == 'pepland':
        print('Calculating Pepland representations...')
        calculate_pepland(dataset)
    # print('Calculating pepfunn fingerprint...')
    # calculate_pepfunnfp(dataset)


if __name__ == '__main__':
    typer.run(main)
