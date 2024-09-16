import os
import shutil
import time
import urllib
import urllib.request as request

import pandas as pd
import typer

import rdkit.Chem as Chem
from pyPept.converter import Converter


def fasta2smiles(seq: str) -> str:
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def helm2biln(helm: str) -> str:
    if isinstance(helm, float):
        return pd.NA
    b = Converter(helm=helm)
    biln = b.get_biln()
    return biln


def fasta2biln(seq: str) -> str:
    return '-'.join(seq)


def download_all(data_path: str, collection: str = 'all'):
    if os.path.exists(data_path):
        pass
        # raise RuntimeError(f'Warning! Path: {data_path} already exists')
    else:
        os.makedirs(data_path)
    if collection == 'all' or collection == 'downstream':
        download_downstream_data(data_path)
    if collection == 'all' or collection == 'pretraining':
        download_pretraining(data_path)


def download_pretraining(data_path: str) -> None:
    print('Downloading pretraining data...')
    out_path = os.path.join(data_path, 'pretrain.zip')
    tmp_dir = os.path.join(data_path, f'{time.time()}')
    tmp2_file = os.path.join(tmp_dir, 'SI lookup tables and datasets',
                             'datasets', 'chembl',
                             'all_peptides_helm_and_smiles_chembl28.xlsx')
    outfile = os.path.join(data_path, 'biln_db.csv')

    url = 'https://www.biorxiv.org/content/biorxiv/early/2021/10/28/2021.10.26.465927/DC1/embed/media-1.zip?download=true'
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    shutil.unpack_archive(out_path, tmp_dir)
    df = pd.read_excel(tmp2_file)
    df['biln'] = df['helm_notation'].map(helm2biln)
    df.to_csv(outfile)
    shutil.rmtree(tmp_dir)
    os.remove(out_path)
    print('Pretraining data downloaded successfully!')


def download_downstream_data(data_path: str) -> None:
    print('Downloading Canonical Solubility dataset...')
    status = download_c_solubility(data_path)
    if status:
        print('Canonical Solubility dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print('Downloading Canonical Cell Penetration dataset...')
    status = download_c_cpp(data_path)
    if status:
        print('Canonical Cell Penetration dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print('Downloading Non-canonical Cell Penetration dataset...')
    status = download_nc_cpp(data_path)
    if status:
        print('Non-canonical Cell Penetration dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print('Downloading Canonical binding dataset...')
    status = download_c_binding(data_path)
    if status:
        print('Canonical binding dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print('Downloading Non-canonical binding dataset...')
    status = download_nc_binding(data_path)
    if status:
        print('Non-canonical binding dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')


def download_c_solubility(data_path: str) -> bool:
    out_path = os.path.join(data_path, 'c-sol.csv')
    url = 'https://raw.githubusercontent.com/zhangruochi/pepland/master/data/eval/c-Sol.txt'
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    df = pd.read_csv(out_path, header=None, names=['sequence', 'labels'])
    df['BILN'] = df['sequence'].apply(fasta2biln)
    df['SMILES'] = df['sequence'].apply(fasta2smiles)
    df.dropna(inplace=True)
    df.to_csv(out_path, index=False)
    return True


def download_c_cpp(data_path: str) -> bool:
    out_path = os.path.join(data_path, 'c-cpp.csv')
    url = 'https://raw.githubusercontent.com/zhangruochi/pepland/master/data/eval/c-CPP.txt'
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    df = pd.read_csv(out_path, header=None, names=['sequence', 'labels'])
    df['BILN'] = df['sequence'].apply(fasta2biln)
    df['SMILES'] = df['sequence'].apply(fasta2smiles)
    df.dropna(inplace=True)
    df.to_csv(out_path, index=False)
    return True


def download_nc_cpp(data_path: str) -> bool:
    out_path = os.path.join(data_path, 'nc-cpp.csv')
    url = 'https://raw.githubusercontent.com/zhangruochi/pepland/master/data/eval/nc-CPP.csv'
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    df = pd.read_csv(out_path)
    df['labels'] = df['PAMPA']
    df = df[['SMILES', 'HELM', 'labels']]
    df['BILN'] = df['HELM'].apply(helm2biln)
    df.dropna(inplace=True)
    df.to_csv(out_path, index=False)
    return True


def download_nc_binding(data_path: str) -> bool:
    out_path = os.path.join(data_path, 'nc-binding.csv')
    url = "https://raw.githubusercontent.com/zhangruochi/pepland/master/data/eval/nc-binding.csv"
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    df = pd.read_csv(out_path)
    df['SMILES'] = df['Merge_SMILES']
    df['BILN'] = df['pep_SEQRES']
    df['labels'] = df['affinity']
    df = df[['seq1', 'SMILES', 'BILN', 'labels']]
    df.dropna(inplace=True)
    df.to_csv(out_path, index=False)
    return True


def download_c_binding(data_path: str) -> bool:
    out_path = os.path.join(data_path, 'c-binding.csv')
    url = "https://raw.githubusercontent.com/zhangruochi/pepland/master/data/eval/c-binding.csv"
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    df = pd.read_csv(out_path)
    df['SMILES'] = df['Merge_SMILES']
    df['BILN'] = df['pep_SEQRES']
    df['labels'] = df['affinity']
    df = df[['seq1', 'SMILES', 'BILN', 'labels']]
    df.dropna(inplace=True)
    df.to_csv(out_path, index=False)
    return True


if __name__ == '__main__':
    typer.run(download_all)
