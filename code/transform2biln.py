from typing import List
from multiprocessing import Pool, cpu_count

import pandas as pd
from pyPept.converter import Converter
from tqdm import tqdm
import typer


def parallel_helm2biln(
    list_helm: List[str],
    n_threads: int = cpu_count()
) -> List[str]:
    """Small function for translating from HELM to BILN notation

    :param list_helm: List of helm peptides
    :type list_helm: List[str]
    :param n_threads: Number of threads for parallelization,
    defaults to cpu_count()
    :type n_threads: int, optional
    :return: List of BILN peptides
    :rtype: List[str]
    """
    pool = Pool(n_threads)
    output = pool.map(helm2biln, tqdm(list_helm))
    return output


def helm2biln(helm: str) -> str:
    b = Converter(helm=helm)
    biln = b.get_biln()
    return biln


if __name__ == '__main__':
    df = pd.read_csv('data/all_peptides_helm_and_smiles_chembl28.csv')
    print(len(df))
    print(df.head())
    df = df[df.helm_notation != '']
    df = df[~df.helm_notation.isna()].reset_index(drop=True)
    print(len(df))
    df['biln'] = parallel_helm2biln(df.helm_notation)
    df.to_csv('data/biln_db.csv', index=False)
