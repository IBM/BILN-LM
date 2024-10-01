"""Code adapted from the Pepland
Project: https://github.com/zhangruochi/pepland
"""
import os
import sys
from typing import List

import mlflow
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import dgl
from tqdm import tqdm

from omegaconf import OmegaConf
from .process import Mol2HeteroGraph


class Pepland:
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = self.load_cfg()
        self.model = self.load_model(cfg)
        self.device = torch.device("cuda:{}".format(
            cfg.inference.device_ids[0]) if torch.cuda.is_available()
            and len(cfg.inference.device_ids) > 0 else "cpu")
        self.model.to(self.device)
        self.pool = self.define_pooling(cfg.inference.pool).to(self.device)
        self.atom_index = cfg.inference.atom_index

    def define_pooling(self, pooling: str):
        if pooling == 'max':
            pool = nn.Sequential(
                Permute(),
                nn.AdaptiveMaxPool1d(output_size=1),
                Squeeze(dim=-1)
            )
        elif pooling == 'avg':
            pool = nn.Sequential(
                Permute(),
                nn.AdaptiveAvgPool1d(output_size=1),
                Squeeze(dim=-1)
            )
        return pool

    def load_cfg(self) -> dict:
        cfg_path = os.path.join(self.root_dir, "configs/inference.yaml")
        cfg = OmegaConf.load(cfg_path)
        return cfg

    def load_model(self, cfg: dict):
        model_path = os.path.join(self.root_dir, cfg.inference.model_path)
        sys.path.append(os.path.join(model_path, "code"))
        print("loading model from : {}".format(model_path))
        model = mlflow.pytorch.load_model(model_path, map_location="cpu")
        model.eval()
        return model

    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]
        node_size = bg.batch_num_nodes(ntype)
        start_index = torch.cat(
            [torch.tensor([0], device=device),
             torch.cumsum(node_size, 0)[:-1]])
        max_num_node = max(node_size)
        hidden_lst = []

        for i in range(bg.batch_size):
            start, size = start_index[i], node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d(
                (0, 0, 0, max_num_node - cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))
        hidden_lst = torch.cat(hidden_lst, 0)
        return hidden_lst

    def get_embeddings(self, smiles: List[str]) -> List[torch.Tensor]:
        graphs = []
        for i, smi in enumerate(tqdm(smiles)):
            try:
                sys.path.append(os.path.dirname(
                    os.path.dirname(self.root_dir)))
                graph = Mol2HeteroGraph(smi)
                graphs.append(graph)
            except Exception as e:
                print(e, 'invalid', smi)

        bg = dgl.batch(graphs)
        bg = bg.to(self.device)
        atom_embed, frag_embed = self.model(bg)
        bg.nodes['a'].data['h'] = atom_embed
        bg.nodes['p'].data['h'] = frag_embed
        atom_rep = self.split_batch(bg, 'a', 'h', self.device)

        # if set atom index, only return the atom embedding with the index
        if self.atom_index:
            pep_embeds = atom_rep[:, self.atom_index].detach().cpu()
        else:
            # if not set atom index, return the whole peptide embedding (atom + fragment)
            frag_rep = self.split_batch(bg, 'p', 'h', self.device)
            reps = torch.cat([atom_rep, frag_rep], dim=1)
            pep_embeds = self.pool(reps).detach().cpu()
        return pep_embeds


class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 1))


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)
