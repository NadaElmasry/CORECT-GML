import sys
import random
import logging

import numpy as np
import torch
import pickle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
    print("Seed set", seed)


def get_logger(level=logging.INFO):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def feature_packing(multimodal_feature, lengths):
        batch_size = lengths.size(0)
        node_features = []

        for feature in multimodal_feature:
            for j in range(batch_size):
                cur_len = lengths[j].item()
                node_features.append(feature[j,:cur_len])

        node_features = torch.cat(node_features, dim=0)

        return node_features


def multi_concat(nodes_feature, lengths, n_modals):
    sum_length = lengths.sum().item()
    return torch.cat(torch.split(nodes_feature, sum_length, dim=0), dim=-1)
