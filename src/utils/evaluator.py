import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import stats

def evaluate(prediction, ground_truth, mask):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'

    performance = {}

    loader_ic_values = []
    loader_rank_ic_values = []

    for i in range(prediction.shape[1]):
        pred_slice = prediction[:, i]
        truth_slice = ground_truth[:, i]
        mask_slice = mask[:, i]
        
        valid_mask = (mask_slice > 0.5) & ~np.isnan(truth_slice) & ~np.isnan(pred_slice)
        
        valid_pred = pred_slice[valid_mask]
        valid_truth = truth_slice[valid_mask]

        if len(valid_pred) >= 20:
            try:
                if np.std(valid_pred) != 0 and np.std(valid_truth) != 0:
                    ic, _ = stats.pearsonr(valid_pred, valid_truth)
                    loader_ic_values.append(ic)
                else:
                    loader_ic_values.append(np.nan)

                pred_ranks = stats.rankdata(valid_pred, method='average')
                truth_ranks = stats.rankdata(valid_truth, method='average')
                
                if len(np.unique(pred_ranks)) >= 20 and len(np.unique(truth_ranks)) >= 20:
                    rank_ic, _ = stats.spearmanr(pred_ranks, truth_ranks)
                    loader_rank_ic_values.append(rank_ic)
                else:
                    loader_rank_ic_values.append(np.nan)
            except ValueError:
                loader_ic_values.append(np.nan)
                loader_rank_ic_values.append(np.nan)
        else:
            loader_ic_values.append(np.nan)
            loader_rank_ic_values.append(np.nan)

    loader_ic_array = np.array(loader_ic_values)
    loader_rank_ic_array = np.array(loader_rank_ic_values)

    def safe_mean(arr):
        arr = arr[~np.isnan(arr)]
        return np.mean(arr) if len(arr) > 0 else np.nan

    performance['loader IC'] = safe_mean(loader_ic_array)
    performance['loader Rank IC'] = safe_mean(loader_rank_ic_array)

    valid_mask = (mask > 0.5) & ~np.isnan(ground_truth) & ~np.isnan(prediction)
    valid_pred = torch.tensor(prediction[valid_mask], dtype=torch.float32)
    valid_truth = torch.tensor(ground_truth[valid_mask], dtype=torch.float32)

    return performance