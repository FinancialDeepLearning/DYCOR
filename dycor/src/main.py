import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))
sys.path.insert(0, 'dycor/src')

from config import get_config
from data_loader import StockDataLoader
from train import train_model

from dycor import DYCOR

def set_seed(seed=None):
    if seed is None:
        seed = int(time.time()) % 100000
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed set to: {seed}")
    return seed

def main():
    parser = argparse.ArgumentParser(description='DYCOR: Dynamic Correlation for Stock Trend Prediction')
    parser.add_argument('--market', type=str, default='NASDAQ', choices=['NASDAQ', 'NYSE', 'SP500'],
                        help='Market to train on (default: NASDAQ)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed (default: based on current time)')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device ID (default: 0)')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("GPU not available, using CPU")
    
    seed = set_seed(args.seed)
    
    config = get_config(args.market)
    config['market_name'] = args.market
    config['seed'] = seed
    
    print("\nTraining Configuration:")
    print(f"Market: {args.market}")
    print(f"Number of stocks: {config['stock_num']}")
    print(f"Number of subclusters: {config['n_subclusters']}")
    print(f"Lookback length: {config['lookback_length']}")
    print(f"Temperature: {config['temperature']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Patience: {config['patience']}")
    print(f"Warmup epochs: {config['warmup_epochs']}")
    
    data_loader = StockDataLoader(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DYCOR(
        stock_num=config['stock_num'],
        lookback_length=config['lookback_length'],
        fea_num=config['fea_num'],
        hidden_dim=config['hidden_dim'],
        min_var_ratio=config['min_var_ratio'],
        n_subclusters=config['n_subclusters'],
        temperature=config['temperature'],
        dropout=config['dropout_prob'],
    ).to(device)
    
    _, _, metrics = train_model(model, data_loader, config)
    
    print("\nTraining complete.")
    print(f"Best validation IC: {metrics['best_val_ic']:.4e}")
    print(f"Best test IC: {metrics['best_test_perf']['loader IC']:.4e}")
    print(f"Best test Rank IC: {metrics['best_test_perf']['loader Rank IC']:.4e}")
    
    print(f"\nResults saved to {config['results_dir']}")
    
if __name__ == "__main__":
    main()