import os

COMMON_CONFIG = {
    'lookback_length': 16,
    'fea_num': 5,
    'hidden_dim': 40,
    'min_var_ratio': 0.93,
    'temperature': 0.2,
    'dropout_prob': 0.3,
    'learning_rate': 0.0001,
    'steps': 1,
    'epochs': 100,
    'patience': 50,
    'warmup_epochs': 10,
}

DATASET_CONFIGS = {
    'NASDAQ': {
        'stock_num': 1026,
        'n_subclusters': 4,
        'valid_index': 756,
        'test_index': 1008,
        'data_path': '../data/NASDAQ.pkl',
    },
    'NYSE': {
        'stock_num': 1737,
        'n_subclusters': 6,
        'valid_index': 756,
        'test_index': 1008,
        'data_path': '../data/NYSE.pkl',
    },
    'SP500': {
        'stock_num': 646,
        'n_subclusters': 4,
        'valid_index': 3775,
        'test_index': 4278,
        'data_path': '../data/SP500.pkl',
    }
}

def get_config(market_name): 
    config = {**COMMON_CONFIG, **DATASET_CONFIGS[market_name]}
    config['results_dir'] = f"../results/{market_name}"
    os.makedirs(config['results_dir'], exist_ok=True)
    
    return config