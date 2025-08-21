import numpy as np
import pickle
import random

class StockDataLoader:
    def __init__(self, config):
        self.config = config
        self.market_name = config.get('market_name', 'Unknown')
        self.lookback_length = config['lookback_length']
        self.steps = config['steps']
        
        self._load_data()
        
        self.valid_index = config['valid_index']
        self.test_index = config['test_index']
        self.trade_dates = self.mask_data.shape[1]
        
    def _load_data(self):
        print(f"Loading {self.market_name} dataset...")
        
        with open(self.config['data_path'], 'rb') as f:
            market_dict = pickle.load(f)
            
        data = np.transpose(market_dict['feature_npy'], (1, 0, 2))
        data = np.where(np.isnan(data), 1.1, data)
        
        self.price_data = data[:, :, 0]
        self.mask_data = np.where(self.price_data==1.1, 0, 1)
        
        self.eod_data = np.where(data==1.1, 0, data)
        
        self.gt_data = np.zeros((data.shape[0], data.shape[1]))
        
        for ticker in range(0, data.shape[0]):
            for row in range(1, data.shape[1]):
                self.gt_data[ticker][row] = (data[ticker][row][0] - data[ticker][row - self.steps][0]) / data[ticker][row - self.steps][0]
        
        print(f"Data loaded successfully. Shape: {data.shape}")
    
    def get_batch(self, offset=None, is_train=True):
        if offset is None:
            if is_train:
                offset = random.randrange(0, self.valid_index)
            else:
                offset = random.randrange(self.valid_index, self.test_index)
                
        seq_len = self.lookback_length
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        
        return (
            self.eod_data[:, offset:offset + seq_len, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(self.price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(self.gt_data[:, offset + seq_len + self.steps - 1], axis=1)
        )
    
    def get_train_valid_test_indices(self):
        return 0, self.valid_index, self.test_index, self.trade_dates