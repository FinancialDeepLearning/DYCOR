import os
import time
import numpy as np
import torch
import torch.nn as nn
import pickle
from utils.loss import get_loss
from utils.evaluator import evaluate
from visualization import plot_training_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(model, optimizer, data_loader, cur_offset, config):
    data_batch, mask_batch, price_batch, gt_batch = map(
        lambda x: torch.Tensor(x).to(device),
        data_loader.get_batch(cur_offset)
    )
    
    model.train()
    prediction, clustering_info = model(data_batch)

    if cur_offset % 100 == 0:
        market_reps = clustering_info['market_reps'].detach()
        soft_cluster_weights = clustering_info['soft_cluster_weights'].detach()
        
        print(f"Number of market segments: {market_reps.size(0)}")
        
        effective_cluster_size = (soft_cluster_weights > 0.2).float().sum(dim=0).detach().cpu()
        print(f"Effective cluster sizes (weights > 0.2): {list(map(int, effective_cluster_size))}")
        print()
    
    reg_loss_weight = 0.35
    rank_loss_weight = 0.65
    ic_weight = 0.65
    
    loss, _ = get_loss(
        prediction, gt_batch, price_batch, mask_batch, 
        config['stock_num'], reg_loss_weight, rank_loss_weight, ic_weight
    )

    return_ratio = torch.div(torch.sub(prediction, price_batch), price_batch)

    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.data = param.grad.data.detach()
    
    optimizer.step()
    
    del prediction, clustering_info
    del return_ratio
    
    if cur_offset % 10 == 0:
        torch.cuda.empty_cache()
    
    return loss.item()

def train_epoch(model, optimizer, data_loader, start_index, end_index, config):
    total_loss = 0
    num_steps = 0
    
    for cur_offset in range(start_index, end_index - config['lookback_length'] - config['steps'] + 1):
        if cur_offset > start_index and (cur_offset - start_index) % 100 == 0:
            torch.cuda.empty_cache()
        
        loss = train_step(model, optimizer, data_loader, cur_offset, config)
        
        total_loss += loss
        num_steps += 1
        
        if num_steps % 100 == 0:
            print(f"Step {num_steps}: Average Loss = {total_loss/num_steps:.4f}")
    
    torch.cuda.empty_cache()
    
    return total_loss / num_steps

def validate(model, data_loader, start_index, end_index, config):
    model.eval()
    
    total_loss = 0
    num_steps = 0
    
    predictions = []
    ground_truths = []
    masks = []
    prices = []
    
    with torch.no_grad():
        for cur_offset in range(start_index - config['lookback_length'], end_index - config['lookback_length'] - config['steps'] + 1):
            if num_steps > 0 and num_steps % 100 == 0:
                torch.cuda.empty_cache()
            
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                data_loader.get_batch(cur_offset)
            )
            
            prediction, clustering_info = model(data_batch)
            
            reg_loss_weight = 0.35
            rank_loss_weight = 0.65
            ic_weight = 0.65
            
            loss, curr_rr = get_loss(
                prediction, gt_batch, price_batch, mask_batch, 
                config['stock_num'], reg_loss_weight, rank_loss_weight, ic_weight
            )
            
            total_loss += loss.item()
            num_steps += 1
            
            if cur_offset + config['lookback_length'] >= start_index:
                predictions.append(curr_rr.cpu().numpy())
                ground_truths.append(gt_batch.cpu().numpy())
                masks.append(mask_batch.cpu().numpy())
                prices.append(price_batch.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=1)
    ground_truths = np.concatenate(ground_truths, axis=1)
    masks = np.concatenate(masks, axis=1)
    prices = np.concatenate(prices, axis=1)
    
    perf = evaluate(predictions, ground_truths, masks)
    
    torch.cuda.empty_cache()
    
    return total_loss / num_steps, perf, predictions, ground_truths, masks, prices

def train_model(model, data_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    train_start, valid_start, test_start, end_index = data_loader.get_train_valid_test_indices()
    
    val_ic_history = []
    val_rank_ic_history = []
    test_ic_history = []
    test_rank_ic_history = []
    
    best_val_ic = float('-inf')
    patience_counter = 0
    best_test_data = None
    best_test_perf = None
    best_model_state = None
    
    total_start_time = time.time()
    
    for epoch in range(config['epochs']):
        print(f"\n################ Epoch {epoch + 1} ################")
        
        train_start_time = time.time()
        tra_loss = train_epoch(model, optimizer, data_loader, train_start, valid_start, config)
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        
        torch.cuda.empty_cache()
        
        val_start_time = time.time()
        val_loss, val_perf, _, _, _, _ = validate(model, data_loader, valid_start, test_start, config)
        val_end_time = time.time()
        val_time = val_end_time - val_start_time
        
        torch.cuda.empty_cache()
        
        test_start_time = time.time()
        _, test_perf, test_pred, test_gt, test_mask, test_price = validate(model, data_loader, test_start, end_index, config)
        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        
        val_ic = val_perf["loader IC"]
        val_rank_ic = val_perf["loader Rank IC"]
        test_ic = test_perf["loader IC"]
        test_rank_ic = test_perf["loader Rank IC"]
        
        val_ic_history.append(val_ic)
        val_rank_ic_history.append(val_rank_ic)
        test_ic_history.append(test_ic)
        test_rank_ic_history.append(test_rank_ic)
        
        print('\nValid performance:')
        print(f'IC: {val_ic:.4e}, Rank IC: {val_rank_ic:.4e}\n')
        print('Test performance:')
        print(f'IC: {test_ic:.4e}, Rank IC: {test_rank_ic:.4e}\n')
        
        print(f"Train time (epoch {epoch + 1}): {train_time:.2f} seconds")
        print(f"Validation time (epoch {epoch + 1}): {val_time:.2f} seconds")
        print(f"Test time (epoch {epoch + 1}): {test_time:.2f} seconds\n")
        
        if epoch + 1 > config['warmup_epochs']:
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                patience_counter = 0
                best_model_state = model.state_dict()
                best_test_data = (test_pred, test_gt, test_mask, test_price)
                best_test_perf = test_perf
                
                save_path = os.path.join(config['results_dir'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_ic': val_ic,
                    'test_perf': test_perf
                }, save_path)
                
                save_predictions(config['results_dir'], test_pred, test_gt, test_mask, test_price)
                
                print(f"New best validation IC: {val_ic:.4e}")
                print(f"Model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{config['patience']}")
                print(f"Best validation IC so far: {best_val_ic:.4e}")
            
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best validation IC achieved: {best_val_ic:.4e}")
                
                total_end_time = time.time()
                print(f"Total training time until early stopping: {total_end_time - total_start_time:.2f} seconds")
                
                model.load_state_dict(best_model_state)
                
                break
        
        plot_training_metrics(
            val_ic_history, test_ic_history,
            val_rank_ic_history, test_rank_ic_history,
            config['market_name'], config['results_dir']
        )
    else:
        total_end_time = time.time()
        print(f"Training completed without early stopping")
        print(f"Total training time: {total_end_time - total_start_time:.2f} seconds")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    
    metrics = {
        'val_ic_history': val_ic_history,
        'test_ic_history': test_ic_history,
        'val_rank_ic_history': val_rank_ic_history,
        'test_rank_ic_history': test_rank_ic_history,
        'best_val_ic': best_val_ic,
        'best_test_perf': best_test_perf,
    }
    
    return model, best_test_data, metrics

def save_predictions(save_dir, test_pred, test_gt, test_mask, test_price):
    data_dir = os.path.join(save_dir, 'result_data')
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, 'prediction.npy'), test_pred)
    np.save(os.path.join(data_dir, 'gt.npy'), test_gt)
    np.save(os.path.join(data_dir, 'mask.npy'), test_mask)
    np.save(os.path.join(data_dir, 'price.npy'), test_price)
    
    print(f"Prediction results saved to {data_dir}")
