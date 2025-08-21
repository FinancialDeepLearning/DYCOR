import matplotlib.pyplot as plt
import os

def plot_training_metrics(val_ic_history, test_ic_history,
                          val_rank_ic_history, test_rank_ic_history,
                          market_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(val_ic_history) + 1)
    
    plot_metric(epochs, val_ic_history, test_ic_history, 
                'IC', 'IC Value', 
                os.path.join(save_dir, 'ic_progress.png'))
    
    plot_metric(epochs, val_rank_ic_history, test_rank_ic_history, 
                'Rank IC', 'Rank IC Value', 
                os.path.join(save_dir, 'rank_ic_progress.png'))
    
    print(f"Plots saved to {save_dir}")

def plot_metric(epochs, val_metric, test_metric, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_metric, label=f'Validation {title}', marker='o')
    plt.plot(epochs, test_metric, label=f'Test {title}', marker='s')
    plt.title(f'{title} Progress')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)

    max_epoch = max(epochs)
    plt.xticks([e for e in epochs if e % 5 == 0 or e == max_epoch])
    
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()