import wandb
from main import main
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'values': [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005]},
        'batch_size': {'values': [16, 32, 64, 128]},
        'lstm_hidden_dim': {'values': [128]},
        'conv1_filters': {'values': [64]},
        'conv2_filters': {'values': [128]},
        'epochs': {'value': 100}
    }
}

sweep_id = wandb.sweep(sweep_config, project="ultrasound-cnn-bilstm-0808")
wandb.agent(sweep_id, function=main, count=24)
