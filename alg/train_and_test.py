from constants import *
from alg.models import *
from config import *

def train(model, dataloader, task_id):
    """Train model on a specific task"""
    config = model.config
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    
    prev_loss = float('inf')
    num_consec_worse_epochs = 0
    task_id = model.task_id(task_id)
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            # Monte Carlo sampling
            outputs = torch.zeros(inputs.size(0), config.output_dim,   # len(task)
                                  config.num_samples, device=DEVICE)
            for i in range(config.num_samples):
                net_out = model(inputs, task_id)
                outputs[..., i] = (net_out if config.task_type == 'regression' else F.log_softmax(net_out, dim=-1))

            loss = model.compute_loss(outputs, targets, task_id)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Early stopping
        if epoch_loss + config.early_stop_threshold > prev_loss:
            prev_loss = epoch_loss
            num_consec_worse_epochs = 0
        else:
            num_consec_worse_epochs += 1
            if num_consec_worse_epochs >= config.patience:
                break


def test(model, dataloader, task_id, ret_std=False):
    """Test model supporting both classification and regression"""
    model.eval()
    metrics = []
    config = model.config  # for brevity
    task_id = model.task_id(task_id)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = torch.zeros(inputs.size(0), config.output_dim,
                                config.num_samples, device=DEVICE)
            for i in range(config.num_samples):
                net_out = model(inputs, task_id)
                outputs[..., i] = (net_out if config.task_type == 'regression' else F.log_softmax(net_out, dim=-1))
            
            # Calculate metric based on task type
            metrics.append(config.evaluate(outputs, targets, task_id))
    
    return np.mean(metrics) if not ret_std else (np.mean(metrics), np.std(metrics))