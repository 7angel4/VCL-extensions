from constants import *
from alg.prior import *

class VariationalLayer(nn.Module):
    """Variational continual learning layer with configurable prior"""
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Weight parameters
        self.W_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.W_rho = nn.Parameter(torch.Tensor(output_dim, input_dim))
        
        # Bias parameters
        self.b_mu = nn.Parameter(torch.Tensor(output_dim))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim))
        
        # Initialize priors (either gaussian or exponential)
        self.W_prior = {
            'mu': torch.tensor(config.init_prior_mu),
            'scale': torch.tensor(config.init_prior_scale)
        }
        self.b_prior = {
            'mu': torch.tensor(config.init_prior_mu),
            'scale': torch.tensor(config.init_prior_scale)
        }

        self.prior = config.prior
        self.prior.init_params_for(self, init_mu=config.init_prior_mu, 
                                   init_scale=config.init_prior_scale, 
                                   init_const=config.init_const)
        
    @property
    def W_sigma(self):
        """Convert rho to sigma using softplus"""
        return torch.log1p(torch.exp(self.W_rho))
    
    @property
    def b_sigma(self):
        """Convert rho to sigma using softplus"""
        return torch.log1p(torch.exp(self.b_rho))
    
    def forward(self, x, sample=True):
        act_mu = F.linear(x, self.W_mu, self.b_mu)  # just linear transformation
        if self.training or sample:
            act_var = F.linear(x**2, self.W_sigma**2, self.b_sigma**2)
            act_std = torch.sqrt(act_var + EPS_OFFSET)
            noise = torch.randn_like(act_mu)
            return act_mu + act_std * noise  # local reparameterization trick
        return act_mu
    
    def kl_loss(self):
        """Compute KL divergence based on prior type"""
        W_params = {'mu': self.W_mu, 'scale': self.W_sigma}
        b_params = {'mu': self.b_mu, 'scale': self.b_sigma}
        return (torch.sum(self.prior.kl(W_params, self.W_prior))) + \
               (torch.sum(self.prior.kl(b_params, self.b_prior)))

    def update_prior(self):
        self.W_prior = {
            'mu': self.W_mu.detach().clone(),
            'scale': self.W_sigma.detach().clone()
        }
        self.b_prior = {
            'mu': self.b_mu.detach().clone(),
            'scale': self.b_sigma.detach().clone()
        }
        self.prior = GaussianPrior()


class BaseNN(nn.Module):
    """Base model for SplitMNIST experiments"""
    def __init__(self, config):
        super().__init__()
        self.config = config

    def task_id(self, t):
        return 0 if len(self.task_heads) == 1 else t
        
    def forward(self, x, task_id):
        x = x.view(-1, self.config.input_dim)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # single-headed for PermutedMNIST
        head_id = self.task_id(task_id)
        return self.task_heads[head_id](x)

    def compute_loss(self, outputs, targets, task_id):
        return self.config.loss_fn(outputs, targets, task_id)
    

class VanillaNN(BaseNN):
    """Standard neural network without VCL"""
    def __init__(self, config):
        super().__init__(config)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        ])
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, d) for d in self.config.output_dims
        ])


class VCLNN(BaseNN):
    """Variational Continual Learning model"""
    def __init__(self, config):
        super().__init__(config)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            VariationalLayer(config.input_dim, config.hidden_dim, config),
            VariationalLayer(config.hidden_dim, config.hidden_dim, config)
        ])
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            VariationalLayer(config.hidden_dim, d, config) for d in self.config.output_dims
        ])
    
    def update_priors(self):
        """Update priors to current posteriors after learning a task"""
        for layer in self.hidden_layers + list(self.task_heads):
            layer.update_prior()
    
    def _kl_loss(self, task_id):
        """Compute KL loss for current task"""
        kl_loss = 0.0
        for layer in self.hidden_layers:
            kl_loss += layer.kl_loss()
        kl_loss += self.task_heads[task_id].kl_loss()
        return kl_loss

    # overrides BaseNN's compute_loss
    def compute_loss(self, outputs, targets, task_id):
        head_id = self.task_id(task_id)
        loss = super(VCLNN, self).compute_loss(outputs, targets, head_id)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        loss += self._kl_loss(head_id) / num_params
        return loss