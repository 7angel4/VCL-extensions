from constants import *
from alg.coresets import *
from alg.prior import *
from utils.dataloaders import *

class ExperimentConfig:
    def __init__(self, 
                 dataset_type='split',
                 task_type='classification',
                 prior_type='gaussian',
                 init_prior_mu=0.0,
                 init_prior_scale=0.1,
                 init_const=-3.0,
                 coreset_alg_name="random",
                 coreset_size=200,
                 num_epochs=100,
                 batch_size=256,
                 learning_rate=0.001,
                 early_stop_threshold=1e-4,
                 patience=5
                ):
        # Model hyperparameters
        self.prior_type = prior_type  # 'gaussian' or 'exponential'
        self.task_type = task_type
        self.dataset_type = dataset_type
        self.init_prior_mu = init_prior_mu
        self.init_prior_scale = init_prior_scale
        self.init_const = init_const
        self.input_dim = MNIST_INPUT_DIM
        self.hidden_dim = 256
        self.num_samples = 10
        self.split_tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        self.update_prior = True
        
        # Training parameters
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.coreset_alg_name = coreset_alg_name if coreset_alg_name else None
        self.coreset_size = int(coreset_size) if self.coreset_alg_name else 0
        self.patience = int(patience)
        self.early_stop_threshold = float(early_stop_threshold)
        
        
    def validate(self):
        assert self.prior_type in ['gaussian', 'exponential']
        assert self.dataset_type in ['split', 'permuted']
        assert len(self.tasks) > 0

    @property
    def coreset_alg(self):
        """Get coreset algorithm as a function based on the config"""
        if self.coreset_alg_name == "kcenter":
            return KCenterCoresetAlg(self.coreset_size)
        elif self.coreset_alg_name == "random":
            return RandomCoresetAlg(self.coreset_size)
        else:
            return None

    @property
    def tasks(self):
        return range(NUM_MNIST_CLASSES) if self.dataset_type == 'permuted' else self.split_tasks

    @property
    def dataloaders(self):
        dataloaders = PermutedDataLoader(self.tasks, batch_size=self.batch_size) if self.dataset_type == 'permuted' else \
                        SplitDataLoader(self.tasks, batch_size=self.batch_size)
        return dataloaders.run()

    @property
    def output_dims(self):
        # Multihead NN for SplitMNIST
        if self.dataset_type == 'split' and self.task_type == 'classification':
            return [len(t) for t in self.tasks]
        elif self.dataset_type == 'split' and self.task_type == 'regression':
            return [NUM_MNIST_CLASSES for _ in self.tasks]  # one-hot-encoding of classes will have size NUM_MNIST_CLASSES
        else:
            return [NUM_MNIST_CLASSES] # single head for PermutedMNIST

    @property
    def output_dim(self):
        return self.output_dims[0]

    @property
    def eval_metric(self):
        return 'Accuracy' if self.task_type == 'classification' else 'RMSE'

    @property
    def prior(self):
        return ExponentialPrior() if self.prior_type == 'exponential' else GaussianPrior()

    @property
    def name(self):
        p = 'ExpVCL' if self.prior_type == 'exponential' else 'GaussianVCL'
        if self.coreset_alg_name is not None:
            return f"{p} ({self.coreset_alg_name.capitalize()}, {self.coreset_size})"
        else:
            return f"{p} (None)"

    def prepare_targets(self, targets, task_id):
        task = self.tasks[task_id]
        if self.dataset_type == 'split' and self.task_type == 'classification':
            # always treat SplitMNIST tasks as binary classification with class 0/1
            # so we map the first digit to class 0, second digit to class 1
            task_mapping = {task[0]: 0, task[1]: 1}
            targets = torch.tensor([task_mapping[t.item()] for t in targets], device=DEVICE)
        elif self.task_type == 'regression':
            # one hot encoding of class labels
            targets = F.one_hot(targets, num_classes=self.output_dim).float()  # [batch_size, 10]
        return targets

    def loss_fn(self, outputs, targets, task_id):
        # targets are not yet processed
        final_targets = self.prepare_targets(targets, task_id)
        if self.task_type == 'regression': # Gaussian likelihood - MSE loss
            loss = F.mse_loss(outputs.mean(-1), final_targets)
        else: # Categorical likelihood - cross entropy loss (= log softmax then NLL loss)
            log_output = torch.logsumexp(outputs, dim=-1) - np.log(self.num_samples)
            loss = F.nll_loss(log_output, final_targets)
        return loss

    def evaluate(self, outputs, targets, task_id):
        targets = self.prepare_targets(targets, task_id)
        # Calculate metric based on task type
        if self.task_type == 'regression':
            pred = outputs.mean(-1)
            rmse = torch.sqrt(F.mse_loss(pred, targets))
            return rmse.item()
        else:
            log_output = torch.logsumexp(outputs, dim=-1) - np.log(self.num_samples)
            acc = (log_output.argmax(-1) == targets).float().mean()
            return acc.item()