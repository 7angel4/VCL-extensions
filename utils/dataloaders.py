from constants import *
from torch.utils.data import DataLoader, SubsetRandomSampler

class MNISTDataLoader:
    def __init__(self, tasks, batch_size=256):
        self.tasks = tasks
        self.batch_size = batch_size
        self.dataloaders = None

    def get_class_indices(self, dataset, target_classes):
        """Get indices for specified class targets"""
        idx = torch.zeros_like(dataset.targets, dtype=torch.bool)
        for target in target_classes:
            idx |= (dataset.targets == target)
        return idx

    def _run(self):
        """Creates train/test dataloaders for each task. 
           Changes state of the dataloader. """ 
        pass  # logic to be overriden by children

    def run(self):
        # Only runs the dataloader if it hasn't been run yet
        # to ensure that the same dataset is used for one trial
        if self.dataloaders is None:
            self._run()
        return self.dataloaders
    

class SplitDataLoader(MNISTDataLoader):
    def __init__(self, tasks, batch_size=256):
        super().__init__(tasks, batch_size)
    
    def _run(self):
        transform = transforms.Compose([
            transforms.Resize((MNIST_IMG_SIZE, MNIST_IMG_SIZE)),
            transforms.ToTensor(),
        ])
    
        # Load MNIST datasets
        train_set = torchvision.datasets.MNIST(
            root=DATA_ROOT, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(
            root=DATA_ROOT, train=False, download=True, transform=transform)
    
        dataloaders = []
        
        for classes in self.tasks:
            # Train loader
            train_idx = torch.where(self.get_class_indices(train_set, classes))[0]
            train_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                sampler=SubsetRandomSampler(train_idx)
            )
            
            # Test loader
            test_idx = torch.where(self.get_class_indices(test_set, classes))[0]
            test_loader = DataLoader(
                test_set,
                batch_size=self.batch_size,
                sampler=SubsetRandomSampler(test_idx)
            )
            
            dataloaders.append((train_loader, test_loader))

        self.dataloaders = dataloaders


# The PermutedMNIST data loader is NOT THOROUGHLY TESTED
class PermutedDataLoader(MNISTDataLoader):
    def __init__(self, tasks, batch_size=256):
        super().__init__(tasks, batch_size)

    # subclass representing the permuted version of a dataset
    class PermutedDataset(torch.utils.data.Dataset):
        def __init__(self, original, perm):
            self.original = original
            self.perm = perm
            
        def __len__(self):
            return len(self.original)
            
        def __getitem__(self, idx):
            x, y = self.original[idx]
            return x[self.perm], y  # Permute pixels
    
    def _run(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten
        ])
    
        # Load base MNIST
        train_set = torchvision.datasets.MNIST(
            root=DATA_ROOT, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(
            root=DATA_ROOT, train=False, download=True, transform=transform)
    
        # Generate permutations
        permutations = [np.random.permutation(MNIST_IMG_SIZE*MNIST_IMG_SIZE) for _ in self.tasks]
        
        dataloaders = []
        for perm in permutations:
            # Apply permutation to train/test sets
            permuted_train = PermutedDataset(train_set, perm)
            permuted_test = PermutedDataset(test_set, perm)
            
            train_loader = DataLoader(permuted_train, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(permuted_test, batch_size=self.batch_size)
            dataloaders.append((train_loader, test_loader))
        
        self.dataloaders = dataloaders