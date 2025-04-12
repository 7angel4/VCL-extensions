from constants import *
from utils.dataloaders import *

class CoresetAlg:
    def __init__(self, coreset_size=200):
        self.coresets = []
        self.coreset_size = coreset_size

    def add_coreset(self, dataloader):
        pass


class RandomCoresetAlg(CoresetAlg):
    def __init__(self, coreset_size=200):
        super().__init__(coreset_size)

    def add_coreset(self, dataloader):
        task_indices = dataloader.sampler.indices
        shuffled_indices = task_indices[torch.randperm(len(task_indices))]
        
        # Split into coreset and remaining data
        core_indices, remaining_indices = shuffled_indices[:self.coreset_size], shuffled_indices[self.coreset_size:]
        dataloader.sampler.indices = remaining_indices
        
        # Create coreset loader
        coreset_loader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=SubsetRandomSampler(core_indices)
        )
        self.coresets.append(coreset_loader)


class KCenterCoresetAlg(CoresetAlg):
    def __init__(self, coreset_size=200):
        super().__init__(coreset_size)

    def add_coreset(self, dataloader):
        """ Adds a coreset selected by greedy k-center algorithm to the existing set of coresets. """
        X_train = self._extract_trainset(dataloader)
        
        # Initialize distances and first point in the coreset
        dists = np.full(X_train.shape[0], np.inf)
        current_index = 0
        dists = self._update_distances(dists, X_train, current_index)
        selected_indices = [current_index]
    
        # Select k-center points
        for _ in range(1, self.coreset_size):
            current_index = np.argmax(dists)
            dists = self._update_distances(dists, X_train, current_index)
            selected_indices.append(current_index)

        task_indices = dataloader.sampler.indices
        core_indices = task_indices[selected_indices]  # selected index in X_train = index into task_indices
        remaining_indices = np.setdiff1d(task_indices, core_indices)
        coreset_loader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=SubsetRandomSampler(core_indices)
        )
    
        # Update original dataloader's sampler to exclude coreset points
        dataloader.sampler.indices = remaining_indices
        self.coresets.append(coreset_loader)  # store new coreset
    

    def _extract_trainset(self, dataloader):
        """ Extract flattened trainset and original indices """
        task_indices = dataloader.sampler.indices  # only has target classes in this task
        X_train = []
        for ind in task_indices:
            img, _ = dataloader.dataset[ind]
            X_train.append(img.view(-1).numpy())        
        return np.stack(X_train) 
    
    def _update_distances(self, dists, X, current_id):
        """Update distances to current center"""
        new_dists = np.linalg.norm(X - X[current_id], axis=1)  # row diff
        return np.minimum(dists, new_dists)