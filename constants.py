import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import os

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MNIST_INPUT_DIM = 784
MNIST_IMG_SIZE = int(np.sqrt(MNIST_INPUT_DIM))
EPS_OFFSET = 1e-16  # ensure nonzero sqrt
NUM_MNIST_CLASSES = 10
MNIST_CLASSES = list(range(NUM_MNIST_CLASSES))
VANILLA_MODEL = 'Vanilla'
RESULTS_DIR = os.getcwd() + '/results'
DATA_ROOT = './data'