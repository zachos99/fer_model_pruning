import os
import sys

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import prune
import torch.nn as nn


from networks.dan import DAN

"""
    This script is used for model pruning
"""


def get_model():
    """
    Loads the model from the path
    """

    model = DAN(num_head=4, num_class=8, pretrained=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model.load_state_dict(torch.load("    "))  works too
    checkpoint = torch.load('./checkpoints/affecnet8_epoch5_acc0.6209.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    model.cuda()
    model.eval()
    return model


def prune_model(model, layer_type, amount):
    """
    Prunes the model.
    """
    # prune at random (proportion)% of the connections in the parameter named weight in the passed module
    for name, module in model.features.named_modules():
        if isinstance(module, layer_type):
            prune.random_unstructured(module, 'weight', amount)
            prune.remove(module, 'weight')


    # Another way of pruning is the L1 unstructed
    # We prune the 3 smallest entries in the bias (or weight) by L1 norm, as implemented in the l1_unstructured pruning function

    # prune.l1_unstructured(module, name="bias", amount=3)

    return model





model= get_model()

# Choose percentage of pruning
amount = 0.75

# First check what modules the model contains: nn.Conv2d, nn.BatchNorm2d, nn.Linear etc.
# Pass the module type along with the amount (percentage)
model = prune_model(model, nn.Conv2d, amount)

"""
    Save the pruned model
"""
path = "./checkpoints/affecnet8_epoch5_acc0.6209_pruned_"+str(amount)+".pth"
torch.save(model.state_dict(), path)







