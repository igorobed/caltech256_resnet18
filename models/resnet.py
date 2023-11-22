import torch.nn as nn
from torchvision import models


def get_resnet_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 256)
        
        return model
    
    return None