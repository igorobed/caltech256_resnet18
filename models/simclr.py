import torch
import torch.nn as nn
from torchvision import models


class AddProjection(nn.Module):
    def __init__(self, embedding_size=128, mlp_dim=512):
        super(AddProjection, self).__init__()
        embedding_size = embedding_size
        self.backbone = models.resnet18(pretrained=False, num_classes=256)
        mlp_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # add mlp projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)


def get_simclr_model(weights: str) -> nn.Module:
    model = AddProjection()
    if weights is not None:
        model.load_state_dict(torch.load("./runs/run_ssl_0/weights/best.pt"))
        model = model.backbone
        model.fc = nn.Linear(512, 256)
        return model
    return model