import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from models.resnet import get_resnet_model

# девайс
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# размер изображения
image_size = 256

# размер батча
batch_size = 32

# трансформации для обучения и валидации
transforms_train = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.Rotate(30, p=0.3),
    A.Resize(image_size, image_size),
    A.Normalize(),
    ToTensorV2()
])

transforms_val = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(),
    ToTensorV2()
])

# используемая модель("resnet18", "resnet18_my")
model = get_resnet_model("resnet18")
model.to(device)

# количество эпох
num_epochs = 3

# лосс
criterion = nn.CrossEntropyLoss()

# оптимизатор
optimizer = optim.AdamW(model.parameters())

# шедулер(может быть None)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
