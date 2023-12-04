import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.dropout.cutout import Cutout
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
batch_size = 64

# трансформации для обучения и валидации
transforms_train = A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(30, p=0.3),
    A.RandomSizedCrop((image_size - 60, image_size), image_size, image_size, p=0.3),
    A.CoarseDropout(max_holes=5, max_height=50, max_width=50, min_holes=3, min_height=30, min_width=30),
    A.Normalize(),
    ToTensorV2()
])

transforms_val = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(),
    ToTensorV2()
])

mixup = True
alpha=0.2

# используемая модель("resnet18", "resnet18_my")
model_name = "resnet18_simclr"
model = get_resnet_model(model_name)
# model.load_state_dict(torch.load("./runs/run_train_9/weights/best.pt"))
model.to(device)

# количество эпох
num_epochs = 100
# лосс
label_smoothing = 0.1
# criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
# loss = "crossentropy"
loss = "crossentropybalance"
# loss = "kldiv"  # для использования этой штуки выход модели нужно обернуть в софтмакс

# оптимизатор
# optimizer = optim.AdamW(model.parameters(), lr=1e-2)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
# optimizer = optim.RAdam(model.parameters(), lr=1e-2)

# шедулер(может быть None)  # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling
# scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# scheduler = None
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 14, 1e-5)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
scheduler_name = "ReduceLROnPlateau"