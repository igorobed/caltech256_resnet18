import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from utils import make_runs_subdir
import cv2
from dataset import get_caltech_dataframe
import torchvision.models as models
import time
from tqdm import tqdm
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import lr_scheduler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scheduler_name = "ReduceLROnPlateau"

# модель
class AddProjection(nn.Module):
    def __init__(self, embedding_size=128, model=None, mlp_dim=512):
        super(AddProjection, self).__init__()
        embedding_size = embedding_size
        self.backbone = models.resnet18(pretrained=False, num_classes=256)
        mlp_dim = self.backbone.fc.in_features
        print('Dim MLP input:', mlp_dim)
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
    

class CaltechDatasetSSLFinetune(Dataset):
    def __init__(self, df):  # , transforms=None):
        self.df = df
        # как тут быть с ресайзом???
        
        self.transforms = T.Compose([
            T.ToTensor(),
            # T.Resize(256),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.idx2label = {idx: self.df[self.df.label_idx == idx].label.values[0] for idx in set(self.df.label_idx.values)}
        self.label2idx = {label: idx for idx, label in self.idx2label.items()}
  
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx].path
        label_idx = self.df.iloc[idx].label_idx

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (256, 256))
        
        aug = self.transforms(img)
        
        return aug, int(label_idx)
        

    def __len__(self):
        return len(self.df)
    

def train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=25,
        save_path="./"
        ):
    since = time.time()
    path_save_best_model = save_path + "best.pt"
    best_acc = 0.0

    val_loss_lst = []
    train_loss_lst = []
    val_acc_lst = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train' and scheduler is not None:
                if scheduler_name == "ReduceLROnPlateau":
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

            if phase == "train":
                train_loss_lst.append(epoch_loss)
            elif phase == "val":
                val_loss_lst.append(epoch_loss)
                val_acc_lst.append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), path_save_best_model)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    return {
        "train_lst": train_loss_lst,
        "val_lst": val_loss_lst,
        "val_acc_lst": val_acc_lst
    }


def train():
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # создадим папку для результатов
    try:
        results_dir = make_runs_subdir(mode="ssl_finetune")
    except Exception:
        print("Error when creating a subdirectory")
        return
    
    df_train = get_caltech_dataframe("./data/Caltech256/train_lst.txt")
    df_val = get_caltech_dataframe("./data/Caltech256/val_lst.txt")

    train_dataset = CaltechDatasetSSLFinetune(df_train)
    val_dataset = CaltechDatasetSSLFinetune(df_val)
    image_datasets = {
        "train": train_dataset,
        "val": val_dataset,
    }

    dataloaders = {
        data_type: DataLoader(
            image_datasets[data_type],
            batch_size=32,
            shuffle=True
        )
        for data_type in ["train", "val"]
    }

    label_smoothing = 0.1

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # model = AddProjection()
    # model.load_state_dict(torch.load("./runs/run_ssl_0/weights/best.pt"))
    # model = model.backbone
    model = models.resnet18(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(512, 256)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    

    num_epochs = 5

    dataset_sizes = {
        data_type: len(image_datasets[data_type]) for data_type in ["train", "val"]
    }
    
    res_dict = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs,
        save_path=f"./runs/{results_dir}/weights/"
        )
    

if __name__ == "__main__":
    train()