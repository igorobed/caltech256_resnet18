# https://theaisummer.com/simclr/
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




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
           'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
           'weight_decay': weight_decay,
           'layer_adaptation': True,
        },
        {
           'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
           'weight_decay': 0.,
           'layer_adaptation': False,
        },
    ]
    return param_groups


def device_as(t1, t2):
    return t1.to(t2.device)


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.07):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    
    def forward(self, proj_1, proj_2):
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss
    

class Augment:
    def __init__(self, img_size, s=1):
        color_jitter = T.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = T.Compose([
            T.ToTensor(),
            T.Resize(img_size),
            T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.2),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
    

class CaltechDatasetSSL(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        # как тут быть с ресайзом???
        self.aug = Augment(256)
        self.transforms = transforms
        self.idx2label = {idx: self.df[self.df.label_idx == idx].label.values[0] for idx in set(self.df.label_idx.values)}
        self.label2idx = {label: idx for idx, label in self.idx2label.items()}
  
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx].path
        label_idx = self.df.iloc[idx].label_idx

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            aug = self.transforms(image=img)
            img = aug["image"]
            return img, int(label_idx)
        else:
            img_1, img_2 = self.aug(img)
            return (img_1, img_2), int(label_idx)

    def __len__(self):
        return len(self.df)
    

def train():
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # создадим папку для результатов
    try:
        results_dir = make_runs_subdir(mode="ssl")
    except Exception:
        print("Error when creating a subdirectory")
        return
    
    df_train = get_caltech_dataframe("./data/Caltech256/train_lst.txt")
    # df_val = get_caltech_dataframe("./data/Caltech256/val_lst.txt")

    

    dataset_sizes = 6400

    
    criterion = ContrastiveLoss(batch_size=32)

    save_path = f"./runs/{results_dir}/weights/"

    num_epochs = 50

    model = AddProjection()
    model.to(device)

    train_loss_lst = []

    max_epochs = int(num_epochs)
    lr = 0.0003
    weight_decay = 1e-4
    param_groups = define_param_groups(model, weight_decay, 'adam')
    
    optimizer = optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(6400 / 256), eta_min=1e-8,
                                                           last_epoch=-1)
    
    since = time.time()
    path_save_best_model = save_path + "best.pt"
    best_loss = np.inf

    accum_iter = 8

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        train_dataset = CaltechDatasetSSL(df_train.sample(6400))
        image_datasets = {
        "train": train_dataset,
        # "val": val_dataset,
        }
        dataloaders = {
        data_type: DataLoader(
            image_datasets[data_type],
            batch_size=32,
            shuffle=True
        )
        for data_type in ["train"]}

        dataset_sizes = {
        data_type: len(image_datasets[data_type]) for data_type in ["train"]
        }


        for phase in ['train']:  # , 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs_1, inputs_2 = inputs[0].to(device), inputs[1].to(device)
                labels = labels.to(device)

                

                # zero the parameter gradients
                # optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_1 = model(inputs_1)
                    outputs_2 = model(inputs_2)
                    # _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs_1, outputs_2)

                    

                    

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # optimizer.step()
                        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloaders["train"])):
                            optimizer.step()
                            optimizer.zero_grad()

                # statistics

                running_loss += loss.item()#  * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            
            
            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train' and scheduler is not None and epoch >= 5:
                scheduler.step()

            if phase == "train":
                train_loss_lst.append(epoch_loss)
            

            print(f'{phase} Loss: {epoch_loss:.4f}')  #  Acc: {epoch_acc:.4f}')

            # deep copy the model
            if  epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), path_save_best_model)

        print()

# я не понимаю кусок кода с получением изображений из датасета, отправкой их в модель и 
# получением лосса

if __name__ == "__main__":
    train()


# на каждой эпохе формировать новый датасет маленького размера, кратный 64



# ПРЕЖДЕ ЧЕМ ДОЛГО УЧИТЬ, НАДО ВСЕ ЖЕ РЕШИТЬ ВОПРОС С РЕСАЙЗОМ и вармапом
