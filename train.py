from os import listdir
import torch
import torch.nn as nn
from utils import make_runs_subdir, save_acc_img, save_loss_img
from dataset import CaltechDataset, get_caltech_dataframe
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
from loggers import logging
from config import (
    device,
    transforms_train,
    transforms_val,
    model,
    num_epochs,
    loss,
    optimizer,
    scheduler,
    model_name
)
from sklearn.utils.class_weight import compute_class_weight


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

            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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

    # для воспроизводимости
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # задаем параметры общего логгера
    # будет еще один логгер, который в поддирректорию будет записывать результаты обучения
    # на каждой эпохе
    # ... перенес в loggers.py
    logging.info("Train start")
    
    # 0. для текущего эксперимента создать подпапку в папке runs
    try:
        results_dir = make_runs_subdir(mode="train")
    except Exception:
        logging.error("Error when creating a subdirectory")
        return
    # логгирование успеха или неудачии создания дирректории для результатов обучения
    logging.info(f"Created subdirrectory for results - {results_dir}")
    
    # 1. Проверить, что в папке data лежит датасет и два файла: train_lst.txt и val_lst.txt
    try:
        data_dir_files = listdir("./data/Caltech256/")
        if "train_lst.txt" not in data_dir_files or "val_lst.txt" not in data_dir_files:
            raise FileNotFoundError
    except FileNotFoundError:
        # здесь нужно логгировать ошибку
        logging.error("Incorrect dataset - FileNotFoundError")
        return
    logging.info("Caltech256 dataset exists in folder data")

    logging.info(f"Device type - {device}")

    # 3. Определим обучающий и валидационный датафреймы
    df_train = get_caltech_dataframe("./data/Caltech256/train_lst.txt")
    df_val = get_caltech_dataframe("./data/Caltech256/val_lst.txt")

    # 4. Определим обучающий и валидационный датасеты
    train_dataset = CaltechDataset(df_train, transforms_train)
    val_dataset = CaltechDataset(df_val, transforms_val)
    image_datasets = {
        "train": train_dataset,
        "val": val_dataset,
    }

    dataset_sizes = {
        data_type: len(image_datasets[data_type]) for data_type in ["train", "val"]
    }

    # 5. Определим даталоадеры
    dataloaders = {
        data_type: DataLoader(
            image_datasets[data_type],
            batch_size=32,
            shuffle=True
        )
        for data_type in ["train", "val"]
    }

    print("All good!")

    if loss == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif loss == "crossentropybalance":
        # получим веса для каждого класса
        all_train_lbls = []
        for item, lbl in train_dataset:
            all_train_lbls.append(lbl)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.array(all_train_lbls)), y=np.array(all_train_lbls))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    elif loss == "kldiv":
        criterion = nn.KLDivLoss()

    
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
    
    save_loss_img(
        res_dict["train_lst"],
        res_dict["val_lst"],
        save_path=f"./runs/{results_dir}/loss.png"
        )
    
    save_acc_img(
        res_dict["val_acc_lst"],
        save_path=f"./runs/{results_dir}/acc.png"
        )
    
    # ЭКСПОРТ В ONNX ФОРМАТ


if __name__ == "__main__":
    train()
