from os import listdir
import torch
from utils import make_runs_subdir
from dataset import CaltechDataset, get_caltech_dataframe
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from loggers import logging
from config import (
    device,
    image_size,
    batch_size,
    transforms_train,
    transforms_val,
    model,
    num_epochs,
    criterion,
    optimizer,
    scheduler
)


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

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), path_save_best_model)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')


def train():
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
    
    train_model(
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
