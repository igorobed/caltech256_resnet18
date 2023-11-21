from os import listdir
import logging
import torch
from utils import make_runs_subdir


def train():
    # задаем параметры общего логгера
    # будет еще один логгер, который в поддирректорию будет записывать результаты обучения
    # на каждой эпохе
    logging.basicConfig(
        level=logging.INFO,
        filename="file_logs.log",
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s"
        )
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

    # 2.Установим device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device type - {device}")


if __name__ == "__main__":
    train()