from os import listdir
import logging
from utils import make_runs_subdir


def train():
    # 0. для текущего эксперимента создать подпапку в папке runs
    make_runs_subdir(mode="val")
    # 1. Проверить, что в папке data лежит датасет и два txt файла train и val

    #try:
    #    data_dir_files = listdir("./data/Caltech256/")
    #    if "train.txt" not in data_dir_files or "val.txt" not in data_dir_files:
    #        raise FileNotFoundError
    #except FileNotFoundError:
        # здесь нужно логгировать ошибку
    #    pass

    
    
    # здесь нужно логгировать успешную проверку, что датасет лежит внутри папки
    # 2. 


if __name__ == "__main__":
    train()