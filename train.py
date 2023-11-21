from os import listdir
import logging


def train():
    # 1. Проверить, что в папке data лежит датасет и два txt файла train и val
    try:
        listdir("./data/Caltech256/")
    except FileNotFoundError:
        # здесь нужно логгировать ошибку
    # здесь нужно логгировать успешную проверку, что датасет лежит внутри папки
    # 2. 


if __name__ == "__main__":
    train()