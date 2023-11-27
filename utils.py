from os import listdir, mkdir
import matplotlib.pyplot as plt


def make_runs_subdir(mode: str = "train") -> str:
    """
    Функция создает подпапку, в которой будут храниться
    результаты обучения/валидации
    """
    runs_files = listdir("./runs/")
    run_subdirs = [item for item in runs_files if "." not in item and mode in item]
    if len(run_subdirs) == 0:
        subdir_name = f"run_{mode}_0"
    else:
        max_num = max([int(item.split("_")[-1]) for item in run_subdirs])
        subdir_name = f"run_{mode}_{str(max_num + 1)}"
    
    mkdir("./runs/" + subdir_name)
    # создадим также подпапку для весов, если mode=="train"
    if mode == "train":
        mkdir("./runs/" + subdir_name + "/weights")

    return subdir_name


def save_loss_img(train_loss_lst, val_loss_lst, save_path):
    plt.figure()
    plt.xlabel("epoch")
    plt.plot(train_loss_lst, label="train_loss")
    plt.plot(val_loss_lst, label="valid_loss")
    plt.legend()
    plt.savefig(save_path)


def save_acc_img(val_acc_lst, save_path):
    plt.figure()
    plt.xlabel("epoch")
    plt.plot(val_acc_lst, label="valid acc")
    plt.legend()
    plt.savefig(save_path)