from os import listdir, mkdir


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