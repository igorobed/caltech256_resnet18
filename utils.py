from os import listdir, mkdir


def make_runs_subdir(mode="train"):
    """
    Функция создает подпапку, в которой будут храниться
    результаты обучения/валидации
    """
    runs_files = listdir("./runs/")
    run_subdirs = [item for item in runs_files if "." not in item and "run_" in item]
    if len(run_subdirs) == 0:
        subdir_name = "run_0"    
    else:
        max_num = max([int(item.split("_")[-1]) for item in run_subdirs])
        subdir_name = "run_" + str(max_num + 1)
    
    mkdir("./runs/" + subdir_name)
    # создадим также подпапку для весов, если mode=="train"
    if mode == "train":
        mkdir("./runs/" + subdir_name + "/weights")