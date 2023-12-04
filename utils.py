from os import listdir, mkdir
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch.onnx as t_onnx
import torch


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
    if mode == "train" or mode == "ssl" or mode == "ssl_finetune":
        mkdir("./runs/" + subdir_name + "/weights")

    return subdir_name


def convert_to_onnx(model, save_path, device="cpu"):
    input_rand = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)
    model.to(device)
    out = model(input_rand)
    t_onnx.export(
        model,
        input_rand,
        save_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size"
            },
            "output": {
                0: "batch_size"
            }
        }
    )


def onnx_predict(inputs: np.array, onnx_session=None, onnx_path=None):
    if onnx_session is None and onnx_path is not None:
        onnx_session = onnxruntime.InferenceSession(onnx_path)
    out = onnx_session.run(None, {"input": inputs})
    preds = out[0].argmax(1)
    return preds


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