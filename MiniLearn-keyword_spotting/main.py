from torch.utils.data import DataLoader

from har import SampleCNN, SimpleTrainer, load_har

import torch
from converter import CMSISConverter





CONFIG = {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.0001,
    "learning_step": 5000,
    "learning_gamma": 0.99,
    "name": "har1dcnn",
    "shape": (62,12,1),
    "dataset": load_har,
    "compilation": "gcc -g -I/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/Core/Include \
            -I/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/DSP/Include \
            -I/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/NN/Include \
            -D__ARM_ARCH_8M_BASE__ \
            /Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/NN/Source/*/*.c \
            /Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/DSP/Source/StatisticsFunctions/arm_max_q7.c \
            main.c -o main",
    "exec_path": "main",
}


def train_cifar(config):
    datasets = config["dataset"]()
    dataloaders = {
        i: DataLoader(
            sett, batch_size=config["batch_size"], shuffle=True, num_workers=2
        )
        for i, sett in zip(["train", "val", "test"], datasets)
    }
    cnn = SampleCNN(shape=config["shape"], batch_size=config["batch_size"])
    trainer = SimpleTrainer(datasets=datasets, dataloaders=dataloaders)
    cnn = trainer.train(cnn, config, config.get("name"))
    accuracy_test = trainer.evaluate(cnn)
    # cnn.load_state_dict(torch.load("har1dcnn.pth"))
    cnn.eval()

    print("Accuracy for test set with PyTorch ", accuracy_test)
    cm_converter = CMSISConverter(
        "cfiles",
        cnn,
        "weights.h",
        "parameters.h",
        8,
        config.get("compilation"),
    )
    cm_converter.convert_model(dataloaders["val"])
    cm_converter.evaluate_cmsis(config.get("exec_path"), dataloaders["test"])
    input, label = next(iter(dataloaders["test"]))
    cm_converter.sample_inference_checker(config.get("exec_path"), input, draw=True)

    # cm_converter.convert_model(dataloaders["val"])
    # #--
    # # cm_converter.generate_intermediate_values(dataloaders['val'])
    # # cm_converter.convert_model_cmsis()
    # #--
    # cm_converter.evaluate_cmsis(config.get("exec_path"), dataloaders["test"])
    # input, label = next(iter(dataloaders["test"]))
    # cm_converter.sample_inference_checker(config.get("exec_path"), input)
    # # cm_converter.sample_inference_checker(config.get("exec_path"), input, draw=True)


if __name__ == "__main__":
    train_cifar(CONFIG)
