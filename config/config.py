import json
from pathlib import Path
import os
import glob

def get_config():
    config = {}
    # Dataset
    config["lang_src"] = "vi"
    config["lang_tgt"] = "en"

    # Device
    config["device"] = "cuda"

    # Model
    config["d_model"] = 512
    config["num_encoder"] = 6
    config["num_decoder"] = 6
    config["nhead"] = 8
    config["d_ff"] = 2048
    config["dropout"] = 0.1
    config["max_len"] = 100
    
    # Word tokenizer
    config["underthesea"] = True
    config["pyvi"] = False

    # Optimier: Adam
    config["weight_decay"] = 0
    config["lr"] = 1e-4
    config["lr_scheduler"] = False
    config["eps"] = 1e-9
    config["betas"] = (0.9, 0.98)

    # Scheduler (config["lr_scheduler"] = True)
    ## LambdaLR
    config["lambdalr"] = False
    config["warmup_steps"] = 4000
    ## StepLR
    config["steplr"] = False
    config["step_size_steplr"] = 1
    config["gamma_steplr"] = 0.5

    # Loss function: Cross_entropy
    config["label_smoothing"] = 0.1

    # Train
    config["batch_size_train"] = 32
    config["batch_size_validation"] = 32
    config["num_epochs"] = 10
    config["train_size"] = 0.9
    config["num_bleu_validation"] = 100

    # Validation Bleu metric
    config["max_beam"] = 1

    # Test
    config["beam_test"] = 2

    # Save
    config["tokenizer_file"] = "tokenizer_{0}.json"
    config["experiment_name"] = "results_train"
    config["save_config"] = "config/config_{0}.json"
    config["model_folder"] = "weights"
    config["model_basename"] = "tmodel_"
    config["data_path"] = "data"
    config["map_data_path"] = "map_data"
    config["save_config_pattern"] = "config/config_*"

    # Different
    config["preload"] = "latest"

    return config

def get_all_config(config):
    list_config = sorted(glob.glob(config["save_config_pattern"]))
    for config in list_config:
        with open(config, "r") as f:
            data = json.load(f)
        print()
        for key, val in data.items():
            print(f"{key}: {val}")
        print()

def save_config(config, epoch):
    if "save_config" in config:
        _config = config
        _config["epoch"] = f"epoch_{epoch}"
        with open(config["save_config"].format(_config["epoch"]), "w") as f:
            json.dump(_config, f)
        print(f"Đã lưu cấu hình tại epoch {epoch}")
    else:
        print(f"Không tìm thấy nơi để lưu cấu hình")

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files

def create_dic(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as error:
        print(f"Đã có lỗi khi tạo thư mục {path}: {error}")

def create_all_dic(config):
    create_dic(config["experiment_name"])
    create_dic(config["model_folder"])
    create_dic("config")

__all__ = [
    "get_config",
    "save_config",
    "weights_file_path",
    "weights_file_path",
    "create_all_dic",
    "get_all_config"
]