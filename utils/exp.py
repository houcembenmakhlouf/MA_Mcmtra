import yaml
from easydict import EasyDict as edict


def init_experiment(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)
