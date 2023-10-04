import argparse
import wandb
from sray.train.trainer import Trainer
from sray.utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
parser.add_argument('--model-path', type=str, default='model.pth')
flags = parser.parse_args()
cfg = load_cfg(flags.cfg)
trainer = Trainer(cfg)
run = wandb.init(project="sray", config=cfg)
run.name = cfg['name']
trainer.eval(flags.model_path)
run.finish()
