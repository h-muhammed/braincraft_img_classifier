import torch
import os
import warnings
warnings.filterwarnings("ignore")

# Some basic setup:
from utils import default_argument_parser
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

from config.default import get_cfg_defaults
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

# used to traning
register_coco_instances("my_dataset_train", {
}, "dataset/coco/annotations/train_annotations.json", "dataset/coco/train2017")
dataset_metadata = MetadataCatalog.get("my_dataset_train")
# get the actual internal representation of the catalog stores
# information about the datasets and how to obtain them.
# The internal format uses one dict to represent the annotations of one image.
dataset_dicts = DatasetCatalog.get("my_dataset_train")
# print(dataset_dicts)

# parse argument from cli
args = default_argument_parser().parse_args()
# configuration
ep_config = get_cfg_defaults()
if args.experiment_file is not None:
    # configuration for this experiment
    ep_config.merge_from_file(args.experiment_file)
ep_config.freeze()
print(ep_config)

# training config
cfg = get_cfg()
cfg.merge_from_file(ep_config.TRAIN.CONFIG_FILE_PATH)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.MODEL.DEVICE = args.device
cfg.DATALOADER.NUM_WORKERS = 2
# Let training initialize from model zoo(Pretrained on ImageNet
cfg.MODEL.WEIGHTS = ep_config.TRAIN.MODEL_WEIGHTS_PATH
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = ep_config.TRAIN.BASE_LR  # pick a good LR
cfg.SOLVER.MAX_ITER = ep_config.TRAIN.MAX_ITER
# faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# 20 classes in this custom dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = ep_config.TRAIN.NUM_CLASSES

# create folder if not exist
os.makedirs(ep_config.TRAIN.LOG_OUTPUT_PATH, exist_ok=True)
cfg.OUTPUT_DIR = ep_config.TRAIN.LOG_OUTPUT_PATH


def trainer():

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':

    trainer()
