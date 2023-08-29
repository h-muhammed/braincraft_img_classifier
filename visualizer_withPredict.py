from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from config.default import get_cfg_defaults
from utils import default_argument_parser
from detectron2.utils.visualizer import Visualizer

import cv2
import os
import torch
import warnings
warnings.filterwarnings("ignore")

# Setup detectron2 logger
setup_logger()

# model config for this training/test
torch.cuda.empty_cache()

register_coco_instances("my_dataset_test", {
}, "dataset/coco/annotations/val_annotations.json", "dataset/coco/val2017")
dataset_metadata = MetadataCatalog.get("my_dataset_test")
# get the actual internal representation of the catalog
# stores information about the datasets and how to obtain them.
# The internal format uses one dict to represent the annotations of one image.
dataset_dicts = DatasetCatalog.get("my_dataset_test")
print(dataset_metadata)
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

# training config (detectron2)
cfg = get_cfg()
cfg.merge_from_file(ep_config.TRAIN.CONFIG_FILE_PATH)
cfg.MODEL.DEVICE = args.device
cfg.DATASETS.TRAIN = ()
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
# Let training initialize from model zoo(Pretrained on ImageNet)
cfg.MODEL.WEIGHTS = ep_config.TRAIN.MODEL_WEIGHTS_PATH
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = ep_config.TRAIN.BASE_LR  # pick a good LR
cfg.SOLVER.MAX_ITER = ep_config.TRAIN.MAX_ITER
# faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# 20 classes in this custom dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = ep_config.TRAIN.NUM_CLASSES

# Inference should use the config with parameters that are used in training
# cfg now already contains everything
# we've set previously. We changed it a little bit for inference:
# path to the model we just trained
cfg.MODEL.WEIGHTS = ep_config.TEST.MODEL_WEIGHTS_PATH_TEST
# set a custom testing threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ep_config.TEST.SCORE_THRESH_TEST
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set a custom testing threshold

predictor = DefaultPredictor(cfg)


def single_pred():

    im = cv2.imread(args.pred_img_file)
    outputs = predictor(im)
    # print(outputs['instances'])
    classes = outputs['instances'].pred_classes
    classes = classes.tolist()
    person = classes.count(0)
    objects = classes.count(1)
    print('\n\\\***____________________________________***///\n\n')
    if objects:
        print('      The image is belong to focused object class!\n')
    elif person > 2:
        print('      The image is belong to group of person class!\n')
    else:
        print('    The image is belong to person class!\n')
    v = Visualizer(im[:, :, ::-1],
                   metadata=dataset_metadata,
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # extract file name from full path
    save_filename = os.path.split(args.pred_img_file)[1]
    print(save_filename)

    cv2.imwrite("pred_" + save_filename, out.get_image()
                [:, :, ::-1])  # save result file
    # print(args.pred_img_file)


if __name__ == '__main__':

    single_pred()
