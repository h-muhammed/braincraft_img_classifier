# model config for this training/test
from config.default import get_cfg_defaults
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
# used to encode segmentation mask and parser argument in cli
from utils import binary_mask_to_rle
from utils import default_argument_parser

import cv2
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
setup_logger()

register_coco_instances("my_dataset_test", {
}, "../dataset/coco/annotations/val_annotations.json",
    "../dataset/coco/val2017")
dataset_metadata = MetadataCatalog.get("my_dataset_test")
# get the actual internal representation of the catalog stores
# information about the datasets and how to obtain them. The internal
# format uses one dict to represent the annotations of one image.
dataset_dicts = DatasetCatalog.get("my_dataset_test")
print(dataset_metadata)
# print(dataset_dicts)

# parse argument from cli
args = default_argument_parser().parse_args()

# configuration
ep_config = get_cfg_defaults()
if args.experiment_file is not None:
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
cfg.MODEL.WEIGHTS = ep_config.TRAIN.MODEL_WEIGHTS_PATH
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = ep_config.TRAIN.BASE_LR
cfg.SOLVER.MAX_ITER = ep_config.TRAIN.MAX_ITER
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = ep_config.TRAIN.NUM_CLASSES

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously.
# We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = ep_config.TEST.MODEL_WEIGHTS_PATH_TEST
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ep_config.TEST.SCORE_THRESH_TEST
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set a custom testing threshold

predictor = DefaultPredictor(cfg)


def evaluate(folder_name):

    categories = ''
    pre_name = ''
    coco_dt = []  # Used to save list of dictionaries
    # iterate each image in testing dataset and form the submmision file
    print('\n***____________________________________***///\n\n')
    for d in tqdm(dataset_dicts, ncols=100, mininterval=1, desc="Img Infer"):
        img_id = d["image_id"]
        im = cv2.imread(d["file_name"])

        outputs = predictor(im)

        instances = outputs["instances"]
        n_instances = len(instances)
        classes = instances.pred_classes
        masks = instances.pred_masks
        scores = instances.scores

        masks = masks.permute(1, 2, 0).to("cpu").numpy()

        classes = classes.tolist()
        person = classes.count(0)
        objects = classes.count(1)

        if objects:
            categories = 'f-object'
        elif person > 2:
            categories = 'g-person'
        else:
            categories = 'person'
        v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # extract file name from full path
        save_filename = os.path.split(d['file_name'])[1]
        if categories == 'f-object':
            pre_name = 'pred_f-object_'
        elif categories == 'g-person':
            pre_name = 'pred_g-person_'
        else:
            pre_name = 'pred_person_'

        path = os.getcwd()
        # Joins the folder that we wanted to create
        # folder_name = 'prediction'
        path = os.path.join(path, folder_name)

        # Creates the folder, and checks if it is created or not.
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, pre_name + save_filename),
                    out.get_image()[:, :, ::-1])  # save result file

        for i in range(n_instances):  # Loop all instances
            # save information of the instance in a
            # dictionary then append on coco_dt list
            pred = {}
            pred['image_id'] = img_id
            pred['category_id'] = int(classes[i]) + 1
            pred['segmentation'] = binary_mask_to_rle(masks[:, :, i])
            pred['score'] = float(scores[i])
            coco_dt.append(pred)

    print('prediction is successful!\n')
    with open("eval_result.json", "w") as f:
        json.dump(coco_dt, f)


if __name__ == '__main__':

    folder_name = '../prediction'
    evaluate(folder_name)
