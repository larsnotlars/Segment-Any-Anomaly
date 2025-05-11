import os
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import numpy as np
from pathlib import Path
import cv2
import glob

LH_DIR = "../datasets/lufthansa"

lufthansa_classes = ['lufthansa']

def load_lufthansa(category, k_shot, experiment_indx):
    assert category == "lufthansa", "Only 'lufthansa' category is supported."
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    luf_root = Path( 
        LH_DIR
    )  # adjust as needed
    ann_file = luf_root / "_annotations.coco.json"
    coco = COCO(str(ann_file))

    test_img_paths, test_gt_paths, test_labels, test_types = [], [], [], []
    train_img_paths, train_gt_paths, train_labels, train_types = [], [], [], []

    all_img_ids = coco.getImgIds()
    selected_train_ids = all_img_ids[:k_shot] if k_shot > 0 else []

    for img_id in all_img_ids:
        info = coco.loadImgs(img_id)[0]
        img_path = str(luf_root / info["file_name"])
        h, w = info["height"], info["width"]

        mask_total = np.zeros((h, w), dtype=np.uint8)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        is_good = len(anns) == 0

        for ann in anns:
            seg = ann.get("segmentation")
            if seg:
                if isinstance(seg, list):
                    rles = mask_utils.frPyObjects(seg, h, w)
                    rle = mask_utils.merge(rles)
                    m = mask_utils.decode(rle)
                else:
                    m = mask_utils.decode(seg)
                mask_total |= m
            else:
                x, y, bw, bh = ann["bbox"]
                x0, y0, x1, y1 = map(int, [x, y, x + bw, y + bh])
                mask_total[y0:y1, x0:x1] = 1

        is_train = img_id in selected_train_ids
        gt_mask_path = None

        # Save GT mask temporarily
        if not is_good:
            gt_mask_path = f"/tmp/luf_gt_{img_id}.png"
            cv2.imwrite(gt_mask_path, (mask_total * 255).astype(np.uint8))

        if is_train:
            train_img_paths.append(img_path)
            train_gt_paths.append(gt_mask_path if gt_mask_path else 0)
            train_labels.append(0 if is_good else 1)
            train_types.append("good" if is_good else "anomaly")
        else:
            test_img_paths.append(img_path)
            test_gt_paths.append(gt_mask_path if gt_mask_path else 0)
            test_labels.append(0 if is_good else 1)
            test_types.append("good" if is_good else "anomaly")

    return (train_img_paths, train_gt_paths, train_labels, train_types), (
        test_img_paths,
        test_gt_paths,
        test_labels,
        test_types,
    )
