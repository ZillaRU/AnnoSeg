import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, bbox):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    x1, y1, x2, y2 = bbox
    segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (255, 255, 255))
    # transparency_mask = np.zeros_like((), dtype=np.uint8)
    transparency_mask = np.zeros(
        (image_array.shape[0], image_array.shape[1]), dtype=np.uint8
    )
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def format_results(masks, scores, logits, filter=0):
    annotations = []
    n = len(scores)
    for i in range(n):
        annotation = {}

        mask = masks[i]
        tmp = np.where(mask != 0)
        if np.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask
        annotation["bbox"] = [
            np.min(tmp[0]),
            np.min(tmp[1]),
            np.max(tmp[1]),
            np.max(tmp[0]),
        ]
        annotation["score"] = scores[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations


def filter_masks(annotations):  # filter the overlap mask
    annotations.sort(key=lambda x: x["area"], reverse=True)
    to_remove = set()
    for i in range(0, len(annotations)):
        a = annotations[i]
        for j in range(i + 1, len(annotations)):
            b = annotations[j]
            if i != j and j not in to_remove:
                # check if
                if b["area"] < a["area"]:
                    if (a["segmentation"] & b["segmentation"]).sum() / b[
                        "segmentation"
                    ].sum() > 0.8:
                        to_remove.add(j)

    return [a for i, a in enumerate(annotations) if i not in to_remove], to_remove


def box_prompt(masks, bbox, target_height, target_width):
    h = masks.shape[1]
    w = masks.shape[2]
    if h != target_height or w != target_width:
        bbox = [
            int(bbox[0] * w / target_width),
            int(bbox[1] * h / target_height),
            int(bbox[2] * w / target_width),
            int(bbox[3] * h / target_height),
        ]
    bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
    bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

    # IoUs = torch.zeros(len(masks), dtype=torch.float32)
    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

    masks_area = torch.sum(masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2))
    orig_masks_area = torch.sum(masks, dim=(1, 2))

    union = bbox_area + orig_masks_area - masks_area
    IoUs = masks_area / union
    max_iou_index = torch.argmax(IoUs)

    return masks[max_iou_index].cpu().numpy(), max_iou_index


def point_prompt(masks, points, point_label, target_height, target_width):  # numpy 处理
    h = masks[0]["segmentation"].shape[0]
    w = masks[0]["segmentation"].shape[1]
    if h != target_height or w != target_width:
        points = [
            [int(point[0] * w / target_width), int(point[1] * h / target_height)]
            for point in points
        ]
    onemask = np.zeros((h, w))
    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation["segmentation"]
        else:
            mask = annotation
        for i, point in enumerate(points):
            if mask[point[1], point[0]] == 1 and point_label[i] == 1:
                onemask += mask
            if mask[point[1], point[0]] == 1 and point_label[i] == 0:
                onemask -= mask
    onemask = onemask >= 1
    return onemask, 0
