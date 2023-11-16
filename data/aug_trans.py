"""
Follows Hugging Face docs for DETR
"""

import albumentations
import numpy as np
import torch
from transformers import AutoImageProcessor
import datasets
import os
import json
from datasets import load_dataset


def create_original_dataset(dataset_path):
    # Use the default Hugging Face dataset loading script
    dataset = load_dataset("imagefolder", data_dir=dataset_path, split="train")
    return dataset

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

# transforming a batch
def transform_aug_ann(examples):
    image_ids = [examples["image_id"]]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        # Calcuate area of bounding box
        areas = []
        for box in out["bboxes"]:
            areas.append(box[2] * box[3])
        area.append(areas)

        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

def aug_trans_dataset(dataset_path):
    """
    Create the dataset with augmentation and transformation.

    Args:
        dataset_path (str): The directory containing the images.
    """
    dataset = create_original_dataset(dataset_path)
    return dataset.with_transform(transform_aug_ann)
