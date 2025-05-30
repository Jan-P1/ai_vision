import albumentations as A
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

transform = A.Compose([
    A.RandomRotate90(),
    A.Transpose(),
    A.GaussNoise(),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.RandomBrightnessContrast(),
    ], p=0.3),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, min_area=3700, label_fields=['class_labels']))

# Paths
input_dir_images = "INPUT_DIR_IMAGES"  # Replace with your input directory for images
output_dir_images = "OUTPUT_DIR_IMAGES"  # Replace with your output directory for images
input_dir_labels = "INPUT_DIR_LABELS"  # Replace with your input directory for labels
output_dir_labels = "OUTPUT_DIR_LABELS"  # Replace with your output directory for labels

os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_labels, exist_ok=True)

num_augmentations = 3

for file_name in os.listdir(input_dir_images):
    if not file_name.lower().endswith(('.jpg', '.jpeg')):
        continue

    image_path = os.path.join(input_dir_images, file_name)

    # Validate image
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        print(f"Skipping invalid image {file_name}: {e}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {file_name}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load label
    base_name, ext = os.path.splitext(file_name)
    label_path = os.path.join(input_dir_labels, f"{base_name}.txt")
    if not os.path.exists(label_path):
        print(f"Label file not found for {file_name}")
        continue

    try:
        labels = np.loadtxt(label_path)
        if labels.ndim == 1:
            labels = labels.reshape(1, -1)
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        continue

    bboxes = labels[:, 1:]
    class_labels = labels[:, 0].astype(int)

    for i in range(num_augmentations):
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_class_labels = transformed["class_labels"]

        # Save image
        output_file_name = f"{base_name}_aug_{i+1}.jpg"
        output_path = os.path.join(output_dir_images, output_file_name)
        cv2.imwrite(output_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

        # Save labels
        output_label_file = f"{base_name}_aug_{i+1}.txt"
        output_label_path = os.path.join(output_dir_labels, output_label_file)
        augmented_labels = np.hstack([
            np.array(transformed_class_labels).reshape(-1, 1),
            np.array(transformed_bboxes)
        ])
        np.savetxt(output_label_path, augmented_labels, fmt="%.6f")
