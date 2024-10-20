import pandas as pd
import cv2
import os

# Define paths for train and validation datasets
train_csv_path = 'C:/Users/SEC/Downloads/ibm/debris-detection/train.csv'
train_img_dir = 'C:/Users/SEC/Downloads/ibm/debris-detection/train'
val_csv_path = 'C:/Users/SEC/Downloads/ibm/debris-detection/val.csv'
val_img_dir = 'C:/Users/SEC/Downloads/ibm/debris-detection/val'

# Create directories for the preprocessed dataset (if they don't already exist)
preprocessed_base_dir = 'C:/Users/SEC/Downloads/ibm/preprocessed'
os.makedirs(f'{preprocessed_base_dir}/train/images', exist_ok=True)
os.makedirs(f'{preprocessed_base_dir}/train/labels', exist_ok=True)
os.makedirs(f'{preprocessed_base_dir}/val/images', exist_ok=True)
os.makedirs(f'{preprocessed_base_dir}/val/labels', exist_ok=True)

# Function to convert bounding box to YOLO format
def convert_bbox_to_yolo(img_width, img_height, bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [x_center, y_center, width, height]

# Preprocess dataset (generalized for both train and val sets)
def preprocess_dataset(csv_path, img_dir, save_dir):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        img_id = row['ImageID']
        bboxes = eval(row['bboxes'])  # Convert the string to list format

        # Load the corresponding image
        img_path = f"{img_dir}/{img_id}.jpg"
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape

            # Save the image to the preprocessed dataset directory
            new_img_path = f"{save_dir}/images/{img_id}.jpg"
            cv2.imwrite(new_img_path, img)

            # Create the corresponding .txt file in YOLO format
            label_path = f"{save_dir}/labels/{img_id}.txt"
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    yolo_bbox = convert_bbox_to_yolo(img_width, img_height, bbox)
                    f.write(f"0 {' '.join(map(str, yolo_bbox))}\n")  # Class ID is 0 for debris

# Preprocess both train and validation datasets
preprocess_dataset(train_csv_path, train_img_dir, f"{preprocessed_base_dir}/train")
preprocess_dataset(val_csv_path, val_img_dir, f"{preprocessed_base_dir}/val")

print("Preprocessing completed!")
