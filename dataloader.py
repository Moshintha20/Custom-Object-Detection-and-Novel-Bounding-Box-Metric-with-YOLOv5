import os
import argparse
import kagglehub
import shutil
import cv2
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def download_dataset(output_dir):
    """Download the dataset from Kaggle and extract it to the specified output directory."""
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("andrewmvd/dog-and-cat-detection")
    extracted_path = os.path.join(output_dir, "original_dataset")
    os.makedirs(extracted_path, exist_ok=True)

    # Move all files from the extracted dataset folder to the correct directory
    for item in os.listdir(dataset_path):
        shutil.move(os.path.join(dataset_path, item), extracted_path)

    # Remove the now-empty original extracted folder
    shutil.rmtree(dataset_path, ignore_errors=True)

    print(f"Dataset downloaded and extracted to: {extracted_path}")

    return extracted_path

def parse_annotations(annotation_dir):
    """Parse XML annotations and create a DataFrame with bounding box information."""
    records = []
    annotations = os.listdir(annotation_dir)

    for annot_file in annotations:
        with open(os.path.join(annotation_dir, annot_file), 'r') as f:
            file = f.read()
        data = BeautifulSoup(file, "xml")
        
        objects = data.find_all('object')
        filename = data.find("filename").text
        img_width = int(data.find('size').width.text)
        img_height = int(data.find('size').height.text)

        for obj in objects:
            class_name = obj.find('name').text
            x_min = int(obj.find('bndbox').xmin.text) / img_width
            y_min = int(obj.find('bndbox').ymin.text) / img_height
            x_max = int(obj.find('bndbox').xmax.text) / img_width
            y_max = int(obj.find('bndbox').ymax.text) / img_height

            records.append([filename, class_name, x_min, y_min, x_max, y_max])

    df = pd.DataFrame(records, columns=['filename', 'class', 'x_min', 'y_min', 'x_max', 'y_max'])
    df.drop_duplicates(subset=['filename'], inplace=True)
    df['class'] = df['class'].apply(lambda x: 1 if x == 'dog' else 0)
    return df

def resize_images(image_dir, output_dir):
    """Resize images to 640x640 and save them in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)            
    image_files = os.listdir(image_dir)

    print("Resizing images...")
    for image_file in tqdm(image_files, desc="Processing Images", unit="img"):
        img_path = os.path.join(image_dir, image_file)
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_resized = img.resize((640, 640))
                img_resized.save(os.path.join(output_dir, image_file))
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print("Resizing complete!")

def split_dataset(df, output_dir):
    """Split dataset into train, validation, and test sets."""
    df_cat = df[df['class'] == 0]
    df_dog = df[df['class'] == 1]

    train_cat, temp_cat = train_test_split(df_cat, test_size=0.3, random_state=42, stratify=df_cat['class'])
    val_cat, test_cat = train_test_split(temp_cat, test_size=0.5, random_state=42, stratify=temp_cat['class'])

    train_dog, temp_dog = train_test_split(df_dog, test_size=0.3, random_state=42, stratify=df_dog['class'])
    val_dog, test_dog = train_test_split(temp_dog, test_size=0.5, random_state=42, stratify=temp_dog['class'])

    train_df = pd.concat([train_cat, train_dog]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat([val_cat, val_dog]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([test_cat, test_dog]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Train set images: {len(train_df)}, Validation set images: {len(val_df)}, Test set images: {len(test_df)}")
    return train_df, val_df, test_df

def convert_to_yolo(df, image_dir, output_dir):
    """Convert bounding boxes to YOLO format and save label files."""
    os.makedirs(output_dir, exist_ok=True)
    
    class_mapping = {0: 0, 1: 1}  # 0 -> cat, 1 -> dog

    for _, row in df.iterrows():
        filename = row["filename"]
        class_id = class_mapping[row["class"]]
        x_min, y_min, x_max, y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        label_filename = os.path.join(output_dir, filename.replace(".png", ".txt"))
        with open(label_filename, "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def copy_images(df, src_folder, dst_folder):
    """Copy images from source to destination."""
    os.makedirs(dst_folder, exist_ok=True)
    for filename in df["filename"]:
        src = os.path.join(src_folder, filename)
        dst = os.path.join(dst_folder, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)

def create_data_yaml(output_dir):
    """Create a data.yaml file for YOLO training."""
    yaml_content = f"""train: {output_dir}/train/images
val: {output_dir}/val/images
test: {output_dir}/test/images

names:
    0: Cat
    1: Dog
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"data.yaml created at {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare a dataset for YOLOv5.")
    parser.add_argument("--output", type=str, required=True, help="Output directory to store the processed dataset.")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir = download_dataset(output_dir)
    
    image_dir = os.path.join(dataset_dir, "images")
    annotation_dir = os.path.join(dataset_dir, "annotations")

    df = parse_annotations(annotation_dir)

    resized_img_dir = os.path.join(output_dir, "resized_images")
    resize_images(image_dir, resized_img_dir)

    train_df, val_df, test_df = split_dataset(df, output_dir)

    dataset_structure = os.path.join(output_dir, "new_dataset")
    for split, df_split in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        convert_to_yolo(df_split, resized_img_dir, os.path.join(dataset_structure, split, "labels"))
        copy_images(df_split, resized_img_dir, os.path.join(dataset_structure, split, "images"))

    create_data_yaml(dataset_structure)
    print(f"New Dataset structure created at {dataset_structure}")
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
