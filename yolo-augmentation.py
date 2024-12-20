import albumentations as A
import cv2
import numpy as np
import os
import json

class YOLOAugmentation:
    def __init__(self, 
                 img_dir, 
                 label_dir, 
                 output_img_dir, 
                 output_label_dir, 
                 num_augmentations=5):
        """
        Initialize YOLO dataset augmentation class
        
        :param img_dir: Directory containing original images
        :param label_dir: Directory containing YOLO format label files
        :param output_img_dir: Directory to save augmented images
        :param output_label_dir: Directory to save augmented label files
        :param num_augmentations: Number of augmentations to generate per image
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.output_img_dir = output_img_dir
        self.output_label_dir = output_label_dir
        self.num_augmentations = num_augmentations
        
        # Create output directories if they don't exist
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            # Image-level augmentations
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomFog(p=0.3),
            A.GaussNoise(p=0.3),
            
            # Geometric transformations that support bounding boxes
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5, 
                border_mode=cv2.BORDER_CONSTANT
            )
        ], bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels']
        ))
    
    def read_yolo_labels(self, label_path):
        """
        Read YOLO format label file
        
        :param label_path: Path to YOLO label file
        :return: List of [class_id, x_center, y_center, width, height]
        """
        with open(label_path, 'r') as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]
        return labels
    
    def write_yolo_labels(self, label_path, labels):
        """
        Write labels in YOLO format
        
        :param label_path: Path to save label file
        :param labels: List of [class_id, x_center, y_center, width, height]
        """
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')
    
    def augment_dataset(self):
        """
        Perform augmentation on entire dataset
        """
        # Iterate through all images in the input directory
        for img_filename in os.listdir(self.img_dir):
            if img_filename.endswith(('.jpg', '.png', '.jpeg')):
                # Read image and corresponding labels
                img_path = os.path.join(self.img_dir, img_filename)
                label_filename = os.path.splitext(img_filename)[0] + '.txt'
                label_path = os.path.join(self.label_dir, label_filename)
                
                # Read image and labels
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Read YOLO format labels
                labels = self.read_yolo_labels(label_path)
                
                # Separate class labels and bounding boxes
                class_labels = [int(label[0]) for label in labels]
                bboxes = [label[1:] for label in labels]
                
                # Perform multiple augmentations
                for i in range(self.num_augmentations):
                    # Apply augmentation
                    augmented = self.transform(
                        image=image, 
                        bboxes=bboxes, 
                        class_labels=class_labels
                    )
                    
                    # Generate output filenames
                    base_name = os.path.splitext(img_filename)[0]
                    aug_img_filename = f"{base_name}_aug_{i}.jpg"
                    aug_label_filename = f"{base_name}_aug_{i}.txt"
                    
                    # Save augmented image
                    aug_img_path = os.path.join(
                        self.output_img_dir, 
                        aug_img_filename
                    )
                    aug_img = cv2.cvtColor(
                        augmented['image'], 
                        cv2.COLOR_RGB2BGR
                    )
                    cv2.imwrite(aug_img_path, aug_img)
                    
                    # Prepare and save augmented labels
                    aug_labels = [
                        [class_label] + list(bbox) 
                        for class_label, bbox in zip(
                            augmented['class_labels'], 
                            augmented['bboxes']
                        )
                    ]
                    aug_label_path = os.path.join(
                        self.output_label_dir, 
                        aug_label_filename
                    )
                    self.write_yolo_labels(aug_label_path, aug_labels)

# Example usage
if __name__ == "__main__":
    augmenter = YOLOAugmentation(
        img_dir='path/to/original/images',
        label_dir='path/to/original/labels',
        output_img_dir='path/to/augmented/images',
        output_label_dir='path/to/augmented/labels',
        num_augmentations=5
    )
    augmenter.augment_dataset()
