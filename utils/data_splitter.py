import os
import shutil
from sklearn.model_selection import train_test_split



class DatasetSplitter:
    """Handles the 7:2:1 train/val/test split of the dataset"""
    
    def __init__(self, root_dir, output_dir):
        self.root_dir = root_dir
        self.output_dir = output_dir
        
    def split_dataset(self):
        """Split dataset into train/val/test with 7:2:1 ratio"""
        # Create output directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split), exist_ok=True)
        
        # Get all animal classes
        animal_classes = [d for d in os.listdir(self.root_dir) 
                         if os.path.isdir(os.path.join(self.root_dir, d))]
        
        print(f"Found {len(animal_classes)} animal classes: {animal_classes}")
        
        for animal_class in animal_classes:
            class_path = os.path.join(self.root_dir, animal_class)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            # Split images: 70% train, 20% val, 10% test
            train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
            val_images, test_images = train_test_split(temp_images, test_size=0.33, random_state=42)
            
            print(f"{animal_class}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
            
            # Create class directories in each split
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(self.output_dir, split, animal_class), exist_ok=True)
            
            # Copy images to respective directories
            for img in train_images:
                shutil.copy2(os.path.join(class_path, img), 
                           os.path.join(self.output_dir, 'train', animal_class, img))
            
            for img in val_images:
                shutil.copy2(os.path.join(class_path, img), 
                           os.path.join(self.output_dir, 'val', animal_class, img))
            
            for img in test_images:
                shutil.copy2(os.path.join(class_path, img), 
                           os.path.join(self.output_dir, 'test', animal_class, img))