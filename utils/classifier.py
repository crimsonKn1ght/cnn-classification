import os
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


from models.model import VGG19

class AnimalClassifier:
    """Main classifier class that handles training and inference"""
    
    def __init__(self, data_dir, batch_size=32, learning_rate=0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.setup_transforms()
        self.setup_data_loaders()
        self.setup_model()
        
        print("AnimalClassifier initialized successfully!")
    
    def print_cuda_info(self):
        """Print CUDA system information"""
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def setup_transforms(self):
        """Setup data transformations for training and validation/testing"""
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup_data_loaders(self):
        """Setup data loaders for train, validation, and test sets"""
        train_dataset = ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=self.train_transform
        )
        
        val_dataset = ImageFolder(
            os.path.join(self.data_dir, 'val'),
            transform=self.val_test_transform
        )
        
        test_dataset = ImageFolder(
            os.path.join(self.data_dir, 'test'),
            transform=self.val_test_transform
        )
        
        # Enable pin_memory only if CUDA is available for faster GPU transfer
        pin_memory = torch.cuda.is_available()
        num_workers = 4 if torch.cuda.is_available() else 0  # Avoid multiprocessing issues on some systems
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                     shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                   shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                    shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        self.num_classes = len(train_dataset.classes)
        self.class_names = train_dataset.classes
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    def setup_model(self):
        """Setup model, loss function, and optimizer"""
        self.model = VGG19(num_classes=self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        print(f"Using device: {self.device}")
        
        # Print memory usage if CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    def train_single_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss / len(self.train_loader), 100. * correct / total

    def validate_model(self, data_loader):
        """Validate the model on given data loader"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return running_loss / len(data_loader), 100. * correct / total

    def train(self, num_epochs=25):
        """Train the model for specified number of epochs"""
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_acc = 0.0
        
        print("Starting training...")
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # Training
            train_loss, train_acc = self.train_single_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate_model(self.val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            
            self.scheduler.step()
        
        # Plot training history
        self.plot_training_history(train_losses, train_accs, val_losses, val_accs)
        
        return train_losses, train_accs, val_losses, val_accs

    def plot_training_history(self, train_losses, train_accs, val_losses, val_accs):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def test(self):
        """Test the model on test set"""
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        test_loss, test_acc = self.validate_model(self.test_loader)
        print(f'\nTest Results:')
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_acc:.2f}%')
        
        return test_loss, test_acc

    def predict(self, image_path):
        """Predict class for a single image"""
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_test_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return self.class_names[predicted_class], confidence
