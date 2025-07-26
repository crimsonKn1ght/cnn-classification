# 🐾 Animal Classifier using VGG19

A deep learning-based image classification system that can identify different animal species using the powerful VGG19 convolutional neural network architecture.

## 📋 Overview

This project implements a comprehensive animal classification pipeline using PyTorch and the VGG19 architecture. The system can automatically categorize animal images into different species with high accuracy, making it useful for wildlife research, educational purposes, and conservation efforts.

## 🎯 Features

- **VGG19 Architecture**: Utilizes the proven 19-layer VGG deep neural network for robust feature extraction
- **Automated Dataset Management**: Intelligent 7:2:1 train/validation/test dataset splitting
- **Data Augmentation**: Advanced image preprocessing and augmentation for improved model generalization
- **Training Visualization**: Real-time training progress tracking with loss and accuracy plots
- **GPU Acceleration**: CUDA support for faster training on compatible hardware
- **Model Persistence**: Automatic saving of best-performing models during training
- **Single Image Prediction**: Easy-to-use inference for classifying individual images

## 🏗️ Architecture

The model is based on **VGG19** (Visual Geometry Group), a deep convolutional neural network with:

- **16 Convolutional Layers** organized in 5 blocks with increasing filter depths (64→128→256→512→512)
- **3 Fully Connected Layers** for final classification
- **ReLU Activation Functions** for non-linearity
- **Max Pooling** for spatial dimension reduction
- **Dropout Layers** for regularization and overfitting prevention

### Model Statistics
- **Total Parameters**: ~143 million
- **Input Size**: 224×224×3 RGB images
- **Architecture Depth**: 19 layers
- **Feature Maps**: Up to 512 channels in deeper layers

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Dataset Preparation

1. **Organize your dataset** in the following structure:
```
dataset/
├── animal_class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── animal_class_2/
│   ├── image1.jpg
│   └── ...
└── ...
```

2. **Split the dataset** automatically:
```python
from utils.data_splitter import DatasetSplitter

splitter = DatasetSplitter('path/to/dataset', 'split_dataset')
splitter.split_dataset()
```

### Training the Model

```python
from utils.classifier import AnimalClassifier

# Initialize the classifier
classifier = AnimalClassifier(
    data_dir='split_dataset',
    batch_size=32,
    learning_rate=0.001
)

# Train the model
train_losses, train_accs, val_losses, val_accs = classifier.train(num_epochs=25)

# Evaluate on test set
test_loss, test_acc = classifier.test()
```

### Making Predictions

```python
# Predict a single image
predicted_class, confidence = classifier.predict('path/to/image.jpg')
print(f"Predicted: {predicted_class} (Confidence: {confidence:.3f})")
```

## 📊 Training Features

### Data Augmentation
- **Geometric Transforms**: Random horizontal flips and rotations
- **Color Adjustments**: Brightness, contrast, saturation, and hue variations
- **Normalization**: ImageNet mean and standard deviation normalization

### Training Optimization
- **Adam Optimizer** with weight decay for regularization
- **Learning Rate Scheduling** with step decay
- **Early Stopping** based on validation accuracy
- **Automatic Model Checkpointing** for best performance

### Monitoring and Visualization
- **Real-time Progress Bars** showing loss and accuracy
- **Training History Plots** for loss and accuracy curves
- **GPU Memory Monitoring** for resource optimization
- **Comprehensive Logging** of training metrics

## 🎯 Performance

The VGG19-based classifier offers:

- **High Accuracy**: Proven architecture for image classification tasks
- **Robust Feature Extraction**: Deep convolutional layers capture complex patterns
- **Transfer Learning Ready**: Pre-trained weights can be used for faster convergence
- **Scalable**: Supports multiple animal classes with automatic class detection

## 📁 Project Structure

```
animal-classifier/
├── main.py      # main script
├── requirements.txt           # Python dependencies
├── config.py                  # variable values
├── README.md                  # Project documentation
├── data/                      # Dataset storage
│   ├── train/                 # Training images
│   ├── val/                   # Validation images
│   └── test/                  # Test images
├── models/                    # model folder
│   └── model.py               # model definition
├── utils/                     # utils folder
│   ├── classifier.py          # classifier file
│   └── data_splitter.py       # data splitter file
└── outputs/                   # Training results
    ├── training_history.png   # Training curves
    └── logs/                  # Training logs
```

## 🔧 Configuration

### Model Parameters
- **Batch Size**: Adjustable based on GPU memory (default: 32)
- **Learning Rate**: Configurable with scheduler support (default: 0.001)
- **Epochs**: Customizable training duration (default: 25)
- **Device**: Automatic GPU/CPU detection with manual override

### Data Parameters
- **Image Size**: 224×224 pixels (VGG19 standard)
- **Data Split**: 70% train, 20% validation, 10% test
- **Augmentation**: Configurable transformation pipeline

## 🎓 Use Cases

- **Wildlife Conservation**: Automated species identification from camera traps
- **Educational Tools**: Interactive learning applications for biology students
- **Research Applications**: Large-scale biodiversity surveys and monitoring
- **Pet Recognition**: Domestic animal breed classification
- **Zoological Studies**: Animal behavior and population research

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Adding more animal classes and datasets
- Implementing additional CNN architectures
- Enhancing data augmentation techniques
- Optimizing training performance
- Adding web interface for easy usage

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **VGG Team** for the foundational VGG architecture
- **PyTorch Community** for the excellent deep learning framework
- **ImageNet** for providing normalization standards
- **Wildlife Datasets** contributors for training data

---

*Just a basic project aimed at exploring CNNs 🔭*
