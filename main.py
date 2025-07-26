import torch


from utils.classifier import AnimalClassifier
from utils.data_splitter import DatasetSplitter

from config import BATCH_SIZE, LEARNING_RATE, INFERENCE_IMG_PATH, ORIGINAL_DATASET_DIR, SPLIT_DATASET_DIR, SPLIT



if __name__ == "__main__":
    # Check CUDA availability first
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 30)
    
    # Step 1: Split the dataset
    print("Step 1: Splitting dataset...")
    if SPLIT:
        splitter = DatasetSplitter(ORIGINAL_DATASET_DIR, SPLIT_DATASET_DIR)
        splitter.split_dataset()  # Uncomment to split your dataset
    else:
        SPLIT_DATASET_DIR = ORIGINAL_DATASET_DIR

    # Step 2: Train the classifier
    print("\nStep 2: Training classifier...")
    # You can force CPU by setting device=torch.device('cpu')
    classifier = AnimalClassifier(SPLIT_DATASET_DIR, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    
    # Print model architecture
    print("\nVGG19 Model Architecture:")
    print(classifier.model)
    
    # Train the model
    train_losses, train_accs, val_losses, val_accs = classifier.train(num_epochs=1)
    
    # Test the model
    test_loss, test_acc = classifier.test()
    
    # Example prediction
    if INFERENCE_IMG_PATH:
        predicted_class, confidence = classifier.predict(INFERENCE_IMG_PATH)
        print(f"Predicted: {predicted_class} (Confidence: {confidence:.3f})")
