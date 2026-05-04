import torch
from lib.neural_net_onecyclelr import NeuralNetwork, ModelTrainer
from lib.data_loader import process_and_save_data, create_data_loaders
#from visualization import visualize_processed_data
import argparse
import os
import logging
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_and_plot_confusion_matrix(model, data_loader, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return cm

def main():
    # ---------------------- ARGUMENT PARSING ----------------------
    parser = argparse.ArgumentParser(description='Train neural network classifier on BME specimen data')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing raw data')
    parser.add_argument('--processed_data_path', type=str, default='processed_data.h5', help='Path to save/load processed data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_sizes', type=str, default='512,256,128', help='Comma-separated list of hidden layer sizes')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of data to use for training')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_save_path', type=str, default='trained_model.pth', help='Path to save trained model')
    parser.add_argument('--load_processed', action='store_true', help='Load preprocessed data if available')
    parser.add_argument('--visualize', action='store_true', help='Generate data visualizations')
    parser.add_argument('--visualization_dir', type=str, default='visualizations', help='Directory to save visualizations')
    args = parser.parse_args()

    # ---------------------- REPRODUCIBILITY ----------------------
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    hidden_sizes = [int(size) for size in args.hidden_sizes.split(',')]

    # ---------------------- DATA LOADING ----------------------
    if args.load_processed and os.path.exists(args.processed_data_path):
        logger.info("Loading existing processed data...")
        train_loader, val_loader, class_names = create_data_loaders(
            data_path=args.processed_data_path,
            batch_size=args.batch_size,
            train_split=args.train_split,
            random_seed=args.random_seed
        )
    else:
        logger.info("Processing raw data...")
        train_loader, val_loader, class_names = process_and_save_data(
            raw_data_dir=args.data_dir,
            output_path=args.processed_data_path,
            batch_size=args.batch_size,
            train_split=args.train_split,
            random_seed=args.random_seed
        )

    ## ---------------------- VISUALIZATION ----------------------
    #if args.visualize:
    #    logger.info("Generating data visualizations...")
    #    visualize_processed_data(args.processed_data_path, args.visualization_dir)

    # ---------------------- DYNAMIC MODEL SETUP ----------------------
    sample_batch = next(iter(train_loader))
    X_sample, y_sample = sample_batch
    input_size = X_sample.shape[1]
    num_classes = len(torch.unique(y_sample))
    output_size = num_classes

    logger.info(f"Detected classification task with {num_classes} classes.")
    logger.info(f"Input size: {input_size}, Output size: {output_size}")

    # ---------------------- MODEL & TRAINER ----------------------
    model = NeuralNetwork(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    trainer = ModelTrainer(model, learning_rate=args.learning_rate)

    # ---------------------- TRAINING ----------------------
    logger.info("Starting training...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # ---------------------- VISUALIZE TRAINING ----------------------
    trainer.plot_losses()

    # ---------------------- CONFUSION MATRIX ----------------------
    logger.info("Evaluating model with confusion matrix...")

    # Create class name list: 0,1,2,... if unknown
    if not class_names:
        class_names = [f"Class {i}" for i in range(output_size)]

    cm = evaluate_and_plot_confusion_matrix(model, val_loader, class_names=class_names)
    logger.info(f"Confusion matrix:\n{cm}")

    # ---------------------- SAVE MODEL ----------------------
    trainer.save_model(args.model_save_path)
    logger.info(f"Training completed. Model saved as '{args.model_save_path}'")

    # ---------------------- PRINT CLASS MAPPING ----------------------
    logger.info("Class Mapping:")
    for i, name in enumerate(class_names):
        logger.info(f"{i} = {name}")

if __name__ == "__main__":
    main()
