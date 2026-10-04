##========== Copyright (c) 2025 =========##
## Evaluate ResNet-8 with Approximate Multipliers
##================================================##

import tensorflow as tf
import argparse
import os
import sys
from resnet8_model import build_resnet8_approximate

# Ensure tf-approximate is in path
sys.path.insert(0, '/path/to/tf-approximate/tf2/python')


def load_cifar10_test():
    """
    Load and preprocess CIFAR-10 test set.
    """
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.flatten()
    
    return x_test, y_test


def build_and_evaluate_approx_model(
    weights_path,
    mul_map_file='',
    mul_name='accurate',
    batch_size=128
):
    """
    Build approximate model and evaluate with specific multiplier.
    
    Args:
        weights_path: Path to trained weights
        mul_map_file: Path to multiplier binary file
        mul_name: Name of multiplier for display
        batch_size: Batch size for evaluation
    
    Returns:
        Test accuracy
    """
    print(f"\n{'='*60}")
    print(f"Evaluating with: {mul_name}")
    print(f"Multiplier file: {mul_map_file if mul_map_file else 'Accurate (default)'}")
    print(f"{'='*60}")
    
    # Load test data
    x_test, y_test = load_cifar10_test()
    
    # Build approximate model
    model = build_resnet8_approximate(
        input_shape=(32, 32, 3),
        num_classes=10,
        num_channels=64,
        mul_map_file=mul_map_file
    )
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Load weights from trained accurate model
    model.load_weights(weights_path)
    
    # Evaluate
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet-8 with Approximate Multipliers')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained weights file')
    parser.add_argument('--multiplier', type=str, default='mul8u_1JFF',
                        help='Name of EvoApproxLib multiplier (e.g., mul8u_L40, mul8u_2HH)')
    parser.add_argument('--multiplier_file', type=str, default='',
                        help='Path to multiplier binary file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Verify multiplier file exists (if provided)
    if args.multiplier_file and not os.path.exists(args.multiplier_file):
        print(f"ERROR: Multiplier file not found: {args.multiplier_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"ResNet-8 Approximate Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Multiplier: {args.multiplier}")
    print(f"Multiplier file: {args.multiplier_file if args.multiplier_file else 'Not provided (using accurate)'}")
    
    # Evaluate with specified multiplier
    accuracy = build_and_evaluate_approx_model(
        weights_path=args.checkpoint,
        mul_map_file=args.multiplier_file,
        mul_name=args.multiplier,
        batch_size=args.batch_size
    )
    
    print(f"\nFinal Accuracy with {args.multiplier}: {accuracy*100:.2f}%")


if __name__ == '__main__':
    main()
