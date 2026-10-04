##========== Copyright (c) 2025 =========##
## Compare Accuracy Across Different EvoApproxLib Multipliers
##================================================##

import tensorflow as tf
import argparse
import os
import glob
from resnet8_model import build_resnet8_approximate


def load_cifar10_test():
    """
    Load and preprocess CIFAR-10 test set.
    """
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.flatten()
    return x_test, y_test


def evaluate_with_multiplier(weights_path, mul_file, mul_name):
    """
    Evaluate model with a specific multiplier.
    """
    x_test, y_test = load_cifar10_test()
    
    # Build model
    model = build_resnet8_approximate(
        input_shape=(32, 32, 3),
        num_classes=10,
        num_channels=64,
        mul_map_file=mul_file
    )
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Load weights
    model.load_weights(weights_path)
    
    # Evaluate
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Compare Multipliers Performance')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained weights')
    parser.add_argument('--multiplier_dir', type=str, default='./tf2/examples/axmul_8x8',
                        help='Directory containing multiplier binary files')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Verify checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Verify multiplier directory
    if not os.path.isdir(args.multiplier_dir):
        print(f"ERROR: Multiplier directory not found: {args.multiplier_dir}")
        return
    
    # Find all multiplier files
    mul_files = sorted(glob.glob(os.path.join(args.multiplier_dir, 'mul8u_*.bin')))
    
    if not mul_files:
        print(f"ERROR: No multiplier files found in {args.multiplier_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"ResNet-8 Multiplier Comparison")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Multiplier Directory: {args.multiplier_dir}")
    print(f"Number of multipliers: {len(mul_files)}")
    print(f"{'='*70}\n")
    
    # Evaluate each multiplier
    results = []
    
    for i, mul_file in enumerate(mul_files, 1):
        mul_name = os.path.basename(mul_file).replace('.bin', '')
        
        print(f"[{i:2d}/{len(mul_files)}] Evaluating {mul_name}...", end=' ', flush=True)
        
        try:
            accuracy = evaluate_with_multiplier(args.checkpoint, mul_file, mul_name)
            results.append((mul_name, accuracy))
            print(f"✓ {accuracy*100:.2f}%")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
    
    # Sort results by accuracy (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Results Summary (sorted by accuracy)")
    print(f"{'='*70}")
    print(f"\n{'Multiplier':<20} {'Accuracy':<15} {'Difference from Best':<20}")
    print(f"{'-'*70}")
    
    best_accuracy = results[0][1]
    
    for mul_name, accuracy in results:
        diff = accuracy - best_accuracy
        diff_pct = diff * 100
        marker = ' (BEST)' if diff == 0 else ''
        print(f"{mul_name:<20} {accuracy*100:>6.2f}%         {diff_pct:>+6.2f}%{marker}")
    
    print(f"\n{'='*70}")
    print(f"\nBest Multiplier: {results[0][0]} ({results[0][1]*100:.2f}%)")
    print(f"Worst Multiplier: {results[-1][0]} ({results[-1][1]*100:.2f}%)")
    print(f"Accuracy Range: {(results[0][1]-results[-1][1])*100:.2f}%")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
