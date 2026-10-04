##========== Copyright (c) 2025 =========##
## Train Accurate ResNet-8 on CIFAR-10
##================================================##

import tensorflow as tf
import argparse
import os
from resnet8_model import build_resnet8_accurate


def load_cifar10_data():
    """
    Load and preprocess CIFAR-10 dataset.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Test set: {x_test.shape}, {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def train_model(model, train_data, val_data, epochs=100, batch_size=128, output_dir='./checkpoints'):
    """
    Train the accurate ResNet-8 model.
    """
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'resnet8_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=0
        )
    ]
    
    # Train
    print("\n=== Training Accurate ResNet-8 ===")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final weights
    weights_path = os.path.join(output_dir, 'resnet8_weights.h5')
    model.save_weights(weights_path)
    print(f"\nWeights saved to: {weights_path}")
    
    return model, history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate model on test set.
    """
    print("\n=== Evaluating Accurate Model ===")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Accurate ResNet-8 on CIFAR-10')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_channels', type=int, default=64,
                        help='Number of base channels')
    
    args = parser.parse_args()
    
    # Load data
    train_data, (x_test, y_test) = load_cifar10_data()
    x_train, y_train = train_data
    
    # Split validation
    split = int(0.9 * len(x_train))
    val_data = (x_train[split:], y_train[split:])
    train_data = (x_train[:split], y_train[:split])
    
    # Build model
    model = build_resnet8_accurate(
        input_shape=(32, 32, 3),
        num_classes=10,
        num_channels=args.num_channels
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Train
    model, history = train_model(
        model,
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # Evaluate
    accuracy = evaluate_model(model, x_test, y_test)
    
    print(f"\n" + "="*50)
    print(f"Training completed!")
    print(f"Accurate Model Test Accuracy: {accuracy*100:.2f}%")
    print(f"Checkpoint saved to: {args.output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()
