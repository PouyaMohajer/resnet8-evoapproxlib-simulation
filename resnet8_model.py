##========== Copyright (c) 2025 =========##
## ResNet-8 Model with Approximate Convolutions
##================================================##

import tensorflow as tf
import sys
import os

# Import FakeApproxConv2D from tf-approximate framework
# Adjust path to your tf-approximate installation
sys.path.insert(0, '/path/to/tf-approximate/tf2/python')
from keras.layers.fake_approx_convolutional import FakeApproxConv2D


def build_resnet8_accurate(input_shape=(32, 32, 3), num_classes=10, num_channels=64):
    """
    Build accurate ResNet-8 with standard Conv2D layers.
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of classification classes
        num_channels: Base number of channels
    
    Returns:
        Keras Sequential model
    """
    model = tf.keras.Sequential([
        # Initial conv layer
        tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same',
            input_shape=input_shape, name='conv_init'
        ),
        tf.keras.layers.BatchNormalization(name='bn_init'),
        tf.keras.layers.Activation('relu', name='relu_init'),
        
        # Residual block 1
        tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same', name='conv_b1_1'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b1_1'),
        tf.keras.layers.Activation('relu', name='relu_b1_1'),
        tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same', name='conv_b1_2'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b1_2'),
        tf.keras.layers.Activation('relu', name='relu_b1_2'),
        
        # Residual block 2
        tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same', name='conv_b2_1'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b2_1'),
        tf.keras.layers.Activation('relu', name='relu_b2_1'),
        tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same', name='conv_b2_2'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b2_2'),
        tf.keras.layers.Activation('relu', name='relu_b2_2'),
        
        # Residual block 3
        tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same', name='conv_b3_1'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b3_1'),
        tf.keras.layers.Activation('relu', name='relu_b3_1'),
        tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same', name='conv_b3_2'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b3_2'),
        tf.keras.layers.Activation('relu', name='relu_b3_2'),
        
        # Global average pooling + classifier
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')
    ])
    
    return model


def build_resnet8_approximate(
    input_shape=(32, 32, 3), 
    num_classes=10, 
    num_channels=64,
    mul_map_file=''
):
    """
    Build ResNet-8 with FakeApproxConv2D layers using EvoApproxLib multipliers.
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of classification classes
        num_channels: Base number of channels
        mul_map_file: Path to EvoApproxLib multiplier binary file (e.g., mul8u_L40.bin)
                      If empty, uses accurate 8x8 multiplication
    
    Returns:
        Keras Sequential model
    """
    model = tf.keras.Sequential([
        # Initial approximate conv layer
        FakeApproxConv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same',
            input_shape=input_shape, mul_map_file=mul_map_file,
            num_bits=8, name='approx_conv_init'
        ),
        tf.keras.layers.BatchNormalization(name='bn_init'),
        tf.keras.layers.Activation('relu', name='relu_init'),
        
        # Residual block 1 - both convs are approximate
        FakeApproxConv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same',
            mul_map_file=mul_map_file, num_bits=8, name='approx_conv_b1_1'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b1_1'),
        tf.keras.layers.Activation('relu', name='relu_b1_1'),
        FakeApproxConv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same',
            mul_map_file=mul_map_file, num_bits=8, name='approx_conv_b1_2'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b1_2'),
        tf.keras.layers.Activation('relu', name='relu_b1_2'),
        
        # Residual block 2
        FakeApproxConv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same',
            mul_map_file=mul_map_file, num_bits=8, name='approx_conv_b2_1'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b2_1'),
        tf.keras.layers.Activation('relu', name='relu_b2_1'),
        FakeApproxConv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same',
            mul_map_file=mul_map_file, num_bits=8, name='approx_conv_b2_2'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b2_2'),
        tf.keras.layers.Activation('relu', name='relu_b2_2'),
        
        # Residual block 3
        FakeApproxConv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same',
            mul_map_file=mul_map_file, num_bits=8, name='approx_conv_b3_1'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b3_1'),
        tf.keras.layers.Activation('relu', name='relu_b3_1'),
        FakeApproxConv2D(
            filters=num_channels, kernel_size=(3, 3),
            strides=(1, 1), padding='same',
            mul_map_file=mul_map_file, num_bits=8, name='approx_conv_b3_2'
        ),
        tf.keras.layers.BatchNormalization(name='bn_b3_2'),
        tf.keras.layers.Activation('relu', name='relu_b3_2'),
        
        # Global average pooling + classifier (accurate)
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')
    ])
    
    return model


class ResidualBlock(tf.keras.layers.Layer):
    """
    Reusable residual block with approximate convolutions.
    """
    
    def __init__(self, filters, kernel_size=3, stride=1, mul_map_file='', num_bits=8, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.mul_map_file = mul_map_file
        self.num_bits = num_bits
        
        # First approximate conv
        self.conv1 = FakeApproxConv2D(
            filters=filters, kernel_size=kernel_size,
            strides=stride, padding='same',
            mul_map_file=mul_map_file, num_bits=num_bits
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')
        
        # Second approximate conv
        self.conv2 = FakeApproxConv2D(
            filters=filters, kernel_size=kernel_size,
            strides=1, padding='same',
            mul_map_file=mul_map_file, num_bits=num_bits
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        # Skip connection projection if stride > 1 or channels change
        if stride > 1 or filters != filters:
            self.shortcut = tf.keras.Sequential([
                FakeApproxConv2D(
                    filters=filters, kernel_size=1,
                    strides=stride, padding='same',
                    mul_map_file=mul_map_file, num_bits=num_bits
                ),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.shortcut = None
        
        self.relu_out = tf.keras.layers.Activation('relu')
    
    def call(self, inputs, training=None):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        if self.shortcut is not None:
            skip = self.shortcut(inputs)
        else:
            skip = inputs
        
        # Residual addition
        x = tf.keras.layers.Add()([x, skip])
        x = self.relu_out(x)
        
        return x


def build_resnet8_modular(
    input_shape=(32, 32, 3),
    num_classes=10,
    num_channels=64,
    mul_map_file=''
):
    """
    Build ResNet-8 using ResidualBlock modules.
    More flexible for architectural modifications.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial conv
    x = FakeApproxConv2D(
        filters=num_channels, kernel_size=(3, 3),
        strides=(1, 1), padding='same',
        mul_map_file=mul_map_file, num_bits=8,
        name='stem_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.Activation('relu', name='stem_relu')(x)
    
    # Three residual blocks
    x = ResidualBlock(
        filters=num_channels, kernel_size=3, stride=1,
        mul_map_file=mul_map_file, num_bits=8,
        name='residual_block_1'
    )(x)
    
    x = ResidualBlock(
        filters=num_channels, kernel_size=3, stride=1,
        mul_map_file=mul_map_file, num_bits=8,
        name='residual_block_2'
    )(x)
    
    x = ResidualBlock(
        filters=num_channels, kernel_size=3, stride=1,
        mul_map_file=mul_map_file, num_bits=8,
        name='residual_block_3'
    )(x)
    
    # Global average pooling + classifier
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet8_Approximate')
    
    return model
