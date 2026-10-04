# Approximate ResNet-8 Network using EvoApproxLib and TF-Approximate

This project implements a ResNet-8 neural network architecture using:
- **TensorFlow 2** (GPU-accelerated with tf-approximate)
- **FakeApproxConv2D** layers from tf-approximate repository
- **EvoApproxLib** unsigned 8×8 bit approximate multipliers

## Overview

The framework enables simulation of approximate hardware accelerators by:
1. Using fake quantization (8-bit fixed-point) for inputs and weights
2. Replacing precise multiplication with approximate 8×8 unsigned multipliers from EvoApproxLib
3. Computing gradients with accurate operations (for fine-tuning)

## Architecture

ResNet-8 contains:
- Initial conv layer (3→64 channels, 3×3 kernels)
- 3 residual blocks, each with 2 approximate conv layers
- Global average pooling + dense classifier

## Available EvoApproxLib Multipliers

The project supports all 8×8 unsigned multipliers from EvoApproxLib:
- **mul8u_1JFF** (accurate reference - no approximation)
- **mul8u_L40** (low error, moderate savings)
- **mul8u_2HH**, **mul8u_DM1**, **mul8u_GS2**, **mul8u_YX7**, etc.

## Files

- `resnet8_train_accurate.py` - Train accurate ResNet-8 on CIFAR-10
- `resnet8_eval_approx.py` - Evaluate with approximate multipliers
- `resnet8_model.py` - ResNet-8 model definition
- `compare_multipliers.py` - Compare accuracy across different multipliers
- `requirements.txt` - Dependencies

## Setup

### Prerequisites

1. **TensorFlow 2.1.0+** with GPU support
2. **EvoApproxLib** installed and compiled
3. **tf-approximate** library (libApproxGPUOpsTF.so)

### Installation

```bash
# Install EvoApproxLib
git clone https://github.com/ehw-fit/evoapproxlib.git
cd evoapproxlib
pip install cython
python3 make_cython.py
cd cython
python3 setup.py build_ext
python3 setup.py install --user

# Install TensorFlow with GPU
pip install tensorflow-gpu>=2.1.0

# Clone or download tf-approximate library
# Ensure libApproxGPUOpsTF.so is in LD_LIBRARY_PATH
```

## Usage

### 1. Train Accurate Model

```bash
python resnet8_train_accurate.py --output_dir ./checkpoints
```

### 2. Evaluate with Approximate Multiplier

```bash
python resnet8_eval_approx.py \
  --checkpoint ./checkpoints/resnet8_weights.h5 \
  --multiplier mul8u_L40 \
  --multiplier_file ./tf2/examples/axmul_8x8/mul8u_L40.bin
```

### 3. Compare Multiple Multipliers

```bash
python compare_multipliers.py \
  --checkpoint ./checkpoints/resnet8_weights.h5 \
  --multiplier_dir ./tf2/examples/axmul_8x8
```

## Example Output

```
Accurate Model Accuracy: 92.5%

Approximate Results:
  mul8u_1JFF (accurate):  92.5%
  mul8u_L40:              91.8%  (0.7% loss)
  mul8u_2HH:              90.5%  (2.0% loss)
  mul8u_DM1:              89.2%  (3.3% loss)
```

## References

- **EvoApproxLib**: Mrazek et al., "EvoApprox8b: Library of approximate adders and multipliers for circuit design and benchmarking of approximation methods", DATE 2017
- **tf-approximate**: Vaverka et al., "TFApprox: Towards a Fast Emulation of DNN Approximate Hardware Accelerators on GPU", DATE 2020
- **ALWANN**: Mrazek et al., "ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators without Retraining", ICCAD 2019

## License

MIT
