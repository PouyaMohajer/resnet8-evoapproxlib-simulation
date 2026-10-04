# Setup Guide for ResNet-8 with EvoApproxLib and tf-approximate

## Prerequisites

- Python 3.6+
- NVIDIA GPU with CUDA support (recommended for GPU acceleration)
- CUDA Toolkit 10.0+ (for GPU support)
- cuDNN 7.0+ (for GPU support)

## Step-by-Step Installation

### 1. Clone Repositories

```bash
# Clone EvoApproxLib
git clone https://github.com/ehw-fit/evoapproxlib.git
cd evoapproxlib

# Clone tf-approximate
git clone https://github.com/ehw-fit/tf-approximate.git
```

### 2. Install EvoApproxLib

```bash
cd evoapproxlib

# Install Cython
pip install cython

# Generate Cython sources
python3 make_cython.py

# Compile and install
cd cython
python3 setup.py build_ext
python3 setup.py install --user

# Verify installation
python3 -c "import evoapproxlib as eal; print('EvoApproxLib imported successfully')"
cd ../..
```

### 3. Install TensorFlow with GPU Support

```bash
# For TensorFlow 2.1.0+ with GPU
pip install tensorflow-gpu>=2.1.0

# Or CPU-only (if GPU not available)
pip install tensorflow>=2.1.0

# Verify TensorFlow
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')"
```

### 4. Build tf-approximate

```bash
cd tf-approximate/tf2

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DTFAPPROX_CUDA_ARCHS="75"
# Replace "75" with your GPU compute capability:
#   - RTX 2080/2070: 75
#   - RTX 3090/3080: 86
#   - V100: 70
#   - A100: 80

# Build
make -j$(nproc)

# The built library will be at: build/libApproxGPUOpsTF.so
```

### 5. Configure Library Paths

```bash
# Add to ~/.bashrc or ~/.zshrc
export PYTHONPATH=/path/to/tf-approximate/tf2/python:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/tf-approximate/tf2/build:$LD_LIBRARY_PATH

# Apply changes
source ~/.bashrc
```

### 6. Install Project Dependencies

```bash
cd resnet8-evoapproxlib-simulation
pip install -r requirements.txt
```

### 7. Update Paths in Code

Edit `resnet8_model.py` and update the path:

```python
# Change this line to your actual path
sys.path.insert(0, '/path/to/tf-approximate/tf2/python')
```

## Verification

```bash
# Test EvoApproxLib
python3 -c "
import evoapproxlib as eal
mul = eal.mul8u_L40
print(f'mul8u_L40 MAE: {mul.MAE}')
print(f'mul8u_L40 WCE: {mul.WCE}')
print(f'Test: 128 * 64 = {mul.calc(128, 64)} (expected ~8192)')
"

# Test tf-approximate
python3 -c "
import tensorflow as tf
try:
    approx_op_module = tf.load_op_library('/path/to/tf-approximate/tf2/build/libApproxGPUOpsTF.so')
    print('✓ tf-approximate library loaded successfully')
except Exception as e:
    print(f'✗ Error loading tf-approximate: {e}')
"

# Test ResNet-8 import
python3 -c "
from resnet8_model import build_resnet8_accurate
model = build_resnet8_accurate()
print('✓ ResNet-8 model built successfully')
model.summary()
"
```

## Troubleshooting

### Problem: "libApproxGPUOpsTF.so not found"

**Solution:**
```bash
# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/tf-approximate/tf2/build:$LD_LIBRARY_PATH

# Or copy the library to a standard location
cp /path/to/tf-approximate/tf2/build/libApproxGPUOpsTF.so /usr/local/lib/
ldconfig
```

### Problem: "ImportError: No module named 'keras.layers.fake_approx_convolutional'"

**Solution:**
```bash
# Ensure PYTHONPATH includes tf-approximate
export PYTHONPATH=/path/to/tf-approximate/tf2/python:$PYTHONPATH
```

### Problem: CUDA compatibility issues

**Solution:**
```bash
# Check CUDA version
nvcc --version

# Check cuDNN version
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

# Ensure TensorFlow is built for your CUDA version
pip install tensorflow-gpu==2.1.0  # or your specific version
```

### Problem: GPU not recognized by TensorFlow

**Solution:**
```bash
# Verify GPU
tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# If empty, reinstall TensorFlow GPU version
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow-gpu>=2.1.0
```

## Running Examples

```bash
# Train accurate model (requires CIFAR-10 download)
python3 resnet8_train_accurate.py \
  --output_dir ./checkpoints \
  --epochs 100 \
  --batch_size 128

# Evaluate with approximate multiplier
python3 resnet8_eval_approx.py \
  --checkpoint ./checkpoints/resnet8_weights.h5 \
  --multiplier mul8u_L40 \
  --multiplier_file ./tf2/examples/axmul_8x8/mul8u_L40.bin

# Compare all multipliers
python3 compare_multipliers.py \
  --checkpoint ./checkpoints/resnet8_weights.h5 \
  --multiplier_dir ./tf2/examples/axmul_8x8
```

## References

- [EvoApproxLib GitHub](https://github.com/ehw-fit/evoapproxlib)
- [tf-approximate GitHub](https://github.com/ehw-fit/tf-approximate)
- [TensorFlow GPU Setup Guide](https://www.tensorflow.org/install/gpu)
