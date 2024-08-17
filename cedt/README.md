# Project: Distributed Transformer Model Training with CUDA Optimization and Debugging
## Objective:	       
Develop a simple transf	ormer model for a text classification task, optimize its performance using CUDA, and analyze/debug the performance using Nsight System, Nsight Compute, CUDA-GDB, Memcheck, and SMI. Expand the project to distributed training across multiple GPUs.

## Steps:
1. Set Up the Environment:

Install CUDA, PyTorch with CUDA support, Nsight System, Nsight Compute, CUDA-GDB, Memcheck, and SMI tools.
Set up a basic transformer model using PyTorch for a text classification task.

2. Implement CUDA Optimizations:

Optimize data loading and preprocessing using CUDA.
Implement custom CUDA kernels for parts of the model (e.g., attention mechanism).

3. Training the Transformer Model:

Train the transformer model on a single GPU.
Use Nsight System and Nsight Compute to profile the training process and identify bottlenecks.
Apply optimizations based on the profiling results.

4. Debugging and Error Checking:

Use CUDA-GDB to debug any CUDA-related errors in the custom kernels.
Use Memcheck to check for memory errors and leaks in the CUDA code.

5. Monitoring GPU Usage:

Use SMI tools to monitor GPU utilization, memory usage, and temperature during training.

6. Expanding to Distributed Training:

Modify the training script to use PyTorch's distributed data parallel (DDP) for multi-GPU training.
Ensure the optimized CUDA kernels and data preprocessing work efficiently in a distributed environment.

7. Evaluation and Analysis:

Evaluate the model's performance on a test dataset.
Analyze the scalability and performance improvements with distributed training.
Use Nsight tools to profile and analyze the distributed training process.

## Tools and Techniques:
CUDA: For optimizing data preprocessing and implementing custom kernels.
PyTorch: For building and training the transformer model.
Nsight System and Nsight Compute: For profiling and performance analysis.
CUDA-GDB: For debugging CUDA code.
Memcheck: For memory error checking.
SMI Tools: For monitoring GPU usage.
PyTorch DDP: For distributed training across multiple GPUs.
## Potential for Expansion:
Implement advanced optimization techniques like mixed precision training using NVIDIA's Apex library.
Explore more complex transformer architectures and larger datasets.
Scale the project to multiple nodes in a cluster using distributed computing frameworks like Horovod.

## Run the code
install GPU driver and CUDA Toolkit
```
sudo apt update
ubuntu-drivers devices
apt install ubuntu-drivers-common
```
Find the driver version from https://www.nvidia.com/download/index.aspx#
```
sudo apt install nvidia-driver-version
```
Find the CUDA version from https://developer.nvidia.com/cuda-downloads
```
apt install nvidia-cuda-toolkit //cudatoolkit=11.5
```
Disable Nouveau (If Necessary)
```
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```
add
```
blacklist nouveau
options nouveau modeset=0
```
```
sudo update-initramfs -u
sudo reboot
```
Install pytorch
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html //pytorch-cuda11.3
```
install the libraries
```
sudo apt update
sudo apt install python3-pip
pip3 install transformers
pip3 install numpy==1.21.0
pip3 install pandas==1.3.5
```
install and configure  GCC
```
sudo apt install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/local/cuda
ln -s /usr/bin/gcc-10 $CUDA_ROOT/bin/gcc
ln -s /usr/bin/g++-10 $CUDA_ROOT/bin/g++
```
Build the CUDA Extension:
```
python3 setup.py build_ext --inplace
```
