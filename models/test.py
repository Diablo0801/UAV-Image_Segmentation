import torch
import tensorflow as tf
print("PyTorch version:", torch.__version__)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
else:
    print("CUDA is NOT available :(")




print(tf.config.list_physical_devices())
print(tf.config.list_physical_devices("GPU"))
print(tf.config.list_physical_devices("CPU"))
