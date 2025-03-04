import torch
import time

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Create large tensors
size = 30000
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

print(f"Performing {size}x{size} matrix multiplication...")
start_time = time.time()

# Perform matrix multiplication
c = torch.mm(a, b)

end_time = time.time()
print(f"Matrix multiplication completed in {end_time - start_time:.2f} seconds")
