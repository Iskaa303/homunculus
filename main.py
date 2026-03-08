import torch
import time

def check_torch_environment():
    print("--- PyTorch Environment Report ---")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {cuda_available}")
    
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Reserved:  {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("Running on CPU (No GPU detected or 'cpu' extra installed).")
    
    print("-" * 34)
    return device

def run_sample_operation(device):
    print(f"\nPerforming sample operation on {device}...")
    
    # Create two large random matrices directly on the target device
    # This avoids the overhead of creating on CPU and moving to GPU
    size = 2000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Matrix Multiplication (matmul)
    start_time = time.time()
    result = torch.matmul(a, b)
    
    # Synchronize if using CUDA to get accurate timing
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    print(f"Result shape: {result.shape}")
    print(f"Operation took: {end_time - start_time:.4f} seconds")

def main():
    current_device = check_torch_environment()
    run_sample_operation(current_device)


if __name__ == "__main__":
    main()
