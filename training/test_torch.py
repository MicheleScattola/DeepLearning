import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA Available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    
    # Test a live mathematical operation on the GPU matrix
    x = torch.rand(5, 5).cuda()
    y = torch.rand(5, 5).cuda()
    z = torch.mm(x, y)
    print("GPU Matrix Multiplication Test: SUCCESS!")
else:
    print("ERROR: PyTorch cannot see your GPU!")