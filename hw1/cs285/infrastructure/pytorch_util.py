import torch

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if use_gpu and cuda_available:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    elif use_gpu and mps_available:
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("GPU/MPS not detected. Defaulting to CPU.")


def set_device(gpu_id):
    global device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda:" + str(gpu_id))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
