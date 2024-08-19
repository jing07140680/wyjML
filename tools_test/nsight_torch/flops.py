import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

def flop_ops():
    SIZE = 2**12
    ones = torch.ones((SIZE, SIZE), device=torch.device('cuda:0'))

    torch.matmul(ones, ones, out=ones)   # matrix multiplication
    ones.mul_(0.5)                       # in place multiplication
    result_mul = ones.mul(0.5)           # out of place multiplication
    total = ones + result_mul            # adding tensors
    result_sum = torch.sum(ones)         # adding elements of a tensor
    result_sqrt = torch.sqrt(ones)       # sqrt takes 7 ops
    result_sin = torch.sin(ones)         # sin takes 17 ops (14 fp64, 3 fp32)
    result_sigmoid = torch.sigmoid(ones) # sigmoid takes 24 ops
    result = torch.log10(ones)           # log10 takes 24 ops
    result = torch.pow(ones, 3.14159)    # pow takes 142 ops


flop_ops()

trace_handler = tensorboard_trace_handler(dir_name="./flops_trace", use_gzip=True)
with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
     on_trace_ready = trace_handler,
     record_shapes = True,
     with_stack = True
) as prof:
    flop_ops()
