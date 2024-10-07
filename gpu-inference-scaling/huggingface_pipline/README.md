batch: By grouping input data into batches, you can represent the entire batch as a matrix (or tensor) and perform matrix operations like matrix multiplication across all the inputs in the batch at once, which is much faster than processing them sequentially.

cudastream(batch+stream): Instead of processing one input at a time (which can be inefficient), a batch of inputs is processed together in a stream which allows for overlapping computations with memory transfers (using asynchronous operations), taking advantage of parallelism in GPUs.

distributed:data parallelisum using two nodes, each with one GPU.
when using mpi+nccl for distributed training/inference, need to make sure ssh on all the nodes work.
1. generate ssh key.
2. copy pub key to all the other nodes' authorized_key file

Note: during the test, the GPU doesn't support Nsight-compute. as for Nsight-system, not sure why distributed.py doesn't work when using  mpi with hostfile 