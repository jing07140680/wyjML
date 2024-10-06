from mpi4py import MPI
import numpy as np

def distributed_benchmark(num_samples=1000):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each process handles a portion of the total samples
    samples_per_process = num_samples // size
    texts = ["I love using Hugging Face's Transformers library!" for _ in range(samples_per_process)]
    
    # Initialize the pipeline only on the first GPU (or the specified device)
    device = rank % torch.cuda.device_count()  # Use modulo to balance loads on multiple GPUs
    sentiment_pipeline = pipeline("sentiment-analysis", device=device)

    # Start the timer
    start_time = time.time()
    
    # Perform inference
    results = sentiment_pipeline(texts)

    # End the timer
    end_time = time.time()
    
    # Calculate the total time taken
    total_time = end_time - start_time
    avg_time_per_sample = total_time / samples_per_process

    # Gather results
    all_results = comm.gather(results, root=0)
    
    if rank == 0:
        print(f"Total samples: {num_samples}, Processes: {size}")
        print(f"Total time taken: {total_time:.4f} seconds")
        print(f"Average time per sample: {avg_time_per_sample:.4f} seconds")

# To run the distributed benchmark, use the command line:
# mpiexec -n <number_of_processes> python your_script.py
