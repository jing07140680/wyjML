from mpi4py import MPI
from transformers import pipeline

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get the rank of the process
size = comm.Get_size()  # Get the total number of processes

# Example number of samples per process
samples_per_process = 500  # Assuming 1000 samples total across 2 nodes
texts = ["I love using Hugging Face's Transformers library!" for _ in range(samples_per_process)]

# Load the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", device=0)

# Run inference
results = sentiment_pipeline(texts)

# Gather results at the root process
all_results = comm.gather(results, root=0)

# If you're the root, combine the results
if rank == 0:
    combined_results = [item for sublist in all_results for item in sublist]  # Flatten the list
    # Output combined results
    for i, result in enumerate(combined_results):
        print(f"Sample {i}: {result}")

# Optional: Benchmarking performance (timing)
if rank == 0:
    import time
    start_time = time.time()

    # Run the distributed inference
    all_results = comm.gather(results, root=0)

    # Measure end time
    end_time = time.time()
    print(f"Total time for inference: {end_time - start_time} seconds")
