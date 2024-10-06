import torch
from transformers import pipeline
import time


# Run inference in parallel using CUDA streams
start_time = time.time()

# Assuming a GPU with device=0, and we will use two CUDA streams
device = torch.device("cuda:0")

# Create the streams for data parallelism
num_streams = 2
streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

# Create a Hugging Face pipeline for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

# Number of samples you want to process
num_samples = 1000
texts = ["I love using Hugging Face's Transformers library!" for _ in range(num_samples)]

# Manually split the texts into chunks for each stream
split_texts = [texts[i::num_streams] for i in range(num_streams)]

# A function to process each chunk within a stream
def process_chunk(chunk, stream):
    with torch.cuda.stream(stream):  # Assign the work to the given stream
        results = sentiment_pipeline(chunk)  # Perform inference on this chunk
        torch.cuda.synchronize()  # Ensure the stream is complete
    return results


results = []

# Launch inference on each chunk using a separate stream
for i in range(num_streams):
    chunk = split_texts[i]
    stream = streams[i]
    result = process_chunk(chunk, stream)
    results.append(result)

# Synchronize the streams to make sure everything is done
torch.cuda.synchronize()

end_time = time.time()

# Calculate the total time taken
total_time = end_time - start_time
print(f"Inference completed in {total_time:.4f} seconds.")

# Combine the results from all streams
final_results = []
for result in results:
    final_results.extend(result)

# Process the final combined results
for res in final_results[:10]:  # Only display first 10 results
    print(res)
