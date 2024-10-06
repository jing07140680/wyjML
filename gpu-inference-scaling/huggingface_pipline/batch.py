import torch
from transformers import pipeline
import time

# Number of CUDA streams
num_streams = 2
device = torch.device("cuda:0")
streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

# Create the sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

# Number of samples and batch size
num_samples = 1000
texts = ["I love using Hugging Face's Transformers library!" for _ in range(num_samples)]
batch_size = num_samples // num_streams

# Function to process each chunk
def process_batch(batch, stream):
    with torch.cuda.stream(stream):
        sentiment_pipeline(batch)
        torch.cuda.synchronize()

# Split the data and run on different streams
start_time = time.time()
for i in range(num_streams):
    batch = texts[i * batch_size:(i + 1) * batch_size]
    process_batch(batch, streams[i])

# Wait for all streams to finish
torch.cuda.synchronize()

end_time = time.time()
total_time = end_time - start_time
print(f"Inference completed in {total_time:.4f} seconds.")
