import time
from transformers import pipeline
import torch

# Run inference in batches without streams
start_time = time.time()

# Assuming a GPU with device=0
device = torch.device("cuda:0")

# Create a Hugging Face pipeline for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

# Number of samples you want to process
num_samples = 1000
texts = ["I love using Hugging Face's Transformers library!" for _ in range(num_samples)]

# Split the texts into 2 batches
batch_size = num_samples // 2
batch1 = texts[:batch_size]
batch2 = texts[batch_size:]

# Perform inference on the first batch
results_batch1 = sentiment_pipeline(batch1)

# Perform inference on the second batch
results_batch2 = sentiment_pipeline(batch2)

# Combine the results
results = results_batch1 + results_batch2

# Calculate the total time taken
end_time = time.time()
total_time = end_time - start_time
print(f"Inference completed in {total_time:.4f} seconds.")

