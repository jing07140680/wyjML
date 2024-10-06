import time
from transformers import pipeline

def benchmark_sentiment_analysis(model_name="sentiment-analysis", device=0, num_samples=1000):
    # Load the sentiment analysis pipeline
    sentiment_pipeline = pipeline(model_name, device=device)
    
    # Generate sample input texts
    texts = ["I love using Hugging Face's Transformers library!" for _ in range(num_samples)]
    
    # Start the timer
    start_time = time.time()
    
    # Perform inference
    results = sentiment_pipeline(texts)
    
    # End the timer
    end_time = time.time()
    
    # Calculate the total time taken
    total_time = end_time - start_time
    avg_time_per_sample = total_time / num_samples
    
    # Print results
    print(f"Device: {device}")
    print(f"Total time for {num_samples} samples: {total_time:.4f} seconds")
    print(f"Average time per sample: {avg_time_per_sample:.4f} seconds")
    return results
# Run benchmark on a single GPU (device 0)
print("Benchmarking on a single GPU:")
results_single_gpu = benchmark_sentiment_analysis(device=0)
