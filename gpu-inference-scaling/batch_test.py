import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import time

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to('cuda')
model.eval()  # Set the model to evaluation mode

# Load a dataset
dataset = load_dataset("imdb", split='test[:1000]')

# Function for inference using CUDA streams
def run_inference(texts, stream):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    with torch.no_grad():
        with torch.cuda.stream(stream):
            with torch.cuda.amp.autocast():  # Use mixed precision
                outputs = model(**inputs)
    return outputs.logits

# Measure time for batch processing
batch_size = 16  # Adjusted batch size for demonstration
num_batches = len(dataset) // batch_size
start_time = time.time()

# Create CUDA streams
streams = [torch.cuda.Stream() for _ in range(num_batches)]

# Launch inference in parallel using CUDA streams
results = []
for i in range(num_batches):
    # Each stream is launched to run inference on a batch
    texts = dataset[i * batch_size: (i + 1) * batch_size]['text']
    stream = streams[i]
    results.append(run_inference(texts, stream))

# Synchronize streams and measure the time taken
for stream in streams:
    stream.synchronize()

end_time = time.time()
print(f"Time taken for inference with {len(dataset)} samples: {end_time - start_time:.4f} seconds")

# Results processing (optional)
# You can concatenate the results or process them as needed
all_logits = torch.cat(results, dim=0)
