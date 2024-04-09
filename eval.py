import requests
import os
import time
import random
import matplotlib.pyplot as plt
from google.cloud import storage

# Configuration
GCS_BUCKET = 'ds756data'
SERVERFUL_URL = 'http://34.125.66.221:8000/detect'
LOCAL_IMAGE_DIR = 'data'
BATCH_SIZES = [100, 200, 500, 750, 1000]
NUM_BATCHES = {100: 5, 200: 5, 500: 3, 750: 2, 1000: 2}
#BATCH_SIZES = [10]
#NUM_BATCHES = {10: 5}

# Initialize GCS client
storage_client = storage.Client(project='driven-tenure-419701')
bucket = storage_client.get_bucket(GCS_BUCKET)

def send_images_serverful(image_paths):
    #Send images to the serverful endpoint via POST requests.
    start_time = time.time()
    for image_path in image_paths:
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(SERVERFUL_URL, files=files)
            if response.status_code != 200:
                print(f"Failed to upload {image_path} - Response code {response.status_code}")
            #else:
                #print(f"Uploaded {image_path}")
    return time.time() - start_time

def send_images_serverless(image_paths, bucket):
    #Upload a list of image paths to GCS, simulating serverless trigger.
    start_time = time.time()
    for image_path in image_paths:
        blob = bucket.blob(os.path.basename(image_path))
        try:
            blob.upload_from_filename(image_path)
            #print(f"Uploaded {image_path} to gs://{bucket}/{image_path}")
        except Exception as e:
            print(f"Failed to upload {image_path}: {e}")
        
    return time.time() - start_time

def select_image_paths(directory, num_images):
    # List all files in the directory
    all_files = [os.path.join(os.getcwd(), directory, f) for f in os.listdir(directory)]
    
    # Filter out files that are not images
    image_files = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(('.jpg'))]
    
    # Ensure the directory has enough images
    if len(image_files) < num_images:
        raise ValueError(f"Requested {num_images} images, but only found {len(image_files)} available.")

    # Randomly select the requested number of image paths
    selected_paths = random.sample(image_files, num_images)
    
    return selected_paths

def plot_results(results):
    sizes = [result[0] for result in results]
    sl_times = [result[1] for result in results]
    sf_times = [result[2] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, sl_times, label='Serverless', marker='o')
    plt.plot(sizes, sf_times, label='Serverful', marker='x')
    plt.xlabel('Batch Size')
    plt.ylabel('Time Taken (seconds)')
    plt.title('Performance Comparison: Serverless vs. Serverful')
    plt.legend()
    plt.savefig('Upload Performance Comparison')
    plt.show()
    
def evaluate_performance():
    results = []
    for batch_size in BATCH_SIZES:
        for batch in range(NUM_BATCHES[batch_size]):
            # Select or prepare a batch of image paths
            image_paths = select_image_paths(LOCAL_IMAGE_DIR, batch_size)
            # Serverful upload
            sf_duration = send_images_serverful(image_paths)
            # Serverless upload
            sl_duration = send_images_serverless(image_paths, bucket)      
            print(f"Batch size: {batch_size}, Serverless: {sl_duration}s, Serverful: {sf_duration}s")
            results.append((batch_size, sf_duration, sl_duration))
    plot_results(results)

# Main execution
if __name__ == "__main__":
    evaluate_performance()