import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():
    base_url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    local_file = "sift-128-euclidean.hdf5"
    
    if not os.path.exists(local_file):
        response = requests.get(base_url, stream=True)
        with open(local_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    # Load the dataset
    with h5py.File(local_file, 'r') as f:
        # Load database vectors for indexing
        database = f['train'][:]  # shape: (1000000, 128)
        # Load test vectors for querying
        test = f['test'][:]      # shape: (10000, 128)
    
    # Get dimensions from the data
    num_vectors, dimension = database.shape
    
    # Create HNSW index with specified parameters
    # M=16: maximum number of connections per layer
    # efConstruction=200: size of the dynamic candidate list during construction
    index = faiss.IndexHNSWFlat(dimension, 16)  # dimension=128, M=16
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 200
    
    # Add vectors to the index
    index.add(database)
    
    # Perform search using the first test vector
    query_vector = test[0:1]  # Take the first test vector
    k = 10  # number of nearest neighbors to retrieve
    distances, indices = index.search(query_vector, k)
    
    # Write indices to output.txt
    with open('output.txt', 'w') as f:
        for idx in indices[0]:  # indices[0] because search returns a 2D array
            f.write(f"{idx}\n")

if __name__ == "__main__":
    evaluate_hnsw()
