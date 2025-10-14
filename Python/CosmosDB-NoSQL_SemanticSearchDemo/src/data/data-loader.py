import argparse
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from asyncio import Semaphore

import requests
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from project root
import os

def find_project_root(current_path=__file__):
    """Find the project root by looking for README.md"""
    current_dir = os.path.dirname(os.path.abspath(current_path))
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        if os.path.exists(os.path.join(current_dir, 'README.md')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return os.path.dirname(os.path.abspath(__file__))  # Fallback to script directory

project_root = find_project_root()
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path, override=True)

# Initialize Cosmos DB client
endpoint = os.getenv("COSMOS_URI")

# Initialize OpenAI client with Python 3.13 compatibility
try:
    openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
except TypeError as e:
    if "proxies" in str(e):
        # Fallback for Python 3.13 compatibility issues
        import httpx
        # Create a custom httpx client without the problematic arguments
        custom_client = httpx.Client()
        openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            http_client=custom_client
        )
    else:
        raise e


def initialize_cosmos(database_name):
    client = CosmosClient(endpoint, credential=DefaultAzureCredential())
    
    # Create database if it doesn't exist
    database = client.create_database_if_not_exists(database_name)
    
    # Define the vector property and dimensions (same as main app)
    cosmos_vector_property = "embedding"
    cosmos_full_text_property = "text"
    openai_embeddings_dimensions = 1536

    # policies and indexes (same as main app)
    full_text_policy = {
        "defaultLanguage": "en-US",
        "fullTextPaths": [
            {
                "path": "/" + cosmos_full_text_property,
                "language": "en-US",
            }
        ]
    }
    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/" + cosmos_vector_property,
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": openai_embeddings_dimensions
            },
        ]
    }
    qflat_indexing_policy = {
        "includedPaths": [
            {"path": "/*"}
        ],
        "excludedPaths": [
            {"path": "/\"_etag\"/?"}
        ],
        "vectorIndexes": [
            {
                "path": "/" + cosmos_vector_property,
                "type": "quantizedFlat",
            }
        ]
        # Note: fullTextIndexes require full_text_policy which is not supported in current SDK version
        # "fullTextIndexes": [
        #     {
        #         "path": "/" + cosmos_full_text_property
        #     }
        # ]
    }
    diskann_indexing_policy = {
        "includedPaths": [
            {"path": "/*"}
        ],
        "excludedPaths": [
            {"path": "/\"_etag\"/?"}
        ],
        "vectorIndexes": [
            {
                "path": "/" + cosmos_vector_property,
                "type": "diskANN",
            }
        ]
        # Note: fullTextIndexes require full_text_policy which is not supported in current SDK version
        # "fullTextIndexes": [
        #     {
        #         "path": "/" + cosmos_full_text_property
        #     }
        # ]
    }
    
    # Create containers if they don't exist (same as main app)
    containers = {}
    
    # Create search container without any index - commented out in main app
    # container_name = 'search'
    # containers[container_name] = database.create_container_if_not_exists(
    #     id=container_name,
    #     partition_key=PartitionKey(path="/id"),
    #     full_text_policy=full_text_policy,
    #     vector_embedding_policy=vector_embedding_policy
    # )
    
    # Create search_qflat container with QFLAT vector index
    container_name_qflat = 'search_qflat'
    containers[container_name_qflat] = database.create_container_if_not_exists(
        id=container_name_qflat,
        partition_key=PartitionKey(path="/id"),
        vector_embedding_policy=vector_embedding_policy,
        full_text_policy=full_text_policy,
        indexing_policy=qflat_indexing_policy,
        offer_throughput=10000
    )
    
    # Create search_diskann container with DiskANN vector index
    container_name_diskann = 'search_diskann'
    containers[container_name_diskann] = database.create_container_if_not_exists(
        id=container_name_diskann,
        partition_key=PartitionKey(path="/id"),
        vector_embedding_policy=vector_embedding_policy,
        full_text_policy=full_text_policy,
        indexing_policy=diskann_indexing_policy,
        offer_throughput=10000
    )
    
    return containers


def load_json_data(file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # Handle URLs
        response = requests.get(file_path)
        response.raise_for_status()  # Raise an error if the request fails
        return response.json()
    elif os.path.exists(file_path):
        # Handle local file paths
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        raise ValueError(f"Invalid file path or URL: {file_path}")


def generate_embedding(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
    json_response = response.model_dump_json(indent=2)
    parsed_response = json.loads(json_response)
    return parsed_response['data'][0]['embedding']


def upsert_item_sync(container, item):
    import time
    import random
    
    max_retries = float('inf')  # Infinite retries
    retry_count = 0
    base_delay = 1  # Start with 1 second
    
    while retry_count < max_retries:
        try:
            container.upsert_item(body=item)
            return  # Success, exit the function
        except exceptions.CosmosHttpResponseError as e:
            if e.status_code == 429:  # Too Many Requests (rate limiting)
                retry_count += 1
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** min(retry_count, 10)), 60)  # Cap at 60 seconds
                jitter = random.uniform(0.1, 0.5)  # Add some randomness
                sleep_time = delay + jitter
                print(f"Rate limited, retrying in {sleep_time:.2f} seconds (attempt {retry_count})")
                time.sleep(sleep_time)
            else:
                print(f"Failed to insert document: {e.message}")
                return  # For non-rate-limit errors, don't retry


async def upsert_items_async(containers, items, text_field_name, max_concurrency, vector_field_name=None, re_embed=False):
    semaphore = Semaphore(max_concurrency)
    loop = asyncio.get_event_loop()
    progress_counter = 0
    batch_size = 50  # Process items in batches to avoid overwhelming the system

    async def process_item(item):
        nonlocal progress_counter
        async with semaphore:
            # Rename vector_field_name to embedding if present
            if vector_field_name in item:
                # Re-embed the text if re_embed is True
                if re_embed:
                    item['embedding'] = generate_embedding(item[text_field_name])
                    # get rid of previous vector_field_name, not needed anymore
                    item.pop(vector_field_name)
                    # Rename text_field_name to text so that it matches the streamlit app
                    item['text'] = item.pop(text_field_name)
                else:
                    # Rename vector_field_name to embedding so that it matches the streamlit app
                    item['embedding'] = item.pop(vector_field_name)
                    # Rename text_field_name to text so that it matches the streamlit app
                    item['text'] = item.pop(text_field_name)
            # Generate embedding for text_field_name if present
            elif text_field_name in item:
                item['embedding'] = generate_embedding(item[text_field_name])
                # Rename text_field_name to text so that it matches the streamlit app
                item['text'] = item.pop(text_field_name)

            # Upsert item to all containers
            for container in containers.values():
                await loop.run_in_executor(None, upsert_item_sync, container, item)

            # Update progress counter
            progress_counter += 1
            if progress_counter % 100 == 0:
                print(f"{progress_counter} records processed.")

    # Process items in batches to avoid creating too many tasks at once
    print(f"Processing {len(items)} items in batches of {batch_size}")
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(items) + batch_size - 1) // batch_size
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
        
        # Create tasks for this batch
        tasks = [process_item(item) for item in batch]
        await asyncio.gather(*tasks)
        
        # Small delay between batches to give the system a breather
        await asyncio.sleep(1)



async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Upsert items into Cosmos DB.")
    parser.add_argument("--text_field_name", required=True, help="The name of the field containing text to generate embeddings.")
    parser.add_argument("--path_to_json_array", required=True, help="The path to the JSON file containing the array of items.")
    parser.add_argument("--database_name", required=True, help="The name of the Cosmos DB database.")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum number of concurrent upsert operations.")
    parser.add_argument("--vector_field_name", help="The name of the field containing pre-generated embeddings.")
    parser.add_argument("--re_embed", type=bool, default=False, help="Whether to re-embed the text or not.")
    args = parser.parse_args()

    # Initialize containers and load data
    containers = initialize_cosmos(args.database_name)
    items = load_json_data(args.path_to_json_array)

    # Call the upsert function with the specified parameters
    await upsert_items_async(containers, items, text_field_name=args.text_field_name, max_concurrency=args.concurrency, vector_field_name=args.vector_field_name, re_embed=args.re_embed)

    # how to call this function
    # python src/data/data-loader.py --text_field_name "overview" --path_to_json_array "https://raw.githubusercontent.com/microsoft/AzureDataRetrievalAugmentedGenerationSamples/refs/heads/main/DataSet/Movies/MovieLens-4489-256D.json" --database_name "ignite2024demo" --concurrency 20 --vector_field_name "vector" --re_embed True


if __name__ == "__main__":
    asyncio.run(main())
