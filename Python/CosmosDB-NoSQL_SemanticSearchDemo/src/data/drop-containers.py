import os
import time
import random
from azure.cosmos import CosmosClient, exceptions
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Cosmos DB client
endpoint = os.getenv("COSMOS_FABCON_URI")

def delete_container_with_retry(database, container_name):
    """Delete a container with exponential backoff retry on rate limiting."""
    max_retries = 20
    retry_count = 0
    base_delay = 5
    
    while retry_count < max_retries:
        try:
            container = database.get_container_client(container_name)
            database.delete_container(container_name)
            print(f"Successfully deleted container: {container_name}")
            return True
        except exceptions.CosmosResourceNotFoundError:
            print(f"Container {container_name} does not exist, skipping...")
            return True
        except exceptions.CosmosHttpResponseError as e:
            if e.status_code == 429:  # Too Many Requests
                retry_count += 1
                delay = min(base_delay * (2 ** retry_count), 300)  # Cap at 5 minutes
                jitter = random.uniform(0.5, 1.0)
                sleep_time = delay + jitter
                print(f"Rate limited while deleting {container_name}, retrying in {sleep_time:.2f} seconds (attempt {retry_count})")
                time.sleep(sleep_time)
            else:
                print(f"Failed to delete container {container_name}: {e.message}")
                return False
    
    print(f"Failed to delete container {container_name} after {max_retries} attempts")
    return False

def drop_containers():
    client = CosmosClient(endpoint, credential=DefaultAzureCredential())
    database_name = 'fabcon25demo'
    
    try:
        database = client.get_database_client(database_name)
        
        # List of containers to drop
        container_names = ['search', 'search_qflat', 'search_diskann']
        
        for container_name in container_names:
            print(f"Attempting to delete container: {container_name}")
            delete_container_with_retry(database, container_name)
            print(f"Finished processing container: {container_name}")
            time.sleep(10)  # Wait 10 seconds between container deletions
                
    except exceptions.CosmosResourceNotFoundError:
        print(f"Database {database_name} does not exist")
    except exceptions.CosmosHttpResponseError as e:
        print(f"Error accessing database: {e.message}")

if __name__ == "__main__":
    drop_containers()
    print("Container cleanup completed!")
