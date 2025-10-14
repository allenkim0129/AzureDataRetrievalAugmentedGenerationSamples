import streamlit as st
from openai import AzureOpenAI
import os
import json
import pandas as pd
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import time

# Load environment variables from project root
def find_project_root(current_path=__file__):
    """Find the project root by looking for README.md"""
    import os
    current_dir = os.path.dirname(os.path.abspath(current_path))
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        if os.path.exists(os.path.join(current_dir, 'README.md')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return os.path.dirname(os.path.abspath(__file__))  # Fallback to script directory

project_root = find_project_root()
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path, override=True)

def debug_container_capabilities(container, container_name):
    """Debug function to check container's full text search capabilities"""
    try:
        print(f"🔍 DEBUGGING CONTAINER: {container_name}")
        
        # Test a simple full text search query to see if it's supported
        test_query = '''
        SELECT TOP 1 l.id, l.title, l.text
        FROM l
        WHERE FullTextContainsAny(l.text, "test")
        '''
        
        print(f"  Testing basic full text search...")
        results = container.query_items(test_query, enable_cross_partition_query=True)
        results_list = list(results)
        print(f"  ✓ Basic full text search works - found {len(results_list)} results")
        
        # Test a simple full text ranking query
        test_ranking_query = '''
        SELECT TOP 1 l.id, l.title, l.text
        FROM l
        ORDER BY RANK FullTextScore(l.text, "test")
        '''
        
        print(f"  Testing basic full text ranking...")
        results = container.query_items(test_ranking_query, enable_cross_partition_query=True)
        results_list = list(results)
        print(f"  ✓ Basic full text ranking works - found {len(results_list)} results")
        
    except Exception as e:
        print(f"  ❌ Container capability test failed: {str(e)}")
        return False
        
    return True

st.set_page_config(page_title="Azure Cosmos DB Semantic Search Demo", layout="wide", initial_sidebar_state="expanded")
# UI text strings
page_title = "Azure Cosmos DB - Semantic Search Demo"
page_helper = "Showcasing vector search, full text search, text ranking, hybrid search, and semantic reranking capabilities."
empty_search_helper = "Enter text to get started."
semantic_search_header = "Search input"
semantic_search_placeholder = "James Bond"
vector_search_label = "Similarity search"
full_text_ranking_label = "Full text ranking"
full_text_search_label = "Full text search"
venue_list_header = "Research papers"
hybrid_search_label = "Hybrid search"

# Initialize global variables for Cosmos DB client, database, and containers
if "cosmos_client" not in st.session_state:
    endpoint = os.getenv("COSMOS_URI")
    
    # Check if endpoint is configured
    if not endpoint or endpoint == "your_cosmos_db_uri_here":
        st.error("❌ Cosmos DB endpoint not configured. Please update the .env file with your actual Cosmos DB URI.")
        st.info("💡 The app will work for reranker testing, but search functionality requires Cosmos DB endpoint.")
        st.session_state.cosmos_client = None
        st.session_state.cosmos_database = None
        st.session_state.cosmos_container_qflat = None
        st.session_state.cosmos_container_diskann = None
    else:
        try:
            # Use DefaultAzureCredential for keyless authentication
            st.info("🆔 Using DefaultAzureCredential for Cosmos DB (Managed Identity/Azure CLI)")
            credential = DefaultAzureCredential()
            
            st.session_state.cosmos_client = CosmosClient(endpoint, credential=credential)
            database_name = os.getenv("COSMOS_DB_DATABASE", "searchdemo")  # Get database name from environment or use default
            st.session_state.cosmos_database = st.session_state.cosmos_client.create_database_if_not_exists(database_name)
            
            st.success("✅ Connected to Cosmos DB using keyless authentication")
            
        except Exception as e:
            st.error(f"Failed to connect to Cosmos DB: {str(e)}")
            
            st.info("💡 Keyless authentication failed. This could be because:")
            st.info("   • Azure CLI is not logged in (run 'az login')")
            st.info("   • Managed Identity is not configured properly")
            st.info("   • Your account doesn't have access to the Cosmos DB resource")
            st.info("   • Ensure your identity has 'Cosmos DB Built-in Data Contributor' role")
            
            st.session_state.cosmos_client = None
            st.session_state.cosmos_database = None
            st.session_state.cosmos_container_qflat = None
            st.session_state.cosmos_container_diskann = None

    # Define the vector property and dimensions
    cosmos_vector_property = "embedding"
    cosmos_full_text_property = "text"
    openai_embeddings_dimensions = 1536

    # policies and indexes
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
        ],
        "fullTextIndexes": [
            {
                "path": "/" + cosmos_full_text_property
            }
        ]
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
        ],
        "fullTextIndexes": [
            {
                "path": "/" + cosmos_full_text_property
            }
        ]
    }

    # Create listings_search container without any index
    # container_name = 'search'
    # st.session_state.cosmos_container = st.session_state.cosmos_database.create_container_if_not_exists(
    #     id=container_name,
    #     partition_key=PartitionKey(path="/id"),
    #     full_text_policy=full_text_policy,
    #     vector_embedding_policy=vector_embedding_policy#,
    #     #offer_throughput=1000
    # )


    # Create containers only if we have a valid database connection
    if st.session_state.cosmos_database is not None:
        # Create listings_search_qflat container with QFLAT vector index
        container_name_qflat = 'search_qflat'
        st.session_state.cosmos_container_qflat = st.session_state.cosmos_database.create_container_if_not_exists(
            id=container_name_qflat,
            partition_key=PartitionKey(path="/id"),
            # full_text_policy=full_text_policy,  # Temporarily commented out for compatibility
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=qflat_indexing_policy,
            offer_throughput=400
        )

        # Create listings_search_diskann container with DiskANN vector index
        container_name_diskann = 'search_diskann'
        st.session_state.cosmos_container_diskann = st.session_state.cosmos_database.create_container_if_not_exists(
            id=container_name_diskann,
            partition_key=PartitionKey(path="/id"),
            # full_text_policy=full_text_policy,  # Temporarily commented out for compatibility
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=diskann_indexing_policy,
            offer_throughput=400
        )
        
        # Debug container capabilities
        print("🔍 DEBUGGING CONTAINER CAPABILITIES:")
        debug_container_capabilities(st.session_state.cosmos_container_qflat, "QFLAT")
        debug_container_capabilities(st.session_state.cosmos_container_diskann, "DiskANN")

# Initialize session state variables
if "embedding_gen_time" not in st.session_state:
    st.session_state.embedding_gen_time = ""
if "query_time" not in st.session_state:
    st.session_state.query_time = ""
if "ru_consumed" not in st.session_state:
    st.session_state.ru_consumed = ""
if "executed_query" not in st.session_state:
    st.session_state.executed_query = ""
if "server_query_time" not in st.session_state:
    st.session_state.server_query_time = ""

# Function to log times
def log_time(start):
    end = time.perf_counter()
    elapsed_time = end - start
    return f"{elapsed_time:.4f} seconds"

# Initialize the embedding client only once
if "embedding_client" not in st.session_state:
    try:
        st.session_state.embedding_client = AzureOpenAI(
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
            st.session_state.embedding_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2023-05-15",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                http_client=custom_client
            )
        else:
            raise e

# Handler functions
def embedding_query(text_input):
    print("text_input", text_input)
    start_time = time.perf_counter()
    response = st.session_state.embedding_client.embeddings.create(
        input=text_input,
        model="text-embedding-ada-002"  # Use the appropriate model
    )

    json_response = response.model_dump_json(indent=2)
    parsed_response = json.loads(json_response)
    embedding = parsed_response['data'][0]['embedding']
    st.session_state.embedding_gen_time = log_time(start_time)
    print(f"Embedding generation time: {st.session_state.embedding_gen_time}")
    return embedding

def rerank_results_with_cosmos_sdk(container, query, results_df, use_reranker=False):
    """
    Rerank search results using Azure Cosmos DB's built-in semantic reranker.
    
    Args:
        container: The Cosmos container instance to use for reranking
        query: The search query string
        results_df: DataFrame containing search results with 'text' column
        use_reranker: Boolean flag to enable/disable reranking
    
    Returns:
        DataFrame with reranked results (or original if reranker disabled/failed)
    """
    if not use_reranker or results_df.empty or container is None:
        return results_df
    
    try:
        # Extract documents for reranking from the text column
        documents = results_df['text'].tolist() if 'text' in results_df.columns else []
        
        if not documents:
            print("No text content found in results for reranking")
            return results_df
        
        print(f"🔍 Calling semantic reranker with {len(documents)} documents...")
        
        # Call the Cosmos SDK semantic_rerank method
        try:
            reranked_data = container.semantic_rerank(
                context=query,
                documents=documents,
                options={
                    "return_documents": True,
                    "top_k": len(documents),  # Return all documents reranked
                    "batch_size": 32,
                    "sort": True
                }
            )
            print("✓ Semantic reranking successful!")
        except Exception as e:
            print(f"⚠ Semantic reranking failed: {str(e)}")
            print(" Returning original search results without reranking")
            return results_df
        
        # Process the response - expect "Scores" key
        if 'Scores' in reranked_data:
            scores_data = reranked_data['Scores']
            print(f"✓ Successfully reranked {len(scores_data)} results")
            
            # The response contains documents sorted by score
            reranked_df = pd.DataFrame()
            
            for i, score_item in enumerate(scores_data):
                document_text = score_item.get('document', '')
                score = score_item.get('relevance_score', score_item.get('score', 0))
                original_index = score_item.get('index', i)
                
                # Try to use the index first, then fall back to text matching
                if original_index < len(results_df):
                    original_row = results_df.iloc[original_index].copy()
                    original_row['rerank_score'] = score
                    reranked_df = pd.concat([reranked_df, original_row.to_frame().T], ignore_index=True)
                else:
                    # Fallback: Find the original row by matching text content
                    matching_rows = results_df[results_df['text'] == document_text]
                    
                    if len(matching_rows) > 0:
                        original_row = matching_rows.iloc[0].copy()
                        original_row['rerank_score'] = score
                        reranked_df = pd.concat([reranked_df, original_row.to_frame().T], ignore_index=True)
            
            if len(reranked_df) > 0:
                return reranked_df
            else:
                print("⚠ Could not match reranked results with original data")
                return results_df
        else:
            print(f"⚠ Unexpected response format from semantic_rerank")
            return results_df
            
    except Exception as e:
        print(f"⚠ Semantic reranking failed: {e}")
        print(" Returning original search results without reranking")
        return results_df


def handler_vector_search(indices, ask):
    emb = embedding_query(ask)
    num_results = 10

    # Query strings
    vector_search_query = f'''
    SELECT TOP {num_results} l.id, l.title, l.text, VectorDistance(l.embedding, {emb}) as SimilarityScore
    FROM l
    ORDER BY VectorDistance(l.embedding,{emb})
    '''

    obfuscated_query = vector_search_query.replace(str(emb), "REDACTED")

    container = {
        #'No Index': st.session_state.cosmos_container,
        'QFLAT & Full Text Search Index': st.session_state.cosmos_container_qflat,
        'DiskANN & Full Text Search Index': st.session_state.cosmos_container_diskann
    }.get(indices)

    try:
        start_time = time.perf_counter()  # Capture start time
        st.session_state.executed_query = obfuscated_query
        results = container.query_items(vector_search_query, enable_cross_partition_query=True, populate_query_metrics=True)
        results_list = list(results)
        elapsed_time = log_time(start_time)
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results_list)
        
        # Apply reranking if enabled
        use_reranker = st.session_state.get("use_reranker", False)
        if use_reranker:
            results_df = rerank_results_with_cosmos_sdk(container, ask, results_df, use_reranker)
        
        st.session_state.suggested_listings = results_df
        st.session_state.query_time = elapsed_time
        st.session_state.ru_consumed = container.client_connection.last_response_headers['x-ms-request-charge']
        total_execution_time = parse_server_query_time(container.client_connection.last_response_headers['x-ms-documentdb-query-metrics'])
        st.session_state.server_query_time = total_execution_time
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred: {e}")

def handler_text_search(indices, text, search_type):
    num_results = 10

    # Tokenize text into individual words
    keywords = text.split()  # Split the text into words
    formatted_keywords = ', '.join(f'"{keyword}"' for keyword in keywords)
    print(formatted_keywords)# Format keywords for query

    # Construct the query string with tokenized keywords
    if search_type == "all keywords":
        full_text_search_query = f'''
        SELECT TOP {num_results} l.id, l.title, l.text
        FROM l
        WHERE FullTextContainsAll(l.text, {formatted_keywords})
        '''
    else:
        full_text_search_query = f'''
        SELECT TOP {num_results} l.id, l.title, l.text
        FROM l
        WHERE FullTextContainsAny(l.text, {formatted_keywords})
        '''

    container = {
        #'No Index': st.session_state.cosmos_container,
        'QFLAT & Full Text Search Index': st.session_state.cosmos_container_qflat,
        'DiskANN & Full Text Search Index': st.session_state.cosmos_container_diskann
    }.get(indices)

    try:
        start_time = time.perf_counter()  # Capture start time
        st.session_state.executed_query = full_text_search_query
        results = container.query_items(full_text_search_query, enable_cross_partition_query=True, populate_query_metrics=True)
        results_list = list(results)
        elapsed_time = log_time(start_time)
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results_list)
        
        # Apply reranking if enabled
        use_reranker = st.session_state.get("use_reranker", False)
        if use_reranker:
            results_df = rerank_results_with_cosmos_sdk(container, text, results_df, use_reranker)
        
        st.session_state.suggested_listings = results_df
        st.session_state.query_time = elapsed_time
        st.session_state.ru_consumed = container.client_connection.last_response_headers['x-ms-request-charge']
        total_execution_time = parse_server_query_time(container.client_connection.last_response_headers['x-ms-documentdb-query-metrics'])
        st.session_state.server_query_time = total_execution_time
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred: {e}")

def handler_text_ranking(indices, text):
    num_results = 10

    # For FullTextScore, we need the original text as a single quoted string
    formatted_text = f'"{text}"'  # Single quoted string for FullTextScore
    
    print(f"🔍 FULL TEXT RANKING DEBUG:")
    print(f"  Input text: '{text}'")
    print(f"  Formatted for FullTextScore: {formatted_text}")
    print(f"  Selected index: {indices}")

    # Construct the query string with the properly formatted text
    full_text_ranking_query = f'''
    SELECT TOP {num_results} l.id, l.title, l.text
    FROM l
    ORDER BY RANK FullTextScore(l.text, {formatted_text})
    '''
    
    print(f"  Generated query: {full_text_ranking_query}")

    container = {
        #'No Index': st.session_state.cosmos_container,
        'QFLAT & Full Text Search Index': st.session_state.cosmos_container_qflat,
        'DiskANN & Full Text Search Index': st.session_state.cosmos_container_diskann
    }.get(indices)

    try:
        start_time = time.perf_counter()  # Capture start time
        st.session_state.executed_query = full_text_ranking_query
        results = container.query_items(full_text_ranking_query, enable_cross_partition_query=True,populate_query_metrics=True)
        results_list = list(results)
        elapsed_time = log_time(start_time)
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results_list)
        
        # Apply reranking if enabled
        use_reranker = st.session_state.get("use_reranker", False)
        if use_reranker:
            results_df = rerank_results_with_cosmos_sdk(container, text, results_df, use_reranker)
        
        st.session_state.suggested_listings = results_df
        st.session_state.query_time = elapsed_time
        st.session_state.ru_consumed = container.client_connection.last_response_headers['x-ms-request-charge']
        total_execution_time = parse_server_query_time(container.client_connection.last_response_headers['x-ms-documentdb-query-metrics'])
        st.session_state.server_query_time = total_execution_time
    except exceptions.CosmosHttpResponseError as e:
        print(f"❌ FULL TEXT RANKING ERROR:")
        print(f"  Status Code: {e.status_code}")
        print(f"  Message: {e.message}")
        print(f"  Query that failed: {full_text_ranking_query}")
        st.error(f"Full Text Ranking Error: {e.message} (Status: {e.status_code})")
        st.error(f"Failed Query: {full_text_ranking_query}")
    except Exception as e:
        print(f"❌ UNEXPECTED FULL TEXT RANKING ERROR: {str(e)}")
        st.error(f"Unexpected error in Full Text Ranking: {str(e)}")

def handler_hybrid_ranking(indices, text):
    num_results = 10
    emb = embedding_query(text)
    # For FullTextScore, we need the original text as a single quoted string
    formatted_text = f'"{text}"'  # Single quoted string for FullTextScore
    
    print(f"🔍 HYBRID SEARCH DEBUG:")
    print(f"  Input text: '{text}'")
    print(f"  Formatted for FullTextScore: {formatted_text}")
    print(f"  Embedding length: {len(emb)}")
    print(f"  Selected index: {indices}")

    # Construct the query string with properly formatted text
    full_hybrid_ranking_query = f'''
    SELECT TOP {num_results} l.id, l.title, l.text
    FROM l
    ORDER BY RANK RRF(FullTextScore(l.text, {formatted_text}), VectorDistance(l.embedding, {emb}))
    '''
    
    print(f"  Generated query (with embedding): {full_hybrid_ranking_query[:200]}...")

    obfuscated_query = full_hybrid_ranking_query.replace(str(emb), "REDACTED")

    container = {
        #'No Index': st.session_state.cosmos_container,
        'QFLAT & Full Text Search Index': st.session_state.cosmos_container_qflat,
        'DiskANN & Full Text Search Index': st.session_state.cosmos_container_diskann
    }.get(indices)

    try:
        start_time = time.perf_counter()  # Capture start time
        st.session_state.executed_query = obfuscated_query
        results = container.query_items(full_hybrid_ranking_query, enable_cross_partition_query=True,populate_query_metrics=True)
        results_list = list(results)
        elapsed_time = log_time(start_time)
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results_list)
        
        # Apply reranking if enabled
        use_reranker = st.session_state.get("use_reranker", False)
        if use_reranker:
            results_df = rerank_results_with_cosmos_sdk(container, text, results_df, use_reranker)
        
        st.session_state.suggested_listings = results_df
        st.session_state.query_time = elapsed_time
        st.session_state.ru_consumed = container.client_connection.last_response_headers['x-ms-request-charge']
        total_execution_time = parse_server_query_time(container.client_connection.last_response_headers['x-ms-documentdb-query-metrics'])
        st.session_state.server_query_time = total_execution_time
    except exceptions.CosmosHttpResponseError as e:
        print(f"❌ HYBRID SEARCH ERROR:")
        print(f"  Status Code: {e.status_code}")
        print(f"  Message: {e.message}")
        print(f"  Query that failed: {obfuscated_query}")
        st.error(f"Hybrid Search Error: {e.message} (Status: {e.status_code})")
        st.error(f"Failed Query: {obfuscated_query}")
    except Exception as e:
        print(f"❌ UNEXPECTED HYBRID SEARCH ERROR: {str(e)}")
        st.error(f"Unexpected error in Hybrid Search: {str(e)}")

# UI elements
def render_cta_link(url, label, font_awesome_icon):
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">',
        unsafe_allow_html=True)
    button_code = f'''<a href="{url}" target=_blank><i class="fa {font_awesome_icon}"></i> {label}</a>'''
    return st.markdown(button_code, unsafe_allow_html=True)

def parse_server_query_time(query_metrics):
    metrics_parts = query_metrics.split(";")
    total_execution_time_ms = next(
        (part.split("=")[1] for part in metrics_parts if "totalExecutionTimeInMs" in part), "0"
    )
    total_execution_time_s = float(total_execution_time_ms) / 1000
    return f"{total_execution_time_s:.4f} seconds"

def render_search():
    search_disabled = True
    with st.sidebar:
        st.selectbox(label="Index", options=['QFLAT & Full Text Search Index', 'DiskANN & Full Text Search Index'], index=0, key="index_selection")
        st.text_input(label=semantic_search_header, placeholder=semantic_search_placeholder, key="user_query")
        
        # Add reranker checkbox
        st.checkbox("Use Semantic Reranker", key="use_reranker", help="Apply semantic reranking to search results using Azure's reranker service")

        if "user_query" in st.session_state and st.session_state.user_query != "":
            search_disabled = False

        st.button(label=vector_search_label, key="location_search", disabled=search_disabled,
                  on_click=handler_vector_search, args=(st.session_state.index_selection, st.session_state.user_query))

        # Button for Full Text Ranking search using handler_text_ranking
        st.button(label=full_text_ranking_label, key="full_text_ranking", disabled=search_disabled,
                  on_click=handler_text_ranking, args=(st.session_state.index_selection, st.session_state.user_query))

        # Button for Hybrid Ranking search using handler_hybrid_ranking
        st.button(label=hybrid_search_label, key="hybrid_search", disabled=search_disabled,
                  on_click=handler_hybrid_ranking, args=(st.session_state.index_selection, st.session_state.user_query))

        search_type = st.radio("Search type", options=["all keywords", "any keywords"], key="full_text_search_type")

        st.button(label=full_text_search_label, key="full_text_search", disabled=search_disabled,
                  on_click=handler_text_search, args=(st.session_state.index_selection, st.session_state.user_query, search_type))



        st.write("---")
        render_cta_link(url="https://azurecosmosdb.github.io/gallery/", label="Cosmos DB Samples Gallery", font_awesome_icon="fa-cosmosdb")
        render_cta_link(url="https://github.com/TheovanKraay/AzureDataRetrievalAugmentedGenerationSamples", label="GitHub", font_awesome_icon="fa-github")

def render_search_result():
    col1 = st.container()
    col1.write(f"Executed query: {st.session_state.executed_query}")
    col1.write(f"Embedding generation time: {st.session_state.embedding_gen_time}")
    col1.write(f"Total end-to-end query execution time: {st.session_state.query_time}")
    col1.write(f"Total server query execution time: {st.session_state.server_query_time}")
    col1.write(f"RU consumed: {st.session_state.ru_consumed}")
    
    # Show reranker status
    use_reranker = st.session_state.get("use_reranker", False)
    reranker_applied = use_reranker and "rerank_score" in st.session_state.suggested_listings.columns
    col1.write(f"Semantic reranking: {'✓ Applied' if reranker_applied else '✗ Not applied'}")
    
    col1.write(f"Found {len(st.session_state.suggested_listings)} records.")
    col1.table(st.session_state.suggested_listings)

# Main execution
render_search()

st.title(page_title)
st.write(page_helper)
st.write("---")

if "suggested_listings" not in st.session_state:
    st.write(empty_search_helper)
else:
    render_search_result()