
import chromadb
import pandas as pd
import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def index_products_to_chroma():
    # Read the CSV file
    df = pd.read_csv('attached_assets/amazon_prods_final - Sheet1_1757251213016.csv')
    
    print(f"Loaded {len(df)} products from CSV")
    
    # Initialize ChromaDB client
    client = chromadb.EphemeralClient()
    
    # Create collection with OpenAI embedding function
    collection = client.create_collection(
        name="amazon_products",
        embedding_function=OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
    )
    
    # Prepare data for indexing
    ids = []
    documents = []
    metadatas = []
    
    for index, row in df.iterrows():
        # Use the id as string for ChromaDB
        ids.append(str(row['id']))
        
        # Use the TEXT column for embedding
        documents.append(row['TEXT'])
        
        # Prepare metadata with the specified fields
        metadata = {
            'discounted_price': str(row['discounted_price']),
            'actual_price': str(row['actual_price']),
            'discount_percentage': str(row['discount_percentage']),
            'img_link': str(row['img_link']),
            'product_link': str(row['product_link'])
        }
        
        # Handle rating field (might be NaN)
        if pd.notna(row['rating']):
            metadata['rating'] = float(row['rating'])
        else:
            metadata['rating'] = None
            
        metadatas.append(metadata)
    
    print("Adding documents to ChromaDB...")
    
    # Add documents to collection in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_documents = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        
        print(f"Indexed batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
    
    print(f"Successfully indexed {len(ids)} products!")
    
    # Test the collection
    test_results = collection.query(
        query_texts=["laptop computer"],
        n_results=3
    )
    
    print("\nTest query results for 'laptop computer':")
    for i, doc in enumerate(test_results['documents'][0]):
        print(f"{i+1}. {doc[:100]}...")
    
    return collection

if __name__ == "__main__":
    # Make sure to set your OpenAI API key as an environment variable
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        exit(1)
    
    collection = index_products_to_chroma()
