from typing import Union, List, Optional
import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

app = FastAPI()

# Global variable to store the collection
collection = None

class QueryRequest(BaseModel):
    query: str
    full_text_search: str
    n_results: int = 5
    where: Optional[dict] = None

class QueryResponse(BaseModel):
    products: List[dict]

def initialize_chroma_collection():
    """Initialize or load the ChromaDB collection"""
    global collection

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = chromadb.PersistentClient()

    try:
        # Try to get existing collection
        collection = client.get_collection(
            name="amazon_products",
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
        )
        print("Loaded existing ChromaDB collection")
    except:
        # If collection doesn't exist, create and populate it
        print("Creating new ChromaDB collection...")
        collection = client.create_collection(
            name="amazon_products",
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
        )

        # Load and index the CSV data
        df = pd.read_csv('attached_assets/amazon_prods_final - Sheet1_1757251213016.csv')

        ids = []
        documents = []
        metadatas = []

        for index, row in df.iterrows():
            ids.append(str(row['id']))
            documents.append(row['TEXT'])

            # Clean and convert price fields to integers
            discounted_price = str(row['discounted_price']).replace('₹', '').replace(',', '').strip()
            actual_price = str(row['actual_price']).replace('₹', '').replace(',', '').strip()
            discount_percentage = str(row['discount_percentage']).replace('%', '').strip()
            
            metadata = {
                'discounted_price': int(float(discounted_price)) if discounted_price and discounted_price != 'nan' else 0,
                'actual_price': int(float(actual_price)) if actual_price and actual_price != 'nan' else 0,
                'discount_percentage': int(float(discount_percentage)) if discount_percentage and discount_percentage != 'nan' else 0,
                'img_link': str(row['img_link']),
                'product_link': str(row['product_link'])
            }

            # Add rating (random if null) - as integer
            if pd.notna(row['rating']):
                metadata['rating'] = int(float(row['rating']))  # Convert to float first, then to int
            else:
                metadata['rating'] = random.randint(1, 5)

            # Add rating_count (random if null) - as integer
            if pd.notna(row.get('rating_count')):
                # Remove commas and convert to int
                rating_count_str = str(row['rating_count']).replace(',', '')
                try:
                    metadata['rating_count'] = int(rating_count_str)
                except ValueError:
                    # If conversion fails, use a random value
                    metadata['rating_count'] = random.randint(1, 1000)
            else:
                metadata['rating_count'] = random.randint(1, 1000)

            metadatas.append(metadata)

        # Add documents in batches
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

        print(f"Indexed {len(ids)} products to ChromaDB")

@app.on_event("startup")
async def startup_event():
    """Initialize ChromaDB on startup"""
    try:
        initialize_chroma_collection()
    except Exception as e:
        print(f"Failed to initialize ChromaDB: {e}")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/search", response_model=QueryResponse)
def search_products(query_request: QueryRequest):
    """Search for products using semantic similarity with metadata filtering"""
    global collection

    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized")

    try:
        # Build query parameters
        query_params = {
            "query_texts": [query_request.query],
            "n_results": query_request.n_results,
            "where_document": {"$contains": query_request.full_text_search}
        }
        
        # Add metadata filtering if provided
        if query_request.where:
            query_params["where"] = query_request.where

        results = collection.query(**query_params)

        products = []
        for i in range(len(results['ids'][0])):
            product = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            products.append(product)

        return QueryResponse(products=products)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search")
def search_products_get(q: str, full_text_search: str, n_results: int = 5, where: Optional[str] = None):
    """Search for products using GET request"""
    where_dict = None
    if where:
        try:
            import json
            where_dict = json.loads(where)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for 'where' parameter")
    
    query_request = QueryRequest(query=q, full_text_search=full_text_search, n_results=n_results, where=where_dict)
    return search_products(query_request)

@app.get("/collection/info")
def get_collection_info():
    """Get information about the ChromaDB collection"""
    global collection

    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized")

    try:
        count = collection.count()
        return {
            "collection_name": "amazon_products",
            "total_documents": count,
            "status": "ready"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")