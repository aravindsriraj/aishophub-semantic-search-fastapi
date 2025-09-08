# AiShopHub Semantic Search FastAPI

A FastAPI-based semantic search service for AI shopping hub using ChromaDB and OpenAI embeddings. This service provides intelligent product search capabilities for an Amazon product dataset.

## Features

- **Semantic Product Search**: Advanced search using OpenAI's text-embedding-3-small model
- **FastAPI REST API**: High-performance API with automatic OpenAPI documentation
- **ChromaDB Vector Database**: Efficient vector storage and similarity search
- **Bearer Token Authentication**: Secure API access
- **Product Metadata**: Rich product information including prices, ratings, and links

## Live Demo

- **API Documentation**: [https://product-search.replit.app/docs](https://product-search.replit.app/docs)
- **Interactive Swagger UI**: Test all endpoints directly in the browser

## Setup

### Prerequisites

- Python 3.10 or 3.11
- OpenAI API key
- Poetry (for dependency management)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/aravindsriraj/aishophub-semantic-search-fastapi.git
cd aishophub-semantic-search-fastapi
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

4. Run the application:
```bash
poetry run uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Authentication
All endpoints require Bearer token authentication:
```
Authorization: Bearer *ULXrUUDkkjRheg3cjpQAcBbzGgffZBn!32ssr8JRW9VERcVmweQqGnYi!Y8jcPnG
```

### Search Products
- **POST** `/search` - Semantic search with JSON body
- **GET** `/search?q=query&n_results=5` - Simple search with query parameters

### Collection Info
- **GET** `/collection/info` - Get information about the indexed products

## Example Usage

### Using curl:
```bash
curl -X POST "https://product-search.replit.app/search" \
  -H "Authorization: Bearer *ULXrUUDkkjRheg3cjpQAcBbzGgffZBn!32ssr8JRW9VERcVmweQqGnYi!Y8jcPnG" \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless headphones", "n_results": 10}'
```

### Using Python requests:
```python
import requests

headers = {
    "Authorization": "Bearer *ULXrUUDkkjRheg3cjpQAcBbzGgffZBn!32ssr8JRW9VERcVmweQqGnYi!Y8jcPnG",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://product-search.replit.app/search",
    headers=headers,
    json={"query": "smartphone", "n_results": 5}
)

products = response.json()["products"]
```

## Project Structure

```
├── main.py                 # FastAPI application and API endpoints
├── index_products.py       # Script to index products into ChromaDB
├── pyproject.toml         # Poetry dependencies and configuration
├── attached_assets/       # CSV data file with Amazon products
├── chroma/               # ChromaDB storage directory
└── README.md             # This file
```

## Data

The service uses Amazon product data with the following features:
- Product descriptions and titles
- Pricing information (actual and discounted prices)
- Product ratings and review counts
- Product and image links
- Discount percentages

## Technology Stack

- **FastAPI**: Modern Python web framework
- **ChromaDB**: Vector database for embeddings
- **OpenAI API**: Text embeddings generation
- **Pandas**: Data processing
- **Pydantic**: Data validation and serialization

## Development

To run in development mode:
```bash
poetry run uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` to access the interactive API documentation.

## License

This project is available under the MIT License.
