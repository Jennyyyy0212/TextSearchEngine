# TextSearchEngine  
A lightweight, efficient search engine built from scratch for Information Retrieval (IR) applications. This project implements core IR techniques such as text preprocessing, indexing, and ranking to enable fast and relevant document retrieval.

## Features  
- **Inverted Indexing** 📂 - Efficient document storage and retrieval.  
- **Text Preprocessing** 📝 - Tokenization, stemming (Porter Stemmer), and stop-word removal.  
- **Query Processing** 🔍 - Supports TF-IDF scoring and cosine similarity for ranking.  
- **Efficient Search Algorithms** ⚡ - Optimized for speed and relevance.  
- **Scalability** 📊 - Sharding techniques to handle large datasets.  

## Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/TextSearchEngine.git
   cd TextSearchEngine
   ```

2. Run the search engine
    ```
    python search_engine.py
    ```

## Usage
1. Add documents to the search index.
2. Enter a search query.
3. Retrieve ranked results based on TF-IDF scoring.

## Run Small Test Cases 
Execute test cases to ensure the search engine runs correctly:
```
cd test
python search_engine.py
```