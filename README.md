Project Overview

This project is the final milestone in the CS121 Search Engine assignment. The goal was to build a fully functioning search engine that supports efficient and effective query retrieval. We focused on improving both ranking performance and runtime efficiency, implementing various strategies to optimize the engine's behavior under realistic conditions.

The assignment builds upon earlier work by:
- Incorporating test queries to evaluate performance
- Optimizing the engine for poor-performing queries
- Transitioning to a low-memory footprint design using MongoDB for index storage

Deliverables
- A complete codebase for indexing, tokenization, query processing, and TF-IDF ranking
- Optimizations based on poor-performing queries
- All partial indexes are merged and stored in MongoDB for scalable access
-  Ready for live demo with screen share and walkthrough

KEY FEATURES:
Indexing
- Partial index generation
- Stored on disk and then consolidated in MongoDB using the pymongo lib
  
Query Processing
- Tokenization using nltk and stemming via PorterStemmer.
- TF-IDF scoring with document-level normalization.
- Queries are ranked and displayed with URLs and relevance scores.

MongoDB Integration
- All inverted index data is stored in the search_engine.inverted_index collection.
- Document metadata (URLs and word counts) stored in search_engine.doc_urls.
- Efficient querying and index lookups using MongoDB's indexing features.

Optimizations
- Improved performance for poor-performing queries via:
- Weighted tokenization
- Filtering low TF-IDF terms
- Index lookups using MongoDB

Notes
Make sure MongoDB is running locally (mongodb://localhost:27017) before running any indexing or query scripts.
