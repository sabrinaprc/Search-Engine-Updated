import math
from tokenizer import tokenize
from collections import defaultdict

def get_wordcount_dict(inverted_index):
    doc_index = defaultdict(int)
    for term, postings in inverted_index.items():
        for entry in postings:
            doc_index[entry['doc_id']] += entry['frequency']
    
    return doc_index

def calculate_tf_idf(query, inverted_index, total_docs, word_count_dict):
    query_tokens = tokenize(query)
    print(f"Query tokens: {query_tokens}")
    doc_scores = {}

    for token in query_tokens:
        if token in inverted_index:
            # Calculate IDF for the token
            df = len(inverted_index[token])  # Number of documents containing the term
            idf = math.log10(total_docs / (1 + df))  # Add 1 to avoid division by zero
            print(f"Token: '{token}', IDF: {idf:.4f}")
            
            MIN_TF_IDF_THRESHOLD = 0.01
            
            # Process all documents where the term appears
            for entry in inverted_index[token]:
                doc_id = entry['doc_id']
                tf = entry['frequency'] / word_count_dict.get(doc_id) # Calculate TF

                # Calculate TF-IDF
                tf_idf = tf * idf
                
                if tf_idf < MIN_TF_IDF_THRESHOLD:
                    continue

                # Add to the document's score
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += tf_idf

                print(f"  Doc ID: {doc_id}, TF: {tf:.4f}, TF-IDF: {tf_idf:.4f}")
        else:
            print(f"Token '{token}' not found in the inverted index.")

    # Sort documents by TF-IDF scores in descending order
    sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_scores
