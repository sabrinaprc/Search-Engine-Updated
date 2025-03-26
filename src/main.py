import nltk
from nltk.stem import PorterStemmer
from tfidf import calculate_tf_idf
from tokenizer import tokenize
import json, os
from pymongo import MongoClient

# Ensure nltk resources are available
nltk.download('punkt')
ps = PorterStemmer()
client = MongoClient("mongodb://localhost:27017/")
db = client["search_engine"]
index_collection = db["inverted_index"]
url_collection = db["doc_urls"]

def main():
    print("\n--- Manual Query Input ---")
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip() 
        if query.lower() == "exit":
             print("Exiting manual query mode.")
             break
        print(f"\nProcessing query: '{query}'...")
        ranked_results = calculate_tf_idf(query, index_collection, url_collection)
        
        if ranked_results:
             print("\nTop results:")
             for rank, (doc_id, score) in enumerate(ranked_results[:5], start=1):
                 doc_entry = url_collection.find_one({"doc_id": doc_id})
                 url = doc_entry["url"] if doc_entry else f"Document {doc_id}"
                 print(f"{rank}. {url} (TF Score: {score})")
        else:
            print("No matching documents found.")
             

if __name__ == "__main__":
    main()