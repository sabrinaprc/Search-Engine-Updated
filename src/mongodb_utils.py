from email.policy import default
import os
import json
from pymongo import MongoClient
from collections import defaultdict

client = MongoClient("mongodb://localhost:27017/")
db = client["search_engine"]
collection = db["inverted_index"]
collection.create_index("term", unique=True)


def merge_postings(existing, new):
    merged = defaultdict(float)
    for entry in existing + new:
        merged[entry['doc_id']] += entry['frequency']
    return [{"doc_id": doc_id, "frequency": freq} for doc_id, freq in merged.items()]

def load_and_store_partial_indexes():
    PARTIAL_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..')
    files = [f for f in os.listdir(PARTIAL_INDEX_PATH) if f.startswith("partial_index_") and f.endswith(".json")]
    
    for filename in files:
        full_path = os.path.join(PARTIAL_INDEX_PATH, filename)
        print(f"Processing {filename}...")
        with open(full_path, 'r') as f:
            partial_index = json.load(f)

            for term, postings in partial_index.items():
                existing_doc = collection.find_one({"term": term})
                if existing_doc:
                    merged_postings = merge_postings(existing_doc["postings"], postings)
                    collection.update_one(
                        {"term": term},
                        {"$set": {"postings": merged_postings}}
                    )
                else:
                    collection.insert_one({
                        "term": term,
                        "postings": postings
                    })
    print("All partial indexes have been loaded into MongoDB.")

if __name__ == "__main__":
    load_and_store_partial_indexes()
    
