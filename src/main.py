import nltk
from nltk.stem import PorterStemmer
from tfidf import calculate_tf_idf, get_wordcount_dict
from file_utils import load_doc_id_url_mapping
from tokenizer import tokenize
from index_builder import *
import json, os
# Ensure nltk resources are available
nltk.download('punkt')

ps = PorterStemmer()

# Function to write the partial index to disk and clear the in-memory index
# def sort_and_write_to_disk(index, filename):
#     sorted_index = {token: sorted(postings, key=lambda x: x['doc_id']) for token, postings in index.items()}
#     with open(filename, 'w') as f:
#         json.dump(sorted_index, f, indent=2)
#     print(f"Partial index saved to {filename}")


# # Function to calculate the total size of all partial index files in KB
# def calculate_index_size():
#     total_size = 0
#     for filename in os.listdir():
#         if filename.startswith('partial_index_') and filename.endswith('.json'):
#             total_size += os.path.getsize(filename)
#     return total_size / 1024  # Convert bytes to KB


# # Function to load the document ID to URL mapping
# def load_doc_id_url_mapping():
#     # Check if the mapping file exists
#     if os.path.exists("doc_id_to_url.json"):
#         try:
#             with open("doc_id_to_url.json", "r") as f:
#                 return json.load(f)
#         except Exception as e:
#             print(f"Error reading doc_id_to_url.json: {e}")
#             return {}
#     else:
#         print("doc_id_to_url.json does not exist. Creating a new file...")
#         # Create an empty mapping and save it to disk
#         empty_mapping = {}
#         save_doc_id_url_mapping(empty_mapping)
#         return empty_mapping


    
# # Function to write report data to a file
# def write_report(doc_count, unique_token_count, index_size_kb):
#     report = {
#         "Number of Indexed Documents": doc_count,
#         "Number of Unique Tokens": unique_token_count,
#         "Total Index Size (KB)": index_size_kb
#     }
#     with open("index_report.json", "w") as f:
#         json.dump(report, f, indent=2)
#     print("Report saved to index_report.json")

def load_full_index():
    full_index = {}
    for filename in os.listdir():
        if filename.startswith('partial_index_') and filename.endswith('.json'):
            with open(filename, 'r') as f:
                partial_index = json.load(f)
                for token, postings in partial_index.items():
                    if token not in full_index:
                        full_index[token] = postings
                    else:
                        full_index[token].extend(postings)
    return full_index



def main():
    # Load the full inverted index from partial files
    print("Loading inverted index...")
    inverted_index = load_full_index()
    print("Inverted index loaded.")
    wordcount_index = get_wordcount_dict(inverted_index)
    print("wordcount index loaded")

    # Load or generate document URLs
    with open('doc_id_to_url.json', 'r') as f:
        doc_urls = {int(k): v for k, v in json.load(f).items()}  # Ensure keys are integers

    total_docs = len(doc_urls)

    # Manual query input
    print("\n--- Manual Query Input ---")
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting manual query mode.")
            break

        # Process the manual query
        print(f"\nProcessing query: '{query}'...")
        ranked_results = calculate_tf_idf(query, inverted_index, total_docs, wordcount_index)

        # Display results
        if ranked_results:
            print("\nTop results:")
            for rank, (doc_id, score) in enumerate(ranked_results[:5], start=1):
                print(f"{rank}. {doc_urls.get(doc_id, f'Document {doc_id}')} (TF Score: {score})")
        else:
            print("No matching documents found.")



if __name__ == "__main__":
    main()