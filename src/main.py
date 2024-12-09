import os
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import math
from collections import defaultdict
from bs4 import BeautifulSoup
import time


# Ensure nltk resources are available
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to load all JSON files in a directory into a list
def load_data(folder_path):
    pages = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    page = json.load(f)
                    pages.append(page)
    return pages

# Tokenize and stem text, removing stop words
def tokenize(text):
    tokens = word_tokenize(text)
    return [ps.stem(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]

# Parse content and tokenize it, assigning weights to important tokens
def parse_and_tokenize(content):
    soup = BeautifulSoup(content, 'html.parser')

    tokens_with_weights = []

    # Extract tokens from the title and assign a higher weight
    title = soup.title.string if soup.title else ""
    if title:
        tokens_with_weights.extend((token, 3) for token in tokenize(title))  # Weight = 3

    # Extract tokens from headings and assign a medium weight
    for heading_tag in ['h1', 'h2', 'h3']:
        for heading in soup.find_all(heading_tag):
            tokens_with_weights.extend((token, 2) for token in tokenize(heading.get_text()))  # Weight = 2

    # Extract tokens from bold text and assign a lower weight
    for bold in soup.find_all(['b', 'strong']):
        tokens_with_weights.extend((token, 1.5) for token in tokenize(bold.get_text()))  # Weight = 1.5

    # Extract tokens from the rest of the content with normal weight
    body = soup.get_text()
    tokens_with_weights.extend((token, 1) for token in tokenize(body))

    return tokens_with_weights

# Helper function to add tokens to the inverted index and track unique tokens
def add_to_index(inverted_index, tokens_with_weights, doc_id, unique_tokens_set):
    for token, weight in tokens_with_weights:
        if token not in inverted_index:
            inverted_index[token] = []

        # If this is a new unique token across all batches, add it to unique_tokens_set
        if token not in unique_tokens_set:
            unique_tokens_set.add(token)
            print(f"Unique token count: {len(unique_tokens_set)}")  # Print updated count

        # Check if doc_id already exists for this token, if so, update frequency with weight
        for entry in inverted_index[token]:
            if entry["doc_id"] == doc_id:
                entry["frequency"] += weight
                break
        else:
            inverted_index[token].append({"doc_id": doc_id, "frequency": weight})

# Function to write the partial index to disk and clear the in-memory index
def sort_and_write_to_disk(index, filename):
    sorted_index = {token: sorted(postings, key=lambda x: x['doc_id']) for token, postings in index.items()}
    with open(filename, 'w') as f:
        json.dump(sorted_index, f, indent=2)
    print(f"Partial index saved to {filename}")

# Save progress to a file, including unique tokens
def save_progress(doc_id, batch_number, unique_tokens_set):
    progress = {
        "doc_id": doc_id,
        "batch_number": batch_number,
        "unique_tokens_count": len(unique_tokens_set),  # Save the count of unique tokens
        "unique_tokens": list(unique_tokens_set)  # Convert set to list for JSON serialization
    }
    with open('progress.json', 'w') as f:
        json.dump(progress, f)
    print(f"Progress saved!")

# Load progress from a file, including unique tokens
def load_progress():
    if os.path.exists('progress.json'):
        with open('progress.json', 'r') as f:
            progress = json.load(f)
            progress["unique_tokens"] = set(progress["unique_tokens"])  # Convert list back to set
            return progress
    return {"doc_id": 0, "batch_number": 0, "unique_tokens": set()}

# Function to calculate the total size of all partial index files in KB
def calculate_index_size():
    total_size = 0
    for filename in os.listdir():
        if filename.startswith('partial_index_') and filename.endswith('.json'):
            total_size += os.path.getsize(filename)
    return total_size / 1024  # Convert bytes to KB

# Function to map document IDs to URLs
def save_doc_id_url_mapping(doc_id_url_map):
    try:
        with open("doc_id_to_url.json", "w") as f:
            json.dump(doc_id_url_map, f, indent=2)
        print("Document ID to URL mapping saved to doc_id_to_url.json.")
    except Exception as e:
        print(f"Error writing doc_id_to_url.json: {e}")


# Function to load the document ID to URL mapping
def load_doc_id_url_mapping():
    # Check if the mapping file exists
    if os.path.exists("doc_id_to_url.json"):
        try:
            with open("doc_id_to_url.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading doc_id_to_url.json: {e}")
            return {}
    else:
        print("doc_id_to_url.json does not exist. Creating a new file...")
        # Create an empty mapping and save it to disk
        empty_mapping = {}
        save_doc_id_url_mapping(empty_mapping)
        return empty_mapping

# def process_query_boolean_with_tf(query, inverted_index):
#     print(f"\nProcessing Query: {query}")
#     query_tokens = tokenize(query)
#     print(f"Query tokens: {query_tokens}")

#     # Retrieve posting lists for each token
#     posting_lists = []
#     doc_scores = {}  # Dictionary to store term frequency scores for documents
#     for token in query_tokens:
#         if token in inverted_index:
#             postings = inverted_index[token]
#             doc_ids = set(entry['doc_id'] for entry in postings)
#             posting_lists.append(doc_ids)

#             # Update scores based on term frequency
#             for entry in postings:
#                 doc_id = entry['doc_id']
#                 frequency = entry['frequency']
#                 if doc_id not in doc_scores:
#                     doc_scores[doc_id] = 0
#                 doc_scores[doc_id] += frequency
#         else:
#             print(f"No postings found for token '{token}'")

#     # Combine posting lists based on the query type
#     if posting_lists:
#         result_docs = set.intersection(*posting_lists)  # Logical AND

#     # Filter and rank documents by term frequency score
#     ranked_results = sorted(
#         [(doc_id, doc_scores[doc_id]) for doc_id in result_docs],
#         key=lambda x: x[1],
#         reverse=True
#     )

#     return ranked_results

    
# Function to write report data to a file
def write_report(doc_count, unique_token_count, index_size_kb):
    report = {
        "Number of Indexed Documents": doc_count,
        "Number of Unique Tokens": unique_token_count,
        "Total Index Size (KB)": index_size_kb
    }
    with open("index_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report saved to index_report.json")

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