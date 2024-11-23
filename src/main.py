import os
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import math

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

# Parse content and tokenize it
def parse_and_tokenize(content):
    tokens = tokenize(content)
    return tokens

# Helper function to add tokens to the inverted index and track unique tokens
def add_to_index(inverted_index, tokens, doc_id, unique_tokens_set):
    for token in tokens:
        if token not in inverted_index:
            inverted_index[token] = []

        # If this is a new unique token across all batches, add it to unique_tokens_set
        if token not in unique_tokens_set:
            unique_tokens_set.add(token)
            print(f"Unique token count: {len(unique_tokens_set)}")  # Print updated count

        # Check if doc_id already exists for this token, if so, update frequency
        for entry in inverted_index[token]:
            if entry["doc_id"] == doc_id:
                entry["frequency"] += 1
                break
        else:
            inverted_index[token].append({"doc_id": doc_id, "frequency": 1})

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


def process_query(query, index):
    tokens = tokenize(query)
    if not tokens:
        return []
    
    doc_sets = []
    for token in tokens:
        if token in index:
            doc_sets.append(set(entry['doc_id'] for entry in index[token]))
        else:
            return []  # If any token is not in index, return empty result
    
    # Perform intersection to handle AND queries
    result_docs = set.intersection(*doc_sets) if doc_sets else set()
    return sorted(result_docs)

def calculate_tf_idf(query, index, total_docs):
    tokens = tokenize(query)
    doc_scores = {}
    
    for token in tokens:
        if token in index:
            idf = math.log(total_docs / (1 + len(index[token])))
            for entry in index[token]:
                tf = entry['frequency']
                score = tf * idf
                doc_id = entry['doc_id']
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += score
    
    # Sort documents by score in descending order
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

def search_interface(index, total_docs, doc_urls):
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        results = calculate_tf_idf(query, index, total_docs)
        print("\nTop 5 results:")
        for rank, (doc_id, score) in enumerate(results[:5], start=1):
            print(f"{rank}. {doc_urls[doc_id]} (Score: {score:.4f})")


def main():
    # Load the full inverted index from partial files
    print("Loading inverted index...")
    inverted_index = load_full_index()  # Function to load all partial indices
    print("Inverted index loaded.")

    # Assume document URLs are already mapped and saved
    # Create a mock mapping if you don't have URLs stored
    # Replace 'doc_id_to_url.json' with your actual file, if available
    if os.path.exists('doc_id_to_url.json'):
        with open('doc_id_to_url.json', 'r') as f:
            doc_urls = json.load(f)
    else:
        print("No document URL mapping found. Using placeholder URLs.")
        doc_urls = {doc_id: f"Document {doc_id}" for doc_id in range(1, 1001)}  # Example for 1000 docs

    # Total number of documents (adjust to match the indexed dataset)
    total_docs = len(doc_urls)

    # Define test queries
    test_queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering"
    ]

    # Test predefined queries and output results
    print("\nTesting predefined queries...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = calculate_tf_idf(query, inverted_index, total_docs)
        top_5_results = results[:5]
        
        print("Top 5 results:")
        for rank, (doc_id, score) in enumerate(top_5_results, start=1):
            print(f"{rank}. {doc_urls.get(doc_id, f'Document {doc_id}')} (Score: {score:.4f})")

    # Launch interactive search interface for custom queries
    print("\nEntering search interface...")
    search_interface(inverted_index, total_docs, doc_urls)

if __name__ == "__main__":
    main()