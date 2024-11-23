import os
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

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


def main():
    # Define paths for ANALYST and DEV folders
    analyst_folder = 'assets/ANALYST'
    dev_folder = 'assets/DEV'
    
    print("Loading data from folders...")
    # Load data from both folders
    analyst_pages = load_data(analyst_folder)
    dev_pages = load_data(dev_folder)
    all_pages = analyst_pages + dev_pages  # Combine both lists for batch processing
    print(f"Loaded {len(all_pages)} documents in total.")

    # Load or initialize document ID-to-URL mapping
    print("Loading document ID to URL mapping...")
    doc_id_url_map = load_doc_id_url_mapping()
    print(f"Loaded {len(doc_id_url_map)} existing mappings.")

    # Load progress if it exists
    print("Checking for existing progress...")
    progress = load_progress()
    last_doc_id = progress["doc_id"]
    batch_number = progress["batch_number"]
    unique_tokens_set = progress["unique_tokens"]
    print(f"Last processed document ID: {last_doc_id}, Batch number: {batch_number}, Unique tokens: {len(unique_tokens_set)}")

    # Initialize counters and data structures
    batch_size = 100  # Define batch size
    inverted_index = {}  # Initialize the in-memory index
    doc_id = last_doc_id  # Start from the last processed doc_id

    # Resume processing from the next unprocessed batch
    remaining_pages = all_pages[doc_id:]
    if remaining_pages:
        print(f"Starting to process {len(remaining_pages)} remaining documents...")
    else:
        print("No new documents to process. Indexing is up-to-date.")

    while remaining_pages:
        # Get the next batch of documents
        batch = remaining_pages[:batch_size]
        remaining_pages = remaining_pages[batch_size:]  # Remove processed batch from list
        print(f"Processing batch {batch_number}...")

        # Process each document in the batch
        for page in batch:
            doc_id += 1
            url = page.get("url", f"Document {doc_id}")  # Default to "Document {doc_id}" if no URL
            doc_id_url_map[str(doc_id)] = url  # Save the mapping
            print(f"Processing Document ID {doc_id}, URL: {url}")

            if 'content' in page:
                # Tokenize and index the content
                tokens = parse_and_tokenize(page['content'])
                add_to_index(inverted_index, tokens, doc_id, unique_tokens_set)
                print(f"Indexed {len(tokens)} tokens for Document ID {doc_id}.")
            else:
                print(f"Warning: Content not found for Document ID {doc_id}.")

        # Write the current batch's index to disk
        filename = f'partial_index_{batch_number}.json'
        print(f"Writing partial index for batch {batch_number} to {filename}...")
        sort_and_write_to_disk(inverted_index, filename)
        inverted_index.clear()  # Clear the index from memory for the next batch
        print(f"Batch {batch_number} processing complete.")

        # Save progress and document ID to URL mapping after each batch
        batch_number += 1
        save_progress(doc_id, batch_number, unique_tokens_set)
        save_doc_id_url_mapping(doc_id_url_map)

    # Calculate report metrics
    doc_count = doc_id  # Total number of indexed documents
    unique_token_count = len(unique_tokens_set)  # Total number of unique tokens
    index_size_kb = calculate_index_size()  # Total size of index files in KB

    # Write the report to a file
    print("Generating final report...")
    write_report(doc_count, unique_token_count, index_size_kb)
    print("Report saved.")

    # Final summary of the process
    print("Indexing complete.")
    print(f"Total indexed documents: {doc_count}")
    print(f"Total unique tokens: {unique_token_count}")
    print(f"Total index size: {index_size_kb:.2f} KB")
    print(f"Document ID to URL mappings saved to doc_id_to_url.json.")


if __name__ == "__main__":
    main()
