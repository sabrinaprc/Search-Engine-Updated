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

# Main function to build the index in batches
def main():
    # Define paths for ANALYST and DEV folders
    analyst_folder = 'assets/ANALYST'
    dev_folder = 'assets/DEV'
    
    # Load data from both folders
    analyst_pages = load_data(analyst_folder)
    dev_pages = load_data(dev_folder)
    all_pages = analyst_pages + dev_pages  # Combine both lists for batch processing
    
    batch_size = 100  # Define batch size
    inverted_index = {}  # Initialize the in-memory index
    unique_tokens_set = set()  # Track unique tokens across all batches
    doc_id = 0  # Document counter to track doc IDs across batches
    
    while all_pages:
        # Get the next batch of documents
        batch = all_pages[:batch_size]
        all_pages = all_pages[batch_size:]  # Remove processed batch from list

        # Process each document in the batch
        for page in batch:
            doc_id += 1
            if 'content' in page:
                tokens = parse_and_tokenize(page['content'])
                add_to_index(inverted_index, tokens, doc_id, unique_tokens_set)
            else:
                print(f"Error: Content not found in document ID {doc_id}")

        # Write the current batch's index to disk
        filename = f'partial_index_{doc_id // batch_size}.json'  # Use batch number for filename
        sort_and_write_to_disk(inverted_index, filename)
        inverted_index.clear()  # Clear the index from memory for the next batch

    print("Indexing complete.")
    print(f"Total unique tokens: {len(unique_tokens_set)}")

if __name__ == "__main__":
    main()
    