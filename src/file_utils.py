import os
import json

def load_data(folder_path):
    pages = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    pages.append(json.load(f))
    return pages

def save_doc_id_url_mapping(mapping, path="doc_id_to_url.json"):
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)

def load_doc_id_url_mapping(path="doc_id_to_url.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_progress(progress, path="progress.json"):
    with open(path, 'w') as f:
        json.dump(progress, f, indent=2)

def load_progress(path="progress.json"):
    if os.path.exists(path):
        with open(path, 'r') as f:
            progress = json.load(f)
            progress["unique_tokens"] = set(progress["unique_tokens"])
            return progress
    return {"doc_id": 0, "batch_number": 0, "unique_tokens": set()}
