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
