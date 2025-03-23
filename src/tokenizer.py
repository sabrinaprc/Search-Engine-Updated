import os
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

ps = PorterStemmer()


# Tokenize and stem text, removing stop words
def tokenize(text):
    tokens = word_tokenize(text)
    return [ps.stem(token.lower()) for token in tokens if token.isalnum() and token.lower()]

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

