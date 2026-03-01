# nlp_module/components/text_cleaning.py

import re
import string
from typing import List

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))


def lowercase_text(text: str) -> str:
    """
    Convert text to lowercase.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from text.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text: str) -> str:
    """
    Remove digits/numbers from text.
    """
    return re.sub(r'\d+', '', text)


def remove_stopwords(text: str) -> str:
    """
    Remove English stopwords from text.
    """
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(filtered_tokens)


def clean_text(text: str) -> str:

    logging.info("Inside text cleaning")
    """
    Full text cleaning pipeline:
    1. Lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove stopwords
    """
    text = lowercase_text(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = text.strip()
    logging.info("completed text cleaning")
    return text


def clean_corpus(corpus: List[str]) -> List[str]:
    """
    Clean a list of texts using clean_text.
    """
    return [clean_text(doc) for doc in corpus]
