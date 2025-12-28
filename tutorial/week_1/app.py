import os
import re
from typing import List

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from transformers import AutoTokenizer

# Load the pre-trained BPE tokenizer
tokenizer = AutoTokenizer.from_pretrained("p208p2002/llama-traditional-chinese-120M")


def _preprocess_func(text: str):
    # 1. Define special tokens to remove
    special_tokens = {"<s>", "</s>", "[PAD]", "[UNK]"}

    encoded = tokenizer(text)

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    # 2. Remove special tokens
    tokens = [t.replace("▁", "") for t in tokens if t not in special_tokens]

    # 3. Remove punctuation (keep only Chinese/English/number words)
    tokens = [t for t in tokens if re.match(r'[\w一-龥]+', t)]

    # Stringify the tokens
    return [str(token) for token in tokens]


# Read file
def load_poem():
    filename = "唐詩三百首.txt"
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    poems = []

    # Split by blank lines
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]

    for block in blocks:
        entry = {}
        for line in block.split("\n"):
            if line.startswith("詩名:"):
                entry["詩名"] = line.replace("詩名:", "").strip()
            elif line.startswith("作者:"):
                entry["作者"] = line.replace("作者:", "").strip()
            elif line.startswith("詩體:"):
                entry["詩體"] = line.replace("詩體:", "").strip()
            elif line.startswith("詩文:"):
                entry["詩文"] = line.replace("詩文:", "").strip()
        if len(entry) != 0:
            poems.append(entry)

    return poems


def build_documents(poems: List) -> List:

    df_poem = pd.DataFrame(poems)

    documents = []

    for _, row in df_poem.iterrows():
        document = Document(page_content=row['詩文'],
                            metadata={"詩名": row["詩名"],
                                      "作者": row["作者"],
                                      "詩體": row["詩體"]})
        documents.append(document)

    return documents


def build_retriever(k: int):

    poems = load_poem()
    documents = build_documents(poems)

    bm25_poem_retriever = BM25Retriever.from_documents(documents, k=k,
                                                       bm25_params={"k1": 2.5},
                                                       preprocess_func=_preprocess_func
                                                       )