import sys
import json
from pathlib import Path

import invert_index
def read_invert_index(tokens, lex, result, docid_sets):
    file_path = Path("invert_index.jsonl")
    with file_path.open("rb") as f:
        for token in tokens:
            offset = lex[token]["offset"]
            length = lex[token]["length"]

            # find the term at its excatly position
            f.seek(offset)
            raw = f.read(length)
            line = raw.decode("utf-8").rstrip("\n")
            rec = json.loads(line) 
            postings = rec["postings"]
            docids = set(postings.keys())
            docid_sets[token]= docids
            result[token] = postings

    return result, docid_sets

def load_terms_from_index(tokens, high_fre_term):

    remain_tokens = []
    docid_sets = {}
    result = {}

    for token in tokens:
        if token in high_fre_term:
            postings = high_fre_term[token]["postings"]
            docids = set(postings.keys())
            docid_sets[token] = docids
            result[token] = postings
        else:
            remain_tokens.append(tokens)
    if remain_tokens:
        # load lexicon for term position 
        lex = {}
        with open("lexicon.json", "r", encoding="utf-8") as f:
            lex = json.load(f)
        result, docid_sets = read_invert_index(remain_tokens, lex, result, docid_sets)

    return result, docid_sets

def search_doc(query: str, high_fre_term: dict):
    tokens = invert_index.tokenize(query)
    term_postings, term_id_set = load_terms_from_index(tokens, high_fre_term)
