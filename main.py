import sys
from pathlib import Path
import json

import invert_index
import search
import time

def load_high_fre_term(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        high_fre_term = json.load(f)
    return high_fre_term

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Need two text file as input")
        sys.exit(1)
    query = sys.argv[1]
    if (query == "index"):
        root = Path("DEV")
        invert_index.logging.info(f"root: {root}")
        inverted_index, docurl, doclen, dict_ids = invert_index.build_inverted_index(root)
        #save_index_json(inverted_index, docurl, doclen)
        invert_index.save_unique_doc( docurl)
        invert_index.save_doc_len(doclen)
        invert_index.save_unique_token()
        invert_index.merge_json_to_jsonl(docurl, dict_ids)
    else:
        query = " ".join(sys.argv[1:])
        invert_index.logging.info(f"quesy: {query}")
        high_fre_term = load_high_fre_term("high_fre_term.json")
        search.search_doc(query, high_fre_term)