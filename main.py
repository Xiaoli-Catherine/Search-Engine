import sys
from pathlib import Path
import json
import io

import invert_index
import search
import time

def load_data(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
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
        # pre_load the data to memory
        lex_data = load_data("lexicon.json")
        url_data = load_data("indexed_doc.json")
        high_fre_term = load_data("high_fre_term.json")
        doc_len = load_data("doc_len.json")
        print("="*60)
        print("\nPlease into the query that you want to search\n")
        print("If you want to log out, please input: quit\n" )
        print("Please input your query: ")
        query = input()
        invert_index.logging.info(f"quesy: {query}")
        while query != "quit":
            search.search_doc(query, high_fre_term, url_data, lex_data, doc_len)
            print("="*60)
            print("\nPlease into the query that you want to search\n")
            print("If you want to log out, please input: quit\n" )
            print("Please input your query: ")
            query = input()