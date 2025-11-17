import sys
import json
from pathlib import Path

import invert_index
def read_invert_index(tokens, lex, result, docid_lists):
    invert_index.logging.info("inside the read_invert_index")
    file_path = Path("invert_index.jsonl")
    with file_path.open("rb") as f:
        invert_index.logging.info("open the file")
        for token in tokens:
            if token in lex:
                offset = lex[token]["offset"]
                length = lex[token]["length"]
                invert_index.logging.info(f"token: {token}")
                invert_index.logging.info(f"offset: {offset}")

                # find the term at its excatly position
                f.seek(offset)
                raw = f.read(length)
                line = raw.decode("utf-8").rstrip("\n")
                rec = json.loads(line) 
                postings = rec["postings"]
                docids = sorted(int(d) for d in postings.keys()) #change the docid to int
               # docids = list(postings.keys())
                docid_lists.append((token, docids))
                result[token] = postings
            else:
                print("no such term in the storage")

    return result, docid_lists

def load_terms_from_index(tokens):
    # remain_tokens = []
    lex = {}
    with open("lexicon.json", "r", encoding="utf-8") as f:
        lex = json.load(f)

    docid_lists = [] 
    result = {}
        
    result, docid_lists = read_invert_index(tokens, lex, result, docid_lists)
    #sorted the docid list, so fewer docs come first
    sorted_docid_lists = sorted(docid_lists, key=lambda pair: len(pair[1]))
    return result, sorted_docid_lists

    """
    for token in tokens:
        if token in high_fre_term:
            postings = high_fre_term[token]["postings"]
            docids = [postings.keys()]
            docid_lists.append ((token, docids))
            result[token] = postings
        else:
            remain_tokens.append(tokens)
    if remain_tokens:
        # load lexicon for term position 
        lex = {}
        with open("lexicon.json", "r", encoding="utf-8") as f:
            lex = json.load(f)
        result, docid_lists = read_invert_index(remain_tokens, lex, result, docid_lists)
    #sorted the docid list, so fewer docs come first
    sorted_docid_lists = docid_lists.sort(key=lambda pair: len(pair[1]))
    return result, sorted_docid_lists
    """

def match_two_lists(listA, listB):
    i = 0
    j = 0
    result = []
    print(f"listA: {listA}")
    print(f"listB: {listB}")
    while i < len(listA) and j < len(listB):
        if listA[i] == listB[j]:
            result.append(listA[i])
            i += 1
            j += 1
        elif listA[i] < listB[j]:
            i += 1
        else:
            j += 1
    return result

def find_posible_id(term_id_list):
    if not term_id_list:
        return []
    term, current = term_id_list[0]

    for _, ls in term_id_list[1:]:
        current = match_two_lists(current, ls)

        if not current: # no common docid
            break
    return current
def search_doc(query: str):
    tokens = invert_index.tokenize(query)
    invert_index.logging.info(f"tokens: {tokens}")
    term_postings, term_id_list = load_terms_from_index(tokens)
    # invert_index.logging.info(f"term_id_list: {term_id_list}")
    
    posible_id = find_posible_id(term_id_list)
    print(f"posible_id: {posible_id}")
    if len(posible_id) > 5:
        posible_id = posible_id[:5]

    url_file = "indexed_doc.json"
    with open(url_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    id_url = data["id_url"]
    url_list = []
    for docid in posible_id:
        doc_url = id_url[str(docid)]
        url_list.append(doc_url)
    print(f"top 5 url is: {url_list}")



    
    

