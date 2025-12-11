"""
Implement a simple Indexer
• Traversing folders and reading JSON
• Opening and reading one file at a time
• Parsing (dealing with broken HTML!)
• Tokenization & stemming
• Simple in-memory inverted index
• Simple index serialization to disk

"""
import re
import os
import io
import sys
from bs4 import BeautifulSoup, Comment
from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Dict, Iterable, Iterator, List, Tuple
import nltk
from nltk.stem import PorterStemmer
from urllib.parse import urldefrag
import hashlib
import math

import logging

fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh = logging.FileHandler("invert_index.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[ch, fh], force=True)

# index: term -> {"df": int, "postings": {docid: [pos, ...]}}
IndexType = Dict[str, Dict[str, dict]]

index = defaultdict(lambda: {"df": 0, "postings": defaultdict(lambda: {"pos": [], "tf": 0.0, "tf-idf": 0.0, "wt":0.0})})
dict_ids = [] # store the id for every dict
seen_url = set() # set for already seen url
seen_token = set() #set for already seen content
unique_token = set()

file_size = 0

def clean_soup(soup: BeautifulSoup):
    """
    Find visible text from soup by Drop non-content tags, 
    comments and hidden element
    """
    
    # Drop non-content tags
    # Remove JS/CSS/inert markup, Cuts menus/footers
    for t in soup(["script","style","noscript","template","svg","canvas", "nav", "header", "footer"]):
        t.decompose()

    # drop comments
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    # drop hidden elements (non-visible)
    for el in soup.select('[hidden], [aria-hidden="true"],[style*="display:none"], [style*="display: none"], [style*="visibility: hidden"], [style*="visibility:hidden"]'):
        el.decompose()
    
    # Preserve line breaks where it matters
    # Replacing <br> with a newline so we can keep word separated.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    

    # # Get visible text
    # text = soup.get_text(separator=" ", strip=True)
    # text = text.lower()
    # # logging.info(f"visitable text: {text}")
    # return text

def tokenize(text: str)->List[str]:
    text = text.lower()
    words = re.findall(r"\b[A-Za-z0-9]{2,}\b", text)
   
    # Create a Porter Stemmer instance
    porter_stemmer = PorterStemmer()
    # Apply stemming to each word
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    return stemmed_words  

def get_weight(text_node) -> int:
    weight = 0 # default body

    for parent in text_node.parents:
        if not getattr(parent, "name", None):
            continue
        name = parent.name.lower()

        if name == "title":
            weight = 3
        elif name in ("h1", "h2", "h3"):
            weight = 2
        elif name == "strong":
            weight = 1
    return weight

def weighted_tokens_from_soup(soup: BeautifulSoup):
    clean_soup(soup)

    pos = 0
    result: list[tuple[str, int, int]] = []

    for text_node in soup.find_all(string = True):
        # skip empty text
        if not text_node.strip():
            continue
        weight = get_weight(text_node)
        tokens = tokenize(str(text_node))

        for token in tokens:
            result.append((token, pos, weight))
            pos += 1
    return result


def extract_text(doc: dict):
    raw_url = doc.get("url") or ""
    try:
        # remove fragment for url
        url, _ = urldefrag(str(raw_url))
    except Exception:
        url = str(raw_url)
    content = doc.get("content")
    encoding = doc.get("encoding")
    if isinstance(content, str) and content.strip():
        # use html5lib for break html
        soup = BeautifulSoup(content, "html5lib")
    else:
        soup = ""
    # soup = get_soup(content)
    if soup:  
        weighted_tokens = weighted_tokens_from_soup(soup)
        # visible_text = visible_text_from_soup(soup)
       # tokens = tokenize(visible_text)
    else: 
        weighted_tokens = []
        #tokens = ""
    #logging.info(f"tokens: {tokens}")
    return url, weighted_tokens, encoding

def read_json_file(file:Path):
    try: 
        with file.open("r", encoding="utf-8", errors="ignore") as f:
            #doc = list(file.rglob("*.json"))
            doc = json.load(f) 
        # logging.info(f"file: {file}")
       # logging.info(f"doc: {doc}")
        return doc
            
    except TypeError:
        print ("TypeError for open json file ", {file})
        raise

def sort_index(index):
    inverted_index = {}
    for term in sorted(index.keys()): #keep every dict same order
        postings = index[term]["postings"]
        # sorted by docid key: docid, value: {"pos","tf"...}
        sorted_postings = dict(sorted(postings.items(),
                                      key=lambda item: (int(item[0]))))                 
        inverted_index[term] = {
            "df": int(index[term]["df"]),
            "postings": sorted_postings
    }
    return inverted_index
"""
*****************************
offload partial index to disk
*****************************
"""
def offload_dict(num: int):
        global index, dict_ids
        
        if not index:
            return
        
        inverted_index = sort_index(index)
        # store partial index to file
        file_name = f"dict{num}.json"       
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f,indent=2, ensure_ascii=False)
       
        index.clear()
        dict_ids.append(num)


def remove_duplicate(url, tokens):
    # Remove duplicates content and seen url
    if url in seen_url:
        return True
    # Normalizing tokens so same content have the same hashes so I can add to a set
    content = hashlib.blake2b(" ".join(tokens).encode("utf-8"), digest_size=16).hexdigest()
    
    if content is None:
        True
    if content in seen_token:
        True
    return False


def build_inverted_index(root: Path):
    max_num = 1500 #flush the dict every 1000 doc

    docurl: Dict[int, str] = {}
    doclen: Dict[int, int] = {}

    dict_id = 0  #identify the if for every dict
    doc_count = 0 # count the doc in a block
    docid = 0 #Every url have a unique document id

    for file in root.rglob("*.json"): 
        logging.info(f"file: {file}")
        doc = read_json_file(file)
        url, weighted_tokens, encoding = extract_text(doc)

        if not weighted_tokens:
            continue

        tokens_for_hash = [t for (t, _, _) in weighted_tokens]
        # Normalizing tokens so same content have the same hashes so I can add to a set
        content = hashlib.blake2b(" ".join(tokens_for_hash).encode("utf-8"), digest_size=16).hexdigest()
        
        # Remove duplicates content and seen url
        if remove_duplicate(url, content):
            continue

        seen_token.add(content)
        seen_url.add(url)

        docurl[docid] = url
        doclen[docid] = len(tokens_for_hash)
        #store the positions for each term in this document
        terms_in_doc= set()
        for term, pos, weight in weighted_tokens:
            index[term]["postings"][docid]["pos"].append(pos)
            index[term]["postings"][docid]["wt"] +=  weight
            terms_in_doc.add(term)
            unique_token.add(term)
       
        # update the document frequent for each unique term of this file
        # and update the tf for each term in this document
        for term in terms_in_doc:
            index[term]["df"] += 1
            # raw_wt = index[term]["postings"][docid]["wt"]
            tf_row = len(index[term]["postings"][docid]["pos"])
            if tf_row > 0:
                tf = 1+math.log(tf_row)
            else:
                tf = 0.0
            # tf = len(index[term]["postings"][docid]["pos"])/doclen[docid]
            index[term]["postings"][docid]["tf"] = tf

        docid += 1
        doc_count += 1

        if doc_count >= max_num:
            offload_dict(dict_id)
            dict_id +=1
            doc_count = 0

    offload_dict(dict_id)
    return index, docurl, doclen, dict_ids
        


"""
*********************************
Merging partial indexes
*********************************
"""


def merge_two_files(file_a, file_b, out_file):
    a = read_json_file(file_a)
    b = read_json_file(file_b)
    logging.info(f"merge two file: {file_a}, and {file_b}")
    merged = {}
    all_terms = sorted(a.keys() | b.keys())

    for term in all_terms:
        postings_a = a.get(term, {}).get("postings", {})
        postings_b = b.get(term, {}).get("postings", {})

        merge_postings = postings_a
        merge_postings.update(postings_b)
        df = len(merge_postings)

        # sort the postings by id
        sorted_postings = dict(sorted(merge_postings.items(),
                                      key=lambda item: (int(item[0]))))
         
        merged[term] = {
            "df": df,
            "postings": sorted_postings
        }
    with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

def merge_json(dict_ids):
    round = 0
    current_files = []
    for num in dict_ids:
        current_files.append(Path(f"dict{num}.json"))
    logging.info(f"current_file: {current_files}")
    while len(current_files)>1:
        new_files = []
        for i in range(0, len(current_files), 2):
            if i + 1 == len(current_files):
                new_files.append(current_files[i])
            else:
                a = current_files[i]
                b = current_files[i+1]
                out_file = Path(f"merged_round{round}_{i}.json")
                merge_two_files(a, b, out_file)
                new_files.append(out_file)
        current_files = new_files
        round += 1
    final_json = current_files[0]
    return final_json

   
def merge_json_to_jsonl(docurl, dict_ids):
    final_json = merge_json(dict_ids)
    merged_index = read_json_file(final_json)

    N = len(docurl)
    lex = {}
    offset = 0
    high_fre_term=defaultdict(lambda:{"df": 0, "postings":{}})

    jsonl_file = Path("invert_index.jsonl")
    with jsonl_file.open("wb") as wf:
        for term, value in merged_index.items():
            postings = value["postings"]
            df = len(postings)   
            value["df"] = df

            idf = math.log(N/df) 

            for docid, posting in postings.items():
                tf = posting["tf"]
                # if tf_list:
                #     tf = tf_list[0]
                # else:
                #     tf = 0.0
                posting["tf-idf"] = tf * idf
            #record for jsonl file
            rec = {
                "t": term,
                "df": df,
                "postings": postings,
            }

            # store the data for high fre term
            if df >1000:
                # skip any token that contains a digit anywhere
                if any(ch.isdigit() for ch in term):
                    continue
                high_fre_term[term]["df"] =df
                high_fre_term[term]["postings"] = postings 

            # write the data to jsonl
            line = json.dumps(rec, ensure_ascii= False)
            data = (line + "\n").encode("utf-8")
            wf.write(data)

            #store data for lexicon search, 
            #so we can quickly find the location of a term
            lex[term] = {
                "df": df,
                "offset": offset,
                "length": len(data),
            }
            offset += len(data)  #increace the offset (location)
    lex_file ="lexicon.json"
    with open(lex_file, 'w', encoding='utf-8') as f:
        json.dump(lex, f, indent=2, ensure_ascii=False)
      
    high_fre_file = "high_fre_term.json"
    with open(high_fre_file, 'w', encoding='utf-8') as f:
        json.dump(dict(high_fre_term), f, indent=2, ensure_ascii=False)
    logging.info(f"length is: {len(high_fre_term)}")
    logging.info(f"high_fre_term: {high_fre_term.keys()}")
  
"""
*********************************
Save to file
*********************************
"""

def save_unique_doc(docurl, file_name = "indexed_doc.json"):  
    report = {
        "unique url": len(docurl),
        "id_url": docurl
    }
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass 

def save_doc_len(doclen, file_name = "doc_len.json"):
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(doclen, f, indent=2, ensure_ascii=False)
    except Exception:
        pass 

def save_unique_token(file_name = "unique_token.txt"):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(f"total unique tokens {len(unique_token)} \n")
        for token in unique_token:
            f.write(token + "  ")