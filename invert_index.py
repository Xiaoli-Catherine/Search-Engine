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

index = defaultdict(lambda: {"df": 0, "postings": defaultdict(lambda: {"ft": [], "ft-idf": [], "pos": []})})

def visible_text_from_soup(soup: BeautifulSoup) -> str:
    
    # Drop non-content tags
    # Remove JS/CSS/inert markup, Cuts menus/footers
    for t in soup(["script","style","noscript","template","svg","canvas", "nav", "header", "footer"]):
        t.decompose()

    # drop comments
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    # drop hidden elements (non-visible)
    for el in soup.select('[hidden], [aria-hidden="true"], [style*="display:none"], [style*="visibility:hidden"]'):
        el.decompose()

    # Preserve line breaks where it matters
    # Replacing <br> with a newline so we can keep word separated.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for blk in soup.find_all(["p","li","div","h1","h2","h3","h4","h5","h6"]):
        if blk.string is None:
            blk.append("\n")
    # Get visible text
    text = soup.get_text(" ", strip=True)
    text = text.lower()
    # logging.info(f"visitable text: {text}")
    return text

def tokenize(text: str)->List[str]:
    words = re.findall(r"\b[A-Za-z0-9]{2,}\b", text)
   
    # Create a Porter Stemmer instance
    porter_stemmer = PorterStemmer()
    # Apply stemming to each word
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    return stemmed_words
def get_soup(content):
    for parser in ("html5lib", "lxml", "html.parser"):
        try:
            return BeautifulSoup(content, parser)
        except Exception:
            continue
        

def extract_text(doc: dict) -> Tuple[str, str]:
    raw_url = doc.get("url") or ""
    try:
        # remove fragment for url
        url, _ = urldefrag(str(raw_url))
    except Exception:
        url = str(raw_url)
    content = doc.get("content")
    encoding = doc.get("encoding")
    if isinstance(content, str) and content.strip():
        soup = get_soup(content)
    else:
        soup = ""
    # soup = get_soup(content)
    if soup:  
        visible_text = visible_text_from_soup(soup)
        tokens = tokenize(visible_text)
    else: 
        tokens = ""
    return url, tokens, encoding

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
def build_inverted_index(root: Path):
    max_num = 1000 #flush the dict every 1000 doc

    docurl: Dict[int, str] = {}
    doclen: Dict[int, int] = {}
    dict_ids: List[int] = [] # store the id for every dict

    dict_id = 0  #identify the if for every dict
    doc_count = 0 # count the doc in a block
    docid = 0 #Every url have a unique document id

    seen_url = set() # set for already seen url
    seen_token = set() #set for already seen content

    def offload_dict(bin: int):
        global index
        
        if not index:
            return
        
        # save the dict to file
        file_name = f"dict{bin}.json"
        inverted_index = {}
        for term in sorted(index.keys()): #keep every dict same order
            inverted_index[term] = {
                "df": int(index[term]["df"]),
                "postings": index[term]["postings"]
        }
        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(inverted_index, f, indent=2, ensure_ascii=False)
        except Exception:
            pass 
        index.clear()
        dict_ids.append(bin)

    for file in root.rglob("*.json"): 
        doc = read_json_file(file)
        url, tokens, encoding = extract_text(doc)

        # Remove duplicates content and seen url
        if url in seen_url:
            continue
        # Normalizing tokens so same content have the same hashes
        content = hashlib.blake2b(" ".join(tokens).encode("utf-8"), digest_size=16).hexdigest()
        if content is None:
            continue
        if content in seen_token:
            continue

        seen_token.add(content)
        seen_url.add(url)

        docurl[docid] = url
        doclen[docid] = len(tokens)
        #store the positions for each term in this document
        terms= set()
        for pos, term in enumerate(tokens):
            index[term]["postings"][docid]["pos"].append(pos)
            terms.add(term)
       
        # update the document frequent for each unique term of this file
        # and update the ft for each term in this document
        for term in terms:
            index[term]["df"] += 1
            ft = len(index[term]["postings"][docid]["pos"])/doclen[docid]
            index[term]["postings"][docid]["ft"].append(ft)

        docid += 1
        doc_count += 1

        if doc_count >= max_num:
            offload_dict(dict_id)
            dict_id +=1
            doc_count = 0

    offload_dict(dict_id)

    """
    # convert defaultdict to plain dicts
    inverted_index = {}

    for term in sorted(index.keys()): #keep every dict same order
        # postings_map = index[term]["postings"]   # {docid: [pos...]}
        # sort docIDs and positions, and freeze as a list of pairs
        # postings_list = [[int(d), sorted(pos)]
                       # for d, pos in sorted(postings_map.items())]
        inverted_index[term] = {
            "df": int(index[term]["df"]),
            "postings": index[term]["postings"]
        }

    # logging.info(f"return index and url: {inverted_index}, {docurl}")  
    """
    return index, docurl, doclen
        

def save_index_json(inverted_index, docurl, doclen, file_name = "index_report.json"):
    logging.info("save index")
    
    # calulate the ft-idf
    for term in inverted_index:
        for doc_id in inverted_index[term]["postings"]:       
            idf = math.log(len(docurl)/inverted_index[term]["df"])
            tf = len(inverted_index[term]["postings"][doc_id]["pos"])/doclen[doc_id]
            ft_idf = tf * float(idf)
            inverted_index[term]["postings"][doc_id]["ft-idf"].append(ft_idf)


    report = {
       #"unique url": len(docurl),
        "unique_term": len(inverted_index),
        #"doc_len": doclen,
        #"url_id": docurl,
        "inverted_index": inverted_index
    }
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass 

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
def save_index_report():
    for i in range(14):
        file_name = f"dict{i}.json"
        try:
            with open(file_name, 'r', encoding='utf-8')as f:

        except Exception:
            pass

def main():
    root = Path("DEV")
    logging.info(f"root: {root}")
    inverted_index, docurl, doclen = build_inverted_index(root)
    # save_index_json(inverted_index, docurl, doclen)
    save_unique_doc( docurl)
    save_doc_len(doclen)
    #merge_dict()
    save_index_report()

if __name__ == "__main__":
    main()