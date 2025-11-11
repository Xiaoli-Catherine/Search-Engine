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
import argparse

import logging

fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh = logging.FileHandler("scraper.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[ch, fh], force=True)



# index: term -> {"df": int, "postings": {docid: [pos, ...]}}
IndexType = Dict[str, Dict[str, dict]]

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

    # Get visible text
    text = soup.get_text(" ", strip=True)
    text = text.lower()
    logging.info(f"visitable text: {text}")
    return text

def tokenize(text: str)->List[str]:
    words = re.findall(r"[A-Za-z0-9]+", text)
    # Stemming word
    # Create a Porter Stemmer instance
    porter_stemmer = PorterStemmer()
    # Apply stemming to each word
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    return stemmed_words

def extract_text(doc: dict) -> Tuple[str, str]:
    raw_url = doc.get("url") or ""
    try:
        # remove fragment for url
        url, _ = urldefrag(str(raw_url))
    except Exception:
        url = str(raw_url)
    content = doc.get("content")
    encoding = doc.get("encoding")
    soup = BeautifulSoup(content, "html.parser")
    visible_text = visible_text_from_soup(soup)
    tokens = tokenize(visible_text)
    return url, tokens  
def read_json_file(file:Path):
    try: 
        with file.open("r", encoding="utf-8", errors="ignore") as f:
            #doc = list(file.rglob("*.json"))
            doc = json.load(f) 
        logging.info(f"file: {file}")
       # logging.info(f"doc: {doc}")
        return doc
            
    except TypeError:
        print ("TypeError for open json file ", {file})
        raise
def build_inverted_index(root: Path):
    docid = 0 #Every url have a unique document id
    index = defaultdict(lambda: {"df": 0, "postings": defaultdict(list)})
    docurl: Dict[int, str] = {}

    for file in root.rglob("*.json"): 
        docid += 1
        doc = read_json_file(file)
        url, tokens = extract_text(doc)
        docurl[docid] = url

        #store the positions for each term in this document
        terms= set()
        for pos, term in enumerate(tokens):
            index[term]["postings"][docid].append(pos)
            terms.add(term)
        # update the document frequent for each unique term of this file
        for term in terms:
            index[term]["df"] += 1
    # convert defaultdict to plain dicts
    inverted_index = {}
    for term in sorted(index.keys()):
        inverted_index[term] = {
            "df": int(index[term]["df"]),
            "postings": index[term]["postings"]
        }
        
    return inverted_index, docurl
        

def save_index_json(inverted_index, docurl, file_name = "index_report.json"):
    terms = {}
    for term, data in inverted_index.items():
        df = data["df"]
        pos_list = [[int(id), pos] for id, pos in data["postings"].items()]
        terms[term] = [df, pos_list]
    report = {
        "url_id": docurl,
        "terms": terms
    }
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass 

"""
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).parent / "DEV" / "aiclub_ics_uci_edu"),
                   help="Path to dataset root containing JSON files")
    p.add_argument("--out", default="index.json", help="Output index JSON file")
    return p.parse_args()
"""

def main():

    # args = parse_args()
    # root = Path(args.root)

    root = Path("DEV/aiclub_ics_uci_edu")
    logging.info(f"root: {root}")
    inverted_index, docurl = build_inverted_index(root)
    save_index_json(inverted_index, docurl)

if __name__ == "__main__":
    main()