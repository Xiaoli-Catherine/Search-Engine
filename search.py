import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import math
import time

import invert_index


"""
******************************
read invert index from files
******************************
"""
def read_invert_index(tokens, lex, result, docid_lists, df):
    """
    Read the term's data from the disk file invert_index.jsonl
    """
   # invert_index.logging.info("inside the read_invert_index")
    file_path = Path("invert_index.jsonl")
    with file_path.open("rb") as f:
       # invert_index.logging.info("open the file")
        for token in tokens:
            if token in lex:
                offset = lex[token]["offset"]
                length = lex[token]["length"]
               # invert_index.logging.info(f"token: {token}")
               # invert_index.logging.info(f"offset: {offset}")

                # find the term at its excatly position
                f.seek(offset)
                raw = f.read(length)
                line = raw.decode("utf-8").rstrip("\n")
                rec = json.loads(line) 
                postings = rec["postings"]
                docids = sorted(int(d) for d in postings.keys()) #change the docid to int
                docid_lists.append((token, docids))
                result[token] = postings
                df[token] = rec["df"]
            else:
                print("no such term in the storage")

    return result, docid_lists, df

def load_terms_from_index(tokens, high_fre_term, lex_data):
    """
    Try to find the terms from the in-memory dict: high_fre_term first
    Then, looking for the remaining tokens in the disk file
    """
    
    #lex = {}
    # with open("lexicon.json", "r", encoding="utf-8") as f:
    #     lex = json.load(f)
    remain_tokens = []
    docid_lists = [] 
    df = {}
    result = {}
        
    # result, docid_lists = read_invert_index(tokens, lex, result, docid_lists)
    # #sorted the docid list, so fewer docs come first
    # sorted_docid_lists = sorted(docid_lists, key=lambda pair: len(pair[1]))
    # return result, sorted_docid_lists

    for token in tokens:
        if token in high_fre_term:
            df[token] = high_fre_term[token]["df"]
            postings = high_fre_term[token]["postings"]
            docids = sorted(int(d) for d in postings.keys()) #change the docid to int
            docid_lists.append ((token, docids))
            result[token] = postings
        else:
            remain_tokens.append(token)
    print(f"remain_tokens: {remain_tokens}")
    if remain_tokens:
        result, docid_lists, df = read_invert_index(remain_tokens, lex_data, result, docid_lists, df)
    #sorted the docid list, so fewer docs come first
    sorted_docid_lists = sorted(docid_lists, key=lambda pair: len(pair[1]))
    return result, sorted_docid_lists, df
    
"""
******************************
Boolean search
******************************
"""

def match_two_lists(listA, listB):
    """
    Compare two list to find the common id
    """
    i = 0
    j = 0
    result = []
   # print(f"listA: {listA}")
   # print(f"listB: {listB}")
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
    """
    The term_id_list include id_list for each token.
    Compare each id for all token to find the posible_id 
    which all tokens match it.
    """
    print("inside find_posible_id")
    if not term_id_list:
        return []
    term, current = term_id_list[0]

    for _, ls in term_id_list[1:]:
        current = match_two_lists(current, ls)

        if not current: # no common docid
            break
    # print(f"current: {current}")
    return current

def sort_by_tf_idf(posible_id, tokens, term_postings):
    """
    sum up the tf-idf for each token for each page
    sorted the posible_id by the score of tf-idf, 
    """
    # init scores
    scores = {doc_id: 0.0 for doc_id in posible_id}

    #computer the sum tf-idf score for all posible-id that given the query
    for doc_id in posible_id:
        for token in tokens:
            if token in term_postings:
                postings = term_postings[token]
                doc = str(doc_id)
                tf_idf = postings[doc]["tf-idf"]
                scores[doc_id] += tf_idf

    
    #sorted the id so order in decreasing order
    sorted_id = sorted(posible_id, key=lambda doc_id: scores[doc_id], reverse = True)
    return sorted_id

"""
******************************
Ranked search
******************************
"""
def consine_score(term_postings, query, n, df):
    
    tf_q = Counter(query)
    scores = defaultdict(float)  
    doc_norm = defaultdict(float)
    
   
    # accumulate doc product: sum_t:  tf_idf_q * tf_idf_d
    for token, tf in tf_q.items():      
        if token not in term_postings:
            continue
        tf_idf_q = (1+math.log(tf))*math.log(n/df[token])
        # at least one term happened in the document
        postings = term_postings[token]       
        for d_key, info in postings.items():
            tf_idf_d = info["tf-idf"]
            # d = int(d_key)
            scores[d_key] += tf_idf_q * tf_idf_d
            doc_norm[d_key] += tf_idf_d * tf_idf_d
        
    for d in doc_norm:
        doc_norm[d] = math.sqrt(doc_norm[d])
    for d in scores:
        scores[d] = scores[d]/doc_norm[d]

    # scores = sorted(scores, key=lambda d: scores[d], reverse = True)
    return scores

def get_id_list(term_id_list):
    id_list = set()
    for key, value in term_id_list:
        for v in value:
            id_list.add(v)
    return id_list

"""
********************
Sum of tfidf * (1+ field_weight)
********************
"""
def sum_of_tf_idf(term_id_list, term_postings, query):
    unique_q = set(query)
    sum_of_tfidf_weight = defaultdict(float)
    # id_list = get_id_list(term_id_list)
    for token in unique_q:      
        if token not in term_postings:
            continue
        postings = term_postings[token]       
        for d_key, info in postings.items():
            tfidf = info["tf-idf"] 
            weight = info["wt"]
            sum_of_tfidf_weight[d_key] += tfidf * (1 + weight)
    return sum_of_tfidf_weight   

"""
************************
bata * sum_tfidf + (1-bata)* cos_score, 
set bata = 0.5 to give equal weight to term frequency 
(sum of tf-idf) and vector similarity (cosine) 
************************
"""    
def calculate_balance_weight(cos_score, sum_tfidf):
    bata = 0.5
    weight_tfidf_and_soc = defaultdict()
    for key, value in cos_score.items():
        weight_tfidf_and_soc[key] = bata*sum_tfidf[key] + (1-bata)*cos_score[key]
    return weight_tfidf_and_soc

"""
**************************
calculate the "phrase weight" that the query q in the document D, 
compare every two tokens,if they happened in the same order in a document
if so phrase_weight++
**************************
"""
def get_phrase_weight(term_id_list, term_postings, query):
    if len(query) <= 1:
        return {}
    phrase_weight = defaultdict()
    
    id_list = get_id_list(term_id_list)
    # for every doc, check phrase_hits
    for id in id_list:
        phrase_hits = 0
        key_id = str(id)
        for i in range(len(query)-1):
            token1 = query[i]
            token2 = query[i+1]
            posting1 = term_postings[token1]
            posting2 = term_postings[token2]

            # no document have the token
            if not posting1 or not posting2:
                continue

            #check if the doc include this token
            info1 = posting1.get(key_id)
            info2 = posting2.get(key_id)
            if not info1 or not info2:
                continue
            pos1 = info1["pos"]
            pos2 = info2["pos"]

            if not pos1 or not pos2:
                continue

            for p in pos1:
                if(p+1) in pos2:
                    phrase_hits += 1
                    break
        phrase_weight[key_id] = phrase_hits
    return phrase_weight


def calculate_result(weight_tfidf_and_soc, phrase_weight):
    result = {}
    for id in weight_tfidf_and_soc:
        tfidf_and_soc = weight_tfidf_and_soc[id]
        phrase = phrase_weight.get(id, 0.0)
        result[id] =  tfidf_and_soc * (1+ phrase) 
    return result
    
"""
**********************
S(q, D) = [bata * sum_of_tfidf * (1+ field_weight) + (1-bata) * cos(q, D)] * (1+ phrase_weight)
set bata = 0.5
**********************
"""      
def ranked_search(term_id_list, term_postings, query, n, df):
    # (tf_idf_q * tf_idf_d) /sqrt(tf_idf_d * tf_idf_d)
    cos_score = consine_score(term_postings, query, n, df)
    # tfidf * (1 + weight)
    sum_tfidf = sum_of_tf_idf(term_id_list, term_postings, query)
    
    # bata * sum_tfidf + (1-bata)* cos_score, 
    # set bata = 0.5 to give equal weight to term frequency (sum of tf-idf) and vector similarity (cosine) 
    weight_tfidf_and_soc = calculate_balance_weight(cos_score, sum_tfidf)

    phrase_weight = get_phrase_weight(term_id_list, term_postings, query)
    
    #   tfidf_and_soc * (1+ phrase)
    final_result = calculate_result(weight_tfidf_and_soc, phrase_weight)
 
    final_result = sorted(final_result, key=lambda d: final_result[d], reverse = True)
    return final_result


def search_doc(query: str, high_fre_term, url_data, lex_data, doc_len):
    """
    for each quest, seaching for the posible page 
    and print out the top 5 url
    """
    # count the time
    start = time.perf_counter()

    tokens = invert_index.tokenize(query)
    invert_index.logging.info(f"tokens: {tokens}")
    term_postings, term_id_list, df = load_terms_from_index(tokens, high_fre_term, lex_data)
 #   invert_index.logging.info(f"term_id_list: {term_id_list}")
   
   # ranked search
    n = len(url_data)
    sorted_id = ranked_search(term_id_list,term_postings,tokens, n, df)
    
    # boolean search
    # posible_id = find_posible_id(term_id_list)
    # sorted_id = sort_by_tf_idf(posible_id, tokens, term_postings)
    
    id_url = url_data["id_url"]
    url_list = []
    for docid in sorted_id[:5]:
        doc_url = id_url[str(docid)]
        url_list.append(doc_url)

    print("="*60)
    # print(f"\nTop 5 url is: {url_list}\n")
    print("\nTop 5 URLs are:")
    for url in url_list:
        print(url)
    
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000  # seconds → ms
    
    print(f"Query processed in {elapsed_ms:.3f} ms")



