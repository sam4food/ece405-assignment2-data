import os
import hashlib
from collections import defaultdict
import re
import unicodedata
import random

def get_hash(line: str):
    return hashlib.md5(line.encode('utf-8')).hexdigest()

def exact_line_deduplication(input_files: list[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    line_counts = defaultdict(int)
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    line_hash = get_hash(clean_line)
                    line_counts[line_hash] += 1
                    
    for file_path in input_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                clean_line = line.strip()
                if clean_line:
                    line_hash = get_hash(clean_line)
                    if line_counts[line_hash] == 1:
                        f_out.write(line)

def minhash_deduplication(
    input_files: list[str], 
    num_hashes: int, 
    num_bands: int, 
    ngrams: int, 
    jaccard_threshold: float,
    output_dir: str
):
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def get_ngrams(text: str, n: int) -> set:
        words = text.split()
        return set(tuple(words[i:i+n]) for i in range(max(1, len(words)-n+1)))

    os.makedirs(output_dir, exist_ok=True)
    
    # initialize Hashing
    random.seed(42)
    P = 4294967311
    hash_coeffs = [(random.randint(1, P-1), random.randint(0, P-1)) for _ in range(num_hashes)]

    documents = {}

    for doc_id, file_path in enumerate(input_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        norm_text = normalize_text(raw_text)
        ngram_set = get_ngrams(norm_text, ngrams)
        
        sig = [float('inf')] * num_hashes
        if ngram_set:
            for ngram in ngram_set:
                val = hash(ngram) & ((1<<32)-1) 
                for i, (a, b) in enumerate(hash_coeffs):
                    h = (a * val + b) % P
                    if h < sig[i]:
                        sig[i] = h
                        
        documents[doc_id] = {
            'file': file_path,
            'ngrams': ngram_set,
            'sig': sig,
            'raw_text': raw_text
        }

    rows_per_band = num_hashes // num_bands
    buckets = [defaultdict(list) for _ in range(num_bands)]

    for doc_id, doc in documents.items():
        sig = doc['sig']
        for b in range(num_bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_tuple = tuple(sig[start:end])
            buckets[b][band_tuple].append(doc_id)

    candidates = set()
    for b_dict in buckets:
        for bucket_docs in b_dict.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i+1, len(bucket_docs)):
                        u, v = bucket_docs[i], bucket_docs[j]
                        candidates.add((min(u, v), max(u, v)))

    # compute exact jaccard
    parent = {i: i for i in documents.keys()}

    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for u, v in candidates:
        set_u = documents[u]['ngrams']
        set_v = documents[v]['ngrams']
        
        if not set_u and not set_v: jaccard = 1.0
        elif not set_u or not set_v: jaccard = 0.0
        else:
            intersection = len(set_u.intersection(set_v))
            union_sz = len(set_u) + len(set_v) - intersection
            jaccard = intersection / union_sz

        if jaccard >= jaccard_threshold:
            union(u, v)

    # group components
    components = defaultdict(list)
    for doc_id in documents.keys():
        components[find(doc_id)].append(doc_id)

    files_to_keep = set()
    for comp in components.values():
        chosen_doc_id = comp[0]
        doc = documents[chosen_doc_id]
        files_to_keep.add(doc['file'])

    for doc_id, doc in documents.items():
        file_path = doc['file']
        if file_path in files_to_keep:
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc['raw_text'])