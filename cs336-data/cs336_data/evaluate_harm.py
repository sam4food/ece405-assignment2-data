import random
from section2 import extract_text, classify_nsfw, classify_toxic_speech
from fastwarc.warc import ArchiveIterator, WarcRecordType

NSFW_MODEL_PATH = "/home/samuelse/koa_scratch/ece405-assignment2-data/jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_MODEL_PATH = "/home/samuelse/koa_scratch/ece405-assignment2-data/jigsaw_fasttext_bigrams_hatespeech_final.bin"
WARC_PATH = "/home/samuelse/koa_scratch/ece405-assignment2-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

def run_evaluation(warc_path, n=20):
    print(f"{n} random examples\n")
    
    records = []
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if record.http_headers and "text/html" in record.http_headers.get("Content-Type", ""):
                try:
                    text = extract_text(record.reader.read())
                    if len(text.strip()) > 100:
                        records.append(text)
                    if len(records) >= 100:
                        break
                except:
                    continue

    samples = random.sample(records, min(n, len(records)))
    
    for i, text in enumerate(samples):
        nsfw_label, nsfw_score = classify_nsfw(text)
        toxic_label, toxic_score = classify_toxic_speech(text)
        
        preview = " ".join(text.split())[:250]
        
        print(f"--- Example {i+1} ---")
        print(f"NSFW:  {nsfw_label} (Score: {nsfw_score:.4f})")
        print(f"TOXIC: {toxic_label} (Score: {toxic_score:.4f})")
        print(f"TEXT:  {preview}...\n")

if __name__ == "__main__":
    run_evaluation(WARC_PATH)