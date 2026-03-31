import random
from section2 import extract_text, gopher_quality_filter
from fastwarc.warc import ArchiveIterator, WarcRecordType

WARC_PATH = "/home/samuelse/koa_scratch/ece405-assignment2-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

def evaluate_quality_filters(warc_path, n=20):
    records = []
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if record.http_headers and "text/html" in record.http_headers.get("Content-Type", ""):
                try:
                    text = extract_text(record.reader.read())
                    if text.strip():
                        records.append(text)
                    if len(records) >= 100:
                        break
                except: continue

    samples = random.sample(records, min(n, len(records)))
    for i, text in enumerate(samples):
        passes = gopher_quality_filter(text)
        preview = " ".join(text.split())[:200]
        print(f"--- Example {i+1} ---\nPasses Filter: {passes}\nText: {preview}...\n")

if __name__ == "__main__":
    evaluate_quality_filters(WARC_PATH)