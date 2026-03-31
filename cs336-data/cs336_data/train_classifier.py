import fasttext
import random
from fastwarc.warc import ArchiveIterator, WarcRecordType
from section2 import extract_text

HIGH_QUALITY_WARC = "/home/samuelse/koa_scratch/ece405-assignment2-data/pos_urls.warc.gz"
LOW_QUALITY_WARC = "/home/samuelse/koa_scratch/ece405-assignment2-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
TRAINING_OUTPUT_FILE = "quality_train.txt"
MODEL_OUTPUT_FILE = "quality_model.bin"

SAMPLE_LIMIT = 10000 

def extract_and_label_data():
    print("Extracting and labeling data")
    labeled_data = []

    print(f"{SAMPLE_LIMIT} high quality samples")
    count = 0
    with open(HIGH_QUALITY_WARC, 'rb') as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if "text/html" in record.http_headers.get("Content-Type", ""):
                try:
                    text = extract_text(record.reader.read()).replace("\n", " ").strip()
                    if len(text) > 150:
                        labeled_data.append(f"__label__high {text}")
                        count += 1
                except: continue
                if count >= SAMPLE_LIMIT: break
    print(f"     Found {count} high-quality samples.")

    print(f"{SAMPLE_LIMIT} low quality samples")
    count = 0
    with open(LOW_QUALITY_WARC, 'rb') as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if "text/html" in record.http_headers.get("Content-Type", ""):
                try:
                    text = extract_text(record.reader.read()).replace("\n", " ").strip()
                    if len(text) > 150:
                        labeled_data.append(f"__label__low {text}")
                        count += 1
                except: continue
                if count >= SAMPLE_LIMIT: break
    print(f"     Found {count} low-quality samples.")

    # shuffle and save
    print("shuffling data and writing to training file")
    random.shuffle(labeled_data)
    with open(TRAINING_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in labeled_data:
            f.write(item + "\n")
    print("Checker DONE.")

def train_quality_model():
    print("\ntraining fastText model")
    model = fasttext.train_supervised(
        input=TRAINING_OUTPUT_FILE,
        lr=0.1,
        epoch=10,
        wordNgrams=2,
        bucket=200000,
        dim=50,
        loss='hs'
    )
    model.save_model(MODEL_OUTPUT_FILE)
    print(f"model saved as {MODEL_OUTPUT_FILE}")

if __name__ == "__main__":
    extract_and_label_data()
    train_quality_model()