import random
from section2 import extract_text, identify_language
from fastwarc.warc import ArchiveIterator, WarcRecordType

warc_file = "/home/samuelse/koa_scratch/ece405-assignment2-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

def evaluate_20_random_examples(warc_path, num_samples=20):
    print(f"Extracting {num_samples} texts for language evaluation...\n")
    
    extracted_texts = []
    
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if record.http_headers and "text/html" in record.http_headers.get("Content-Type", ""):
                try:
                    html_content = record.reader.read()
                    text = extract_text(html_content)
                    
                    if len(text.strip()) > 50: 
                        extracted_texts.append(text)
                        
                    if len(extracted_texts) >= 100:
                        break
                except Exception:
                    continue

    sampled_texts = random.sample(extracted_texts, min(num_samples, len(extracted_texts)))
    
    english_count = 0
    
    for i, text in enumerate(sampled_texts):
        preview = " ".join(text.split())[:200] 
        
        predicted_lang, score = identify_language(text)
        
        if predicted_lang == 'en':
            english_count += 1
            
        print(f"--- Example {i+1} ---")
        print(f"Predicted Language: {predicted_lang.upper()} (Confidence: {score:.4f})")
        print(f"Text Preview: {preview}...\n")
        
    print("="*40)
    print(f"Total English documents: {english_count} out of {len(sampled_texts)}")
    print(f"Fraction of English documents: {english_count / len(sampled_texts):.2f}")

if __name__ == "__main__":
    evaluate_20_random_examples(warc_file)