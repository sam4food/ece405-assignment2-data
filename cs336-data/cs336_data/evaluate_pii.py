from section2 import extract_text, mask_emails, mask_phone_numbers, mask_ips
from fastwarc.warc import ArchiveIterator, WarcRecordType

warc_file = "/home/samuelse/koa_scratch/ece405-assignment2-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

def find_pii_replacements(warc_path, target_samples=20):    
    found_examples = []
    
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if record.http_headers and "text/html" in record.http_headers.get("Content-Type", ""):
                try:
                    html_content = record.reader.read()
                    text = extract_text(html_content)
                    
                    if len(text.strip()) < 50:
                        continue
                        
                    masked_text, email_count = mask_emails(text)
                    masked_text, phone_count = mask_phone_numbers(masked_text)
                    masked_text, ip_count = mask_ips(masked_text)
                    
                    total_replacements = email_count + phone_count + ip_count
                    
                    if total_replacements > 0:
                        found_examples.append({
                            'text': masked_text,
                            'emails': email_count,
                            'phones': phone_count,
                            'ips': ip_count
                        })
                        print(f"Found example {len(found_examples)}...")
                        
                    if len(found_examples) >= target_samples:
                        break
                except Exception:
                    continue

    print("\n" + "="*50)
    print("PII REPLACEMENT EXAMPLES (Context Window)")
    print("="*50)
    
    for i, ex in enumerate(found_examples):
        text = ex['text']
        idx = text.find('|||') 
        
        if idx != -1:
            start = max(0, idx - 100)
            end = min(len(text), idx + 200)
            snippet = text[start:end].replace('\n', ' ')
            
            print(f"\n--- Example {i+1} ---")
            print(f"Masks applied: Emails({ex['emails']}), Phones({ex['phones']}), IPs({ex['ips']})")
            print(f"Context: ...{snippet}...")

if __name__ == "__main__":
    find_pii_replacements(warc_file)