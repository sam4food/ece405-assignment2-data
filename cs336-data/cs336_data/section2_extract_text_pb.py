from section2 import extract_text
from fastwarc.warc import ArchiveIterator, WarcRecordType

def compare_extraction_methods(warc_path: str, wet_path: str):    
    with open(warc_path, 'rb') as warc_file:
        warc_iter = ArchiveIterator(warc_file, record_types=WarcRecordType.response)
        warc_record = next(warc_iter)
        html_bytes = warc_record.reader.read()
        
    my_extracted_text = extract_text(html_bytes)
    
    with open(wet_path, 'rb') as wet_file:
        wet_iter = ArchiveIterator(wet_file, record_types=WarcRecordType.conversion)
        wet_record = next(wet_iter)
        cc_extracted_text = wet_record.reader.read().decode('utf-8', errors='replace')
        
    print("\n" + "="*50)
    print("RESILIPARSE EXTRACTION (From WARC)")
    print("="*50)
    print(my_extracted_text[:1000]) 
    
    print("\n" + "="*50)
    print("EXTRACTION (From WET)")
    print("="*50)
    print(cc_extracted_text[:1000])
    print("\n" + "="*50)

compare_extraction_methods(
    warc_path='/home/samuelse/koa_scratch/ece405-assignment2-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz', 
    wet_path='/home/samuelse/koa_scratch/ece405-assignment2-data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz'
)