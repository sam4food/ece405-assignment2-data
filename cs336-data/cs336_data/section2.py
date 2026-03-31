import resiliparse.extract.html2text
import resiliparse.parse.encoding
import fasttext
import os
import re

def extract_text(html_bytes: bytes):
    try:
        # attempt to decode using the standard UTF-8 encoding
        decoded_html = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # if UTF-8 fails, detect the encoding using resiliparse
        detected_encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)
        
        # decode using the detected encoding. 
        decoded_html = html_bytes.decode(detected_encoding or 'utf-8', errors='replace')
        
    # extract and return the plain text
    return resiliparse.extract.html2text.extract_plain_text(decoded_html)

MODEL_PATH = "/home/samuelse/koa_scratch/ece405-assignment2-data/lid.176.bin"

try:
    fasttext_model = fasttext.load_model(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load fasttext model from {MODEL_PATH}. Error: {e}")
    fasttext_model = None

def identify_language(text: str):
    if not fasttext_model:
        raise RuntimeError("FastText model not loaded.")

    clean_text = text.replace('\n', ' ')
    
    predictions, probabilities = fasttext_model.predict(clean_text, k=1)
    
    raw_label = predictions[0]
    score = float(probabilities[0])
    
    language_id = raw_label.replace("__label__", "")
    
    return language_id, score

def mask_emails(text: str):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    return re.subn(email_pattern, '|||EMAIL_ADDRESS|||', text)


def mask_phone_numbers(text: str):
    phone_pattern = r'(?<!\d)(?:\+?1[-.\s]?)?(?:\([0-9]{3}\)|[0-9]{3})[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}(?!\d)'
    
    return re.subn(phone_pattern, '|||PHONE_NUMBER|||', text)


def mask_ips(text: str):
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    return re.subn(ip_pattern, '|||IP_ADDRESS|||', text)

NSFW_MODEL_PATH = "/home/samuelse/koa_scratch/ece405-assignment2-data/jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_MODEL_PATH = "/home/samuelse/koa_scratch/ece405-assignment2-data/jigsaw_fasttext_bigrams_hatespeech_final.bin"

nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
toxic_model = fasttext.load_model(TOXIC_MODEL_PATH)

def classify_nsfw(text: str):
    prediction = nsfw_model.predict(text.replace('\n', ' '))
    label = prediction[0][0].replace('__label__', '')
    score = float(prediction[1][0])
    return label, score

def classify_toxic_speech(text: str):
    prediction = toxic_model.predict(text.replace('\n', ' '))
    label = prediction[0][0].replace('__label__', '')
    score = float(prediction[1][0])
    return label, score

def gopher_quality_filter(text: str):
    words = text.split()
    word_count = len(words)
    
    if not (50 <= word_count <= 100000):
        return False
    
    mean_word_length = sum(len(word) for word in words) / word_count
    if not (3 <= mean_word_length <= 10):
        return False
    
    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith('...'))
        if (ellipsis_lines / len(lines)) > 0.3:
            return False
    
    alpha_word_count = sum(1 for word in words if any(c.isalpha() for c in word))
    if (alpha_word_count / word_count) < 0.8:
        return False
    
    return True

quality_model = fasttext.load_model("/home/samuelse/koa_scratch/ece405-assignment2-data/cs336-data/cs336_data/quality_model.bin")

def classify_quality(text: str):
    clean_text = text.replace('\n', ' ')
    
    labels, probabilities = quality_model.predict(clean_text, k=1)
    
    label = labels[0].replace("__label__", "")
    score = float(probabilities[0])
    
    return label, score