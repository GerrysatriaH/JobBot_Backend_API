import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import langdetect
import requests

from fastapi import FastAPI
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

kw_model = KeyBERT(SentenceTransformer('distiluse-base-multilingual-cased-v2'))

def clean_text(text):
    text = text.lower()                       # Konversi ke huruf kecil
    text = re.sub(r'\d+', '', text)           # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)       # Hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    return text

# Extract Keywords Function (with KeyBERT)
def extract_keywords(text):
    lang = langdetect.detect(text)
    stopwords = 'indonesian' if lang == 'id' else 'english'
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=stopwords, top_n=10)
    return [kw[0] for kw in keywords]

def run_openrouter_llm(keywords):
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Job Chatbot"
    }

    prompt = f"""
        Berdasarkan keahlian berikut: {', '.join(keywords)}, berikan rekomendasi pekerjaan yang cocok. 
        Format:
        - Nama Pekerjaan:
        - Deskripsi Singkat:
        - Alasan Cocok:
    """

    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [
            {"role": "system", "content": "Kamu adalah asisten karir yang membantu memberikan saran pekerjaan."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

app = FastAPI()

@app.get("/")
def read_root():
    return { "message": "API sedang berjalan." }

@app.post("/getJobRecommendation")
async def get_job_recommendation(text: str):
    cleaned = clean_text(text)
    keywords = extract_keywords(cleaned)
    recommendation = run_openrouter_llm(keywords)

    return {
        "cv_detected": True,
        "keywords_document": keywords,
        "recommendation_text": recommendation
    }
