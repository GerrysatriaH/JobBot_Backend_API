import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import easyocr
import fitz
import io
import langdetect
import re
import pickle
import requests

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from keybert import KeyBERT
from PIL import Image
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and tokenizer
tokenizer_path = "tokenizer.pkl"
model_path = "cv_classification_model.h5"

model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

ocr_reader = easyocr.Reader(['en', 'id'])
kw_model = KeyBERT(SentenceTransformer('distiluse-base-multilingual-cased-v2'))

# Clean Text Function
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

# Predict Document Function
def predict_document(text, max_len=200):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post")
    prediction = model.predict(padded)[0][0]
    return prediction > 0.5

# Extract Text from PDF Function
def extract_text_from_pdf_file(file: UploadFile):
    doc = fitz.open(stream=file.file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    if text.strip():
        return text
    else:
        result = ""
        for page in doc:
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                buf.seek(0)
                result += " ".join(ocr_reader.readtext(buf, detail=0)) + "\n"
        return result
    
# with Local LLM
def run_llm(keywords):
    joined_keywords = ", ".join(keywords)
    prompt = f"""        
        Berdasarkan keahlian berikut: {joined_keywords}, berikan rekomendasi pekerjaan yang cocok. 
        Tampilkan hasil dalam format:
        - Nama Pekerjaan:
        - Deskripsi Singkat:
        - Alasan Cocok:
        """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2", 
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

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
    return {"message": "API sedang berjalan."}

@app.post("/getJobRecommendationbyText")
async def get_job_recommendation(text: str):
    cleaned = clean_text(text)
    keywords = extract_keywords(cleaned)
    recommendation = run_openrouter_llm(keywords)

    return {
        "cv_detected": True,
        "keywords_document": keywords,
        "recommendation_text": recommendation
    }

@app.post("/getJobRecommendationbyImgOrPDF")
async def get_job_recommendation(file: UploadFile = File(...)):
    if file.filename.endswith(".pdf"):
        raw_text = extract_text_from_pdf_file(file)
    elif file.filename.endswith((".jpg", ".png", ".jpeg")):
        raw_text = " ".join(ocr_reader.readtext(await file.read(), detail=0))
    else:
        return JSONResponse({"error": "Format file tidak didukung."}, status_code=400)

    cleaned = clean_text(raw_text)
    is_cv = predict_document(cleaned)

    if not is_cv:
        return {
            "cv_detected": False, 
            "message": "Dokumen ini bukan merupakan dokumen CV."
        }

    keywords = extract_keywords(cleaned)
    recommendation = run_openrouter_llm(keywords)

    return {
        "cv_detected": True,
        "keywords_document": keywords,
        "recommendation_text": recommendation
    }
