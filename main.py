import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import chromadb
import database
import easyocr
import fitz
import io
import models
import pickle
import re
import requests
import schemas
import subprocess
import time
import uuid
import uvicorn
import smtplib
import toml

from datetime import datetime
from email.message import EmailMessage
from models import MessagesChat, SenderEnum 
from deep_translator import GoogleTranslator
from langdetect import detect
from PIL import Image
from typing import List
from nltk.corpus import stopwords
from fastapi import FastAPI, File, UploadFile, Depends, Form, Query, HTTPException
from fastapi.responses import JSONResponse
from keybert import KeyBERT
from passlib.context import CryptContext
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

config = toml.load(".env.toml")
settings = config["app"]

tokenizer_path = settings["TOKENIZER_PATH"]
model_path = settings["MODEL_PATH"]
skill_pattern_path = settings["SKILL_PATTERN_PATH"]

model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

ocr_reader = easyocr.Reader(['en', 'id'], gpu=False)
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
kw_model = KeyBERT(embedding_model)

client = chromadb.PersistentClient(path=settings["CHROMA_DB_PATH"])
collection = client.get_or_create_collection(name="resumes")
stopwords_eng = set(stopwords.words('english'))

def preprocess_for_ocr(pil_image: Image.Image, max_area: int = 1_000_000):
    w, h = pil_image.size
    if w * h > max_area:
        scale = (max_area / (w * h)) ** 0.5
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    gray_image = pil_image.convert("L")
    return gray_image

# Extract Text from PDF
def extract_text_from_pdf_file(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)
    if num_pages > 2:
        raise HTTPException(
            status_code = 400,
            detail= "Dokumen melebihi 2 halaman, tidak dapat diproses"
        )
    text = "\n".join([page.get_text("text") for page in doc])
    if text.strip():
        return text

    # Jika teks kosong, lakukan OCR
    result = ""
    for page in doc:
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        preprocessed_image = preprocess_for_ocr(image)

        with io.BytesIO() as buf:
            preprocessed_image.save(buf, format="PNG")
            buf.seek(0)
            result += " ".join(ocr_reader.readtext(buf, detail=0)) + "\n"

    # Jika hasil OCR juga kosong, kembalikan error
    if not result.strip():
        raise HTTPException(
            status_code=400, 
            detail = "Dokumen tidak berisikan teks"
        )
    return result

# Clean Text Function
def clean_text(text):
    text = text.lower()                       # Konversi ke huruf kecil
    text = re.sub(r'\d+', '', text)           # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)       # Hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih

    # Stopword Removal
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stopwords_eng]
    return ' '.join(filtered_tokens)

# Extract Keywords Function (w/KeyBERT)
def extract_keywords(text):
    lang = detect(text)
    if lang == 'id':
        translated_text = GoogleTranslator(source='id', target='en').translate(text)
    else:
        translated_text = text

    keywords = kw_model.extract_keywords(
        translated_text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=10
    )
    return [kw[0] for kw in keywords]

# Predict Document Function
def predict_document(text, max_len=500):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post")
    prediction = model.predict(padded)[0][0]
    return prediction > 0.5

# Send Verification Email Function
def send_verification_email(receiver_email: str, token: str):
    msg = EmailMessage()
    msg['Subject'] = 'Verifikasi Email Aplikasi JobChat'
    msg['From'] = 'admjobchat@gmail.com'
    msg['To'] = receiver_email
    msg.set_content(f'''
    Klik link berikut untuk verifikasi email Anda untuk login ke aplikasi JobChat:
    https://{settings['NGROK_DOMAIN']}/verify-email?token={token}
    ''')
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(settings["EMAIL_SENDER"], settings["EMAIL_PASSWORD"])
        smtp.send_message(msg)

def send_reset_password(receiver_email: str, token: str):
    msg = EmailMessage()
    msg['Subject'] = 'Reset Password Aplikasi JobChat'
    msg['From'] = 'admjobchat@gmail.com'
    msg['To'] = receiver_email
    msg.set_content(f'''
    Klik link berikut untuk mengatur ulang password Anda:
    https://{settings['NGROK_DOMAIN']}/reset-password?token={token}
    ''')
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(settings["EMAIL_SENDER"], settings["EMAIL_PASSWORD"])
        smtp.send_message(msg)

# with OpenRouter LLM
def run_openrouter_llm(skills, categories):
    OPENROUTER_API_KEY = settings["OPENROUTER_API_KEY"]
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Job Chatbot"
    }

    prompt = f"""
        Berdasarkan keahlian yang dipunya seperti {skills} dalam kategori pekerjaan {categories}
        berikan rekomendasi pekerjaan yang cocok, dan berikan jawabannya dengan format berikut:
        " 
            Halo, Job Seeker! 
            Berdasarkan keahlian yang Anda miliki, kami telah menemukan beberapa pekerjaan yang mungkin cocok untuk Anda.
            Kategori Pekerjaan: {categories}
            Kami merekomendasikan Anda untuk mempertimbangkan pekerjaan berikut: \n
            \n
            Nama Pekerjaan
            Deskripsi Pekerjaan:
            Alasan Cocok: 
        "
        Jangan tambahkan format lain atau opini pribadi.
        """
    
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [
            {
                "role": "system", 
                "content": (
                    "Kamu adalah asisten karir."
                    "Jika tidak yakin 100 %, katakan ‘Saya tidak begitu yakin dengan rekomendasi saya.’ "
                    "Jawab pertanyaan dengan akurat. Jika tidak tahu, katakan tidak tahu dan jangan mengarang data."
                )
            },
            {
                "role": "user", 
                "content": prompt
            },
        ],
        "temperature": 0.1
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

def save_bot_message(db, chat_id: int, message: str):
    bot_message = MessagesChat(
        chat_id=chat_id,
        sender=SenderEnum.bot,
        message=message,
        is_file=False,
        file_name=None,
        file_url=None,
        timestamp=datetime.now()
    )
    db.add(bot_message)
    db.commit()
    db.refresh(bot_message)
    
def search_and_recommend(user_query, skills_from_user, top_k=5):
    query_embedding = embedding_model.encode(user_query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    categories = set()

    for metadata in results["metadatas"][0]:
        category = metadata.get("category")
        if category:
            categories.add(category)

    formatted_skills = ", ".join(skills_from_user)
    formatted_categories = ", ".join(categories)
    response = run_openrouter_llm(formatted_skills, formatted_categories)
    return response

models.Base.metadata.create_all(bind=database.engine) 
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Chatbot API is running"
    }

@app.post("/getJobRecommendationbyText")
async def get_job_recommendation(text: str = Form(...), chat_id: int = Form(...),              
    db: Session = Depends(database.get_db)
):
    cleaned = clean_text(text)
    keywords = extract_keywords(cleaned)

    if not keywords:
        message = "Tidak ada keyword yang ditemukan dalam teks."
        save_bot_message(db, chat_id, message)
        return {
            "cv_detected": False,
            "message": message,
            "recommendation_text": message
        }
    
    recommendation = search_and_recommend(cleaned, keywords)
    save_bot_message(db, chat_id, recommendation)
    return {
        "cv_detected": True,
        "message": "Berhasil mendapatkan rekomendasi.",
        "recommendation_text": recommendation
    }

@app.post("/getJobRecommendationbyImgOrPDF")
async def get_job_recommendation(file: UploadFile = File(...), chat_id: int = Form(...),
    db: Session = Depends(database.get_db)
):
    try:
        filename = file.filename.lower()
        file_bytes = await file.read()

        # Ekstraksi teks dari PDF
        if filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf_file(file_bytes)
            if isinstance(raw_text, JSONResponse):
                return raw_text

        # Ekstraksi teks dari gambar
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            raw_text = " ".join(ocr_reader.readtext(file_bytes, detail=0))
            if not raw_text.strip():
                return HTTPException(
                    status_code = 400, 
                    detail = "Gambar tidak berisikan teks yang dapat diekstrak."
                )
        else:
            raise HTTPException(
                status_code = 400, 
                detail = "Format file tidak didukung. Hanya PDF, JPG, JPEG, dan PNG yang diperbolehkan."
            )

        # Preprocessing dan deteksi CV
        cleaned = clean_text(raw_text)
        is_cv = predict_document(cleaned)
        if not is_cv:
            message = "Dokumen ini bukan merupakan dokumen CV"
            save_bot_message(db, chat_id, message)
            return {
                "cv_detected": False,
                "message": message,
                "recommendation_text": message
            }

        # Ekstraksi keyword dan rekomendasi
        keywords = extract_keywords(cleaned)
        if not keywords:
            message = "Tidak ada keyword yang ditemukan dalam dokumen."
            save_bot_message(db, chat_id, message)
            return {
                "cv_detected": True,
                "message": message,
                "recommendation_text": message
            }

        recommendation = search_and_recommend(cleaned, keywords)
        save_bot_message(db, chat_id, recommendation)
        return {
            "cv_detected": True,
            "message": "Berhasil mendapatkan rekomendasi.",
            "recommendation_text": recommendation
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail= f"Internal server error: {str(e)}"
        )

@app.post("/register", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code = 400, 
            detail = "Email sudah terdaftar, Silahkan login atau gunakan email lain."
        )
    token = str(uuid.uuid4())
    hashed_password = pwd_context.hash(user.password) 
    new_user = models.User(
        username=user.username, 
        email=user.email, 
        password=hashed_password,
        is_verified=False,
        verification_token=token
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    send_verification_email(user.email, token)
    return JSONResponse(
        status_code=200, 
        content={"message": "Berhasil mendaftar, Silahkan cek email Anda untuk verifikasi."}
    )

@app.get("/verify-email")
def verify_email(token: str = Query(...), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.verification_token == token).first()
    if not user:
        raise HTTPException(
            status_code = 404, 
            detail = "Token tidak valid."
        )
    if user.is_verified:
        return { "message": "Akun sudah terverifikasi." }
    user.is_verified = True
    user.verification_token = None 
    db.commit()
    return { "message": "Verifikasi email berhasil. Silakan login." }

@app.post("/login", response_model=schemas.User)
def login(user: schemas.UserLogin, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.password):
        raise HTTPException(
            status_code = 400, 
            detail = "Email atau password salah."
        )
    if not db_user.is_verified:
        raise HTTPException(
            status_code = 404, 
            detail = "Email belum diverifikasi"
        )
    return JSONResponse(status_code=200, content={
        "message": "Sukses login.",
        "user": {
            "id": db_user.id,
            "username": db_user.username,
            "email": db_user.email
        }
    })

@app.post("/forgot-password")
def forgot_password(user: schemas.ForgotPassword, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user:
        raise HTTPException(
            status_code=404, 
            detail="Email tidak ditemukan."
        )
    token = str(uuid.uuid4())
    hashed_new_pass = pwd_context.hash(user.new_password)

    db_user.verification_token = token
    db_user.new_password_temp = hashed_new_pass
    db.commit()

    send_reset_password(user.email, token)  
    return JSONResponse(
        status_code = 200,
        content = {"message": "Link verifikasi telah dikirim ke email Anda."}
    )

@app.get("/reset-password")
def reset_password(token: str = Query(...), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.verification_token == token).first()
    if not user:
        raise HTTPException(
            status_code=400, 
            detail="Token tidak valid."
        )
    user.password = user.new_password_temp
    user.verification_token = None
    user.new_password_temp = None
    db.commit()
    return {"message": "Password berhasil diubah. Silakan login dengan password baru."}

@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user is None:
        raise HTTPException(
            status_code = 404, 
            detail = "Pengguna tidak ditemukan."
        )
    return db_user

@app.post("/history_chats", response_model=schemas.HistoryChat)
def create_history_chat(history_chat: schemas.HistoryChatCreate, db: Session = Depends(database.get_db)):
    db_history_chat = models.HistoryChat(user_id=history_chat.user_id)
    db.add(db_history_chat)
    db.commit()
    db.refresh(db_history_chat)
    return db_history_chat

@app.get("/history_chats/user/{user_id}", response_model=List[schemas.HistoryChat])
def get_history_chat(user_id: int, db: Session = Depends(database.get_db)):
    db_chat = db.query(models.HistoryChat).filter(models.HistoryChat.user_id == user_id).all()
    if db_chat is None:
        raise HTTPException(
            status_code = 404, 
            detail = "Riwayat chat tidak ditemukan untuk pengguna ini."
        )
    return db_chat

@app.delete("/history_chats/user/{user_id}")
def delete_all_history_chat(user_id: int, db: Session = Depends(database.get_db)):
    chats = db.query(models.HistoryChat).filter(models.HistoryChat.user_id == user_id).all()
    if not chats:
        raise HTTPException(
            status_code = 404, 
            detail = "Riwayat chat tidak ditemukan untuk pengguna ini."
        )
    for chat in chats:
        db.delete(chat)
    db.commit()
    return { "message": "Seluruh riwayat chat telah dihapus." }

@app.delete("/history_chats/{chat_id}")
def delete_history_chat(chat_id: int, db: Session = Depends(database.get_db)):
    chat = db.query(models.HistoryChat).filter(models.HistoryChat.id == chat_id).first()
    if not chat:
        raise HTTPException(
            status_code = 404, 
            detail = "Chat tidak ditemukan."
        )
    db.delete(chat)
    db.commit()
    return { "message": "Chat telah dihapus." }

@app.post("/messages_chats", response_model=schemas.Message)
def create_message(message: schemas.MessageCreate, db: Session = Depends(database.get_db)):
    db_message = models.MessagesChat(
        chat_id=message.chat_id,
        sender=SenderEnum.user,
        message=message.message,
        is_file=message.is_file,
        file_name=message.file_name,
        file_url=message.file_url
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

@app.get("/messages_chats/{chat_id}", response_model= List[schemas.Message])
def get_messages_chat(chat_id: int, db: Session = Depends(database.get_db)):
    db_messages = db.query(models.MessagesChat).filter(models.MessagesChat.chat_id == chat_id).all()
    return db_messages

ngrok_command = [
    "ngrok", "http",
    f"--domain={settings['NGROK_DOMAIN']}", 
    str(settings.get("APP_PORT", 8000))
]
subprocess.Popen(ngrok_command)
time.sleep(2)

public_url = settings["NGROK_PUBLIC_URL"]
print("ngrok public URL:", public_url)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings["PORT"])
