"""
MindClassify AI - Mental Health Text Classification
Premium Modern Interface
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Page Config
st.set_page_config(
    page_title="MindClassify AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern CSS with Beautiful Fonts
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Sora:wght@400;600;700;800&display=swap');

* {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0a0e1a;
    background-image: 
        radial-gradient(at 20% 10%, rgba(59, 130, 246, 0.08) 0px, transparent 50%),
        radial-gradient(at 80% 20%, rgba(139, 92, 246, 0.08) 0px, transparent 50%),
        radial-gradient(at 40% 80%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
}

.block-container { 
    padding: 2.5rem 4rem; 
    max-width: 1500px; 
}

/* ========== HEADER ========== */
.cyber-header {
    text-align: center;
    padding: 2.5rem 0;
    margin-bottom: 3rem;
    position: relative;
}

.cyber-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 3px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    border-radius: 10px;
}

.cyber-title {
    font-family: 'Sora', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 3px;
    margin: 0;
    text-shadow: 0 0 80px rgba(59, 130, 246, 0.3);
}

.cyber-subtitle {
    font-family: 'JetBrains Mono', monospace;
    color: #64748b;
    font-size: 0.9rem;
    letter-spacing: 4px;
    margin-top: 1rem;
    font-weight: 500;
    text-transform: uppercase;
}

/* ========== RESULT CARD ========== */
.result-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.8));
    border: 2px solid rgba(59, 130, 246, 0.3);
    border-radius: 24px;
    padding: 4rem 3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
                0 0 0 1px rgba(59, 130, 246, 0.1) inset;
    margin: 3rem 0;
    backdrop-filter: blur(20px);
}

.result-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, 
        transparent 30%, 
        rgba(59, 130, 246, 0.05) 50%, 
        transparent 70%);
    animation: shimmer 4s linear infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

.result-emoji {
    font-size: 6rem;
    filter: drop-shadow(0 0 40px rgba(59, 130, 246, 0.5));
    position: relative;
    z-index: 1;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.result-label {
    font-family: 'Sora', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 1.5rem 0;
    letter-spacing: 2px;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
}

.result-confidence {
    font-family: 'Sora', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    z-index: 1;
    line-height: 1;
}

.confidence-label {
    font-family: 'JetBrains Mono', monospace;
    color: #64748b;
    font-size: 0.85rem;
    letter-spacing: 2px;
    margin-top: 1rem;
    position: relative;
    z-index: 1;
    text-transform: uppercase;
}

/* ========== SIDEBAR ========== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(59, 130, 246, 0.15);
}

section[data-testid="stSidebar"] h3 {
    font-family: 'Sora', sans-serif !important;
    color: #60a5fa !important;
    letter-spacing: 2px;
    font-size: 1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    margin-top: 1.5rem !important;
}

/* ========== MODEL CARD ========== */
.model-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.08));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.model-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
}

.model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
}

.model-card h4 {
    font-family: 'Sora', sans-serif;
    color: #60a5fa;
    margin: 0 0 0.8rem 0;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 1px;
}

.model-card p {
    font-family: 'JetBrains Mono', monospace;
    color: #94a3b8;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    line-height: 1.6;
}

/* ========== CATEGORY GRID ========== */
.cat-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.6rem;
    margin-top: 1rem;
}

.cat-badge {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.7));
    border: 1px solid rgba(59, 130, 246, 0.15);
    border-radius: 12px;
    padding: 0.7rem 0.8rem;
    text-align: center;
    color: #cbd5e1;
    font-size: 0.85rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    transition: all 0.3s ease;
    cursor: default;
}

.cat-badge:hover {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
    border-color: rgba(59, 130, 246, 0.3);
    transform: translateY(-2px);
}

/* ========== BUTTON ========== */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1.2rem 4rem !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    box-shadow: 0 10px 40px rgba(59, 130, 246, 0.4),
                0 0 0 1px rgba(255, 255, 255, 0.1) inset !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.stButton > button:hover::before {
    width: 300px;
    height: 300px;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 50px rgba(59, 130, 246, 0.6),
                0 0 0 1px rgba(255, 255, 255, 0.2) inset !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* ========== TEXT AREA ========== */
.stTextArea textarea {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.7), rgba(30, 41, 59, 0.5)) !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
    border-radius: 16px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
    padding: 1.2rem !important;
    backdrop-filter: blur(10px) !important;
}

.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15),
                0 10px 30px rgba(59, 130, 246, 0.2) !important;
    outline: none !important;
}

.stTextArea textarea::placeholder {
    color: #475569 !important;
    font-style: italic !important;
}

/* ========== SELECT BOX ========== */
.stSelectbox > div > div {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.7), rgba(30, 41, 59, 0.5)) !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
}

/* ========== PROGRESS BAR ========== */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981) !important;
    border-radius: 10px !important;
    height: 8px !important;
}

.stProgress > div {
    background: rgba(15, 23, 42, 0.5) !important;
    border-radius: 10px !important;
}

/* ========== CAPTIONS ========== */
.stCaption {
    font-family: 'JetBrains Mono', monospace !important;
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
    margin-top: 0.3rem !important;
    font-weight: 500 !important;
}

/* ========== HEADINGS ========== */
h1, h2 {
    font-family: 'Sora', sans-serif !important;
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}

h3 {
    font-family: 'Sora', sans-serif !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

p, li, label, div {
    font-family: 'Space Grotesk', sans-serif;
    color: #94a3b8;
}

/* ========== SECTION HEADER ========== */
.section-header {
    font-family: 'Sora', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 2rem 0 1rem 0;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar { 
    width: 10px; 
    height: 10px; 
}

::-webkit-scrollbar-track { 
    background: #0f172a;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb { 
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 10px;
    border: 2px solid #0f172a;
}

::-webkit-scrollbar-thumb:hover { 
    background: linear-gradient(180deg, #60a5fa, #a78bfa);
}

/* ========== ANIMATIONS ========== */
@keyframes glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.glow {
    animation: glow 2s ease-in-out infinite;
}

/* ========== WARNINGS ========== */
.stWarning {
    background: rgba(251, 191, 36, 0.1) !important;
    border-left: 4px solid #fbbf24 !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ========== SPINNERS ========== */
.stSpinner > div {
    border-color: #3b82f6 transparent transparent transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_lstm_model():
    try:
        model = load_model(os.path.join(BASE_DIR, 'models', 'lstm_model.h5'))
        with open(os.path.join(BASE_DIR, 'models', 'lstm_tokenizer.pkl'), 'rb') as f:
            tokenizer = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        return model, tokenizer, label_encoder
    except:
        return None, None, None

@st.cache_resource
def load_bert_model():
    try:
        model = TFBertForSequenceClassification.from_pretrained(
            os.path.join(BASE_DIR, 'models', 'bert_model'), use_safetensors=False
        )
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        with open(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        return model, tokenizer, label_encoder
    except:
        return None, None, None

@st.cache_resource
def load_distilbert_model():
    try:
        model = TFDistilBertForSequenceClassification.from_pretrained(
            os.path.join(BASE_DIR, 'models', 'distilbert_model'), use_safetensors=False
        )
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        with open(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        return model, tokenizer, label_encoder
    except:
        return None, None, None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def predict_lstm(text, model, tokenizer, label_encoder):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=128)  # Updated to 128 (was 100)
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    label = label_encoder.inverse_transform([predicted_class])[0]
    probs = {label_encoder.classes_[i]: prediction[0][i] * 100 for i in range(len(label_encoder.classes_))}
    return label, confidence, probs

def predict_bert_based(text, model, tokenizer, label_encoder):
    cleaned = clean_text(text)
    encoded = tokenizer.encode_plus(
        cleaned, add_special_tokens=True, max_length=128,
        padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='tf'
    )
    logits = model.predict({
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }, verbose=0).logits
    probs_array = tf.nn.softmax(logits).numpy()[0]
    predicted_class = np.argmax(probs_array)
    confidence = probs_array[predicted_class] * 100
    label = label_encoder.inverse_transform([predicted_class])[0]
    probs = {label_encoder.classes_[i]: probs_array[i] * 100 for i in range(len(label_encoder.classes_))}
    return label, confidence, probs

# Header
st.markdown("""
<div class="cyber-header">
    <div class="cyber-title">MINDCLASSIFY AI</div>
    <div class="cyber-subtitle">Neural Mental Health Analysis System</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ CONFIGURATION")
    model_choice = st.selectbox("Select Model", ["🔷 LSTM Neural Network", "🔶 BERT Transformer", "🔸 DistilBERT Optimized"])
    
    st.markdown("---")
    
    if "LSTM" in model_choice:
        st.markdown('''
        <div class="model-card">
            <h4>🔷 LSTM Neural Network</h4>
            <p>→ Baseline Architecture</p>
            <p>→ Fast Inference Speed</p>
            <p>→ Lightweight Model</p>
        </div>
        ''', unsafe_allow_html=True)
    elif "BERT" in model_choice and "Distil" not in model_choice:
        st.markdown('''
        <div class="model-card">
            <h4>🔶 BERT Transformer</h4>
            <p>→ State-of-the-art NLP</p>
            <p>→ Highest Accuracy</p>
            <p>→ Context Understanding</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="model-card">
            <h4>🔸 DistilBERT Optimized</h4>
            <p>→ Distilled BERT</p>
            <p>→ 60% Faster</p>
            <p>→ 40% Smaller Size</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🏷️ DETECTION CATEGORIES")
    
    cats = [
        ("😰", "Anxiety"),
        ("🎭", "Bipolar"),
        ("😔", "Depression"),
        ("😊", "Normal"),
        ("🧩", "Personality"),
        ("😓", "Stress"),
        ("💔", "Suicidal")
    ]
    
    st.markdown('<div class="cat-grid">', unsafe_allow_html=True)
    for emoji, cat in cats:
        st.markdown(f'<div class="cat-badge">{emoji} {cat}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Content
st.markdown('<div class="section-header">💬 TEXT INPUT</div>', unsafe_allow_html=True)
user_input = st.text_area(
    "Input", 
    height=180, 
    placeholder="Enter your text here for mental health analysis...", 
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 ANALYZE TEXT", use_container_width=True)

if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter text to analyze")
    else:
        with st.spinner("🧠 Processing neural analysis..."):
            # Load model
            if "LSTM" in model_choice:
                m, t, le = load_lstm_model()
                if m: 
                    label, conf, probs = predict_lstm(user_input, m, t, le)
                else: 
                    st.error("❌ LSTM model not found")
                    st.stop()
            elif "BERT" in model_choice and "Distil" not in model_choice:
                m, t, le = load_bert_model()
                if m: 
                    label, conf, probs = predict_bert_based(user_input, m, t, le)
                else: 
                    st.error("❌ BERT model not found")
                    st.stop()
            else:
                m, t, le = load_distilbert_model()
                if m: 
                    label, conf, probs = predict_bert_based(user_input, m, t, le)
                else: 
                    st.error("❌ DistilBERT model not found")
                    st.stop()
            
            # Display Result
            emoji_map = {
                "Anxiety": "😰",
                "Bipolar": "🎭",
                "Depression": "😔",
                "Normal": "😊",
                "Personality disorder": "🧩",
                "Stress": "😓",
                "Suicidal": "💔"
            }
            emoji = emoji_map.get(label, "🔮")
            
            st.markdown(f"""
            <div class="result-card">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{label.upper()}</div>
                <div class="result-confidence">{conf:.1f}%</div>
                <div class="confidence-label">Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution
            st.markdown('<div class="section-header">📊 PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
            sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            
            for cat, prob in sorted_probs.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(prob/100)
                with col2:
                    st.markdown(f"**{prob:.1f}%**")
                st.caption(f"{emoji_map.get(cat, '•')} {cat}")
                st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
    <p style="margin: 0.5rem 0; font-size: 1rem; font-weight: 600;">🧠 MindClassify AI</p>
    <p style="margin: 0.5rem 0; font-size: 0.85rem; color: #475569;">UAP Machine Learning Project</p>
    <p style="margin: 0.5rem 0; font-size: 0.75rem; color: #334155; font-style: italic;">For Educational Purposes Only • Not a Substitute for Professional Medical Diagnosis</p>
</div>
""", unsafe_allow_html=True)