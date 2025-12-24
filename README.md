# 🧠 MindClassify AI - Klasifikasi Kondisi Kesehatan Mental Berbasis Teks

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.57-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-green)
![License](https://img.shields.io/badge/License-Educational-green)

**Proyek UAP - Praktikum Machine Learning**  

---

### 👤 Biodata Mahasiswa

| | |
|---|---|
| **Nama** | Muhammad Syahrul Bachtiar |
| **NIM** | 202210370311046 |
| **Program Studi** | Teknik Informatika |
| **Universitas** | Universitas Muhammadiyah Malang |

---

## ⚠️ DISCLAIMER

**PENTING:** Proyek ini adalah **proyek pembelajaran** untuk klasifikasi teks menggunakan machine learning dan deep learning. Sistem ini **BUKAN alat diagnosis medis** dan tidak dapat menggantikan konsultasi dengan profesional kesehatan mental.

Jika Anda atau seseorang yang Anda kenal mengalami masalah kesehatan mental, silakan hubungi:
- Psikolog atau psikiater profesional
- **Indonesia**: Hubungi 119 ext. 8 (Kementerian Kesehatan)
- **Into The Light Indonesia**: 021-788-42580
- Layanan konseling dan kesehatan mental terdekat

---

## 📋 Deskripsi Proyek

**MindClassify AI** adalah sistem klasifikasi teks berbasis deep learning untuk mengidentifikasi kondisi kesehatan mental dari teks input berbahasa Inggris. Proyek ini mengimplementasikan dan membandingkan performa tiga arsitektur model yang berbeda:

1. **LSTM (Long Short-Term Memory)** - Baseline model neural network yang dilatih dari awal
2. **BERT (Bidirectional Encoder Representations from Transformers)** - Model pretrained transformer state-of-the-art
3. **DistilBERT** - Versi ringan dan efisien dari BERT dengan knowledge distillation

### 🎯 Tujuan Proyek

- ✅ Mengimplementasikan sistem klasifikasi teks multi-kelas untuk 7 kategori kondisi kesehatan mental
- ✅ Membandingkan performa model baseline (LSTM) dengan pretrained transformer models (BERT & DistilBERT)
- ✅ Menerapkan teknik transfer learning dan fine-tuning pada domain text classification
- ✅ Membangun aplikasi web interaktif dengan Streamlit untuk demonstrasi real-time
- ✅ Menganalisis trade-off antara akurasi, kecepatan, dan efisiensi model

### 🏷️ Kategori Klasifikasi

Sistem dapat mengklasifikasikan teks ke dalam **7 kategori** kondisi kesehatan mental:

| Kategori | Emoji | Deskripsi |
|----------|-------|-----------|
| **Anxiety** | 😰 | Kecemasan berlebihan, ketakutan, kekhawatiran |
| **Bipolar** | 🎭 | Gangguan suasana hati ekstrem (mania-depresi) |
| **Depression** | 😔 | Kesedihan mendalam, kehilangan minat, putus asa |
| **Normal** | 😊 | Kondisi mental yang sehat dan seimbang |
| **Personality Disorder** | 🧩 | Pola perilaku dan pemikiran yang tidak sehat |
| **Stress** | 😓 | Tekanan mental, beban berlebihan |
| **Suicidal** | 💔 | Pikiran atau keinginan untuk bunuh diri |

### 💡 Keunggulan Sistem

- 🚀 **Multi-Model Comparison**: Tiga model berbeda dalam satu aplikasi
- 🎨 **Modern UI/UX**: Interface futuristik dengan dark theme menggunakan Streamlit
- ⚡ **Real-time Prediction**: Analisis teks instan dengan confidence score
- 📊 **Probability Distribution**: Visualisasi probabilitas untuk semua kategori
- 🔄 **Model Switching**: User dapat memilih dan membandingkan hasil antar model

---

## 📊 Dataset

### Sumber Dataset
- **Platform**: Kaggle
- **Nama Dataset**: Combined Data 2 - Mental Health Sentiment Analysis
- **Format**: CSV
- **Ukuran Total**: **94,024 samples**
- **Bahasa**: Inggris (English)

### Kolom Dataset
| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `statement` | Text | Teks input berupa kalimat/paragraf tentang kondisi mental |
| `status` | Categorical | Label kondisi kesehatan mental (7 kategori) |

### Karakteristik Dataset

#### Distribusi Data
Setelah preprocessing, dataset dibagi menjadi:
- **Training Set**: 65,816 samples (70%)
- **Validation Set**: 14,104 samples (15%)
- **Test Set**: 14,104 samples (15%)

#### Distribusi Kategori
Dataset mencakup 7 kategori dengan distribusi yang relatif seimbang:

1. **Anxiety** - Gangguan kecemasan
2. **Bipolar** - Gangguan bipolar  
3. **Depression** - Depresi
4. **Normal** - Kondisi sehat
5. **Personality Disorder** - Gangguan kepribadian
6. **Stress** - Stres
7. **Suicidal** - Pikiran bunuh diri

#### Karakteristik Teks
- **Jenis Teks**: Status sosial media, ekspresi perasaan, pernyataan pribadi
- **Panjang Teks**: Bervariasi dari beberapa kata hingga beberapa kalimat
- **Bahasa**: Bahasa Inggris informal (slang, emoji, singkatan)
- **Kualitas**: Data mentah dengan noise (URL, mention, hashtag, typo)

---

## 🔧 Preprocessing Data

Pipeline preprocessing yang diterapkan untuk membersihkan dan mempersiapkan data:

### 1. Text Cleaning Pipeline

Setiap teks melewati tahapan cleaning berikut:

```python
def clean_text(text):
    # 1. Lowercase transformation
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 3. Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Remove hashtags (#tag)
    text = re.sub(r'#\w+', '', text)
    
    # 5. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 6. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # 7. Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
```

**Contoh Transformasi:**
```
Input  : "I'm feeling so depressed today 😔 #mentalhealth @support"
Output : "im feeling so depressed today"
```

### 2. Tokenization & Stopword Removal

**Untuk LSTM Model:**
- **Tokenization**: Menggunakan NLTK word tokenizer
- **Stopword Removal**: Menghapus stopwords bahasa Inggris (the, is, at, etc.)
- **Vocabulary**: Top 20,000 kata paling sering muncul
- **Max Length**: 128 tokens (sama dengan BERT)

**Untuk BERT & DistilBERT:**
- **Tokenization**: Menggunakan pretrained tokenizer (WordPiece)
- **Special Tokens**: [CLS], [SEP], [PAD]
- **Max Length**: 128 tokens
- **Attention Mask**: Untuk membedakan token asli vs padding

### 3. Encoding & Vectorization

#### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['status'])
```

Mapping label kategorikal ke numerik:
```
Anxiety               → 0
Bipolar               → 1
Depression            → 2
Normal                → 3
Personality disorder  → 4
Stress                → 5
Suicidal              → 6
```

#### Text Encoding

**LSTM:**
```python
tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=128, padding='post')
```

**BERT/DistilBERT:**
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True
)
```

### 4. Data Split Strategy

Dataset dibagi dengan stratified split untuk menjaga distribusi kategori:

```python
from sklearn.model_selection import train_test_split

# Split 1: Train + Temp (85% + 15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Split 2: Val + Test (50% + 50% dari temp)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
```

**Hasil Split:**
- Training: 70% (65,816 samples)
- Validation: 15% (14,104 samples)  
- Testing: 15% (14,104 samples)

### 5. Handling Data Quality Issues

✅ **NaN Handling**: `.fillna('').astype(str)` untuk menghindari error  
✅ **Duplicate Removal**: Menghapus data duplikat  
✅ **Empty Text**: Filter teks kosong setelah cleaning  
✅ **Imbalanced Data**: Stratified split untuk distribusi seimbang

---

## 🤖 Model Machine Learning

Proyek ini mengimplementasikan dan membandingkan **tiga arsitektur model** yang berbeda untuk klasifikasi teks kesehatan mental.

---

### 1️⃣ LSTM (Neural Network Baseline - Non-Pretrained)

**Improved Bidirectional LSTM Model v2**

#### Arsitektur Model

```
Input Sequence (128 tokens, vocabulary 20K)
    ↓
Embedding Layer (256-dimensional, trainable)
    ↓
SpatialDropout1D (0.2) - Regularisasi embedding
    ↓
Bidirectional LSTM (256 units) + return_sequences=True
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (128 units) + return_sequences=True  
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.4)
    ↓
Dense (128 units, ReLU activation)
    ↓
Dropout (0.4)
    ↓
Dense (64 units, ReLU activation)
    ↓
Dropout (0.3)
    ↓
Output Dense (7 classes, Softmax activation)
```

#### Hyperparameters

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| **Vocabulary Size** | 20,000 | Top 20K kata paling sering |
| **Embedding Dimension** | 256 | Ukuran vektor kata |
| **Max Sequence Length** | 128 | Sama dengan BERT/DistilBERT |
| **LSTM Layers** | 3 | Triple stacked Bidirectional LSTM |
| **LSTM Units** | 256 → 128 → 64 | Progressive reduction |
| **Dense Layers** | 2 (128, 64) | Classification head |
| **Dropout Rates** | 0.2 - 0.4 | Progressive regularization |
| **Optimizer** | Adam | Learning rate default |
| **Loss Function** | Sparse Categorical Crossentropy | Multi-class classification |
| **Batch Size** | 32 | Optimal untuk model besar |
| **Epochs** | 40 (max) | With early stopping |
| **Early Stop Patience** | 10 | Stop jika val_loss tidak improve |

#### Training Configuration

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)
```

#### Karakteristik

✅ **Kelebihan:**
- Lightweight dan cepat untuk inference (~50ms per prediction)
- Resource komputasi rendah (dapat berjalan di CPU)
- Mudah dikustomisasi dan di-tune
- Tidak memerlukan pretrained weights
- Model size kecil (~15-20 MB)

❌ **Kekurangan:**
- Membutuhkan data training yang banyak untuk performa optimal
- Tidak memanfaatkan transfer learning
- Akurasi lebih rendah dibanding model pretrained
- Hanya menangkap pola sequential, tidak semantic mendalam

#### Performance Metrics

- **Test Accuracy**: ~72-78% (tergantung training run)
- **Training Time**: ~30-50 menit (CPU mode)
- **Inference Speed**: ~50ms per sample
- **Model Size**: ~15-20 MB
- **Total Parameters**: ~2-3 Million

---

### 2️⃣ BERT (Pretrained Transformer Model)

**BERT Base Uncased - Fine-tuned for Mental Health Classification**

#### Arsitektur Model

```
Input Text (max 128 tokens)
    ↓
BERT Tokenizer (WordPiece)
    ↓
Input IDs + Attention Mask + Token Type IDs
    ↓
BERT Base Model (12 Transformer Layers)
│   ├── Multi-Head Self-Attention (12 heads)
│   ├── Feed-Forward Network
│   └── Layer Normalization + Residual
    ↓
[CLS] Token Representation (768-dim)
    ↓
Dropout (0.1)
    ↓
Classification Head (Dense 7 units, Softmax)
```

#### Model Specifications

| Specification | Value |
|---------------|-------|
| **Base Model** | bert-base-uncased |
| **Transformer Layers** | 12 |
| **Hidden Size** | 768 |
| **Attention Heads** | 12 |
| **Parameters** | ~110 Million |
| **Max Sequence Length** | 128 tokens |
| **Vocabulary Size** | 30,522 (WordPiece) |
| **Pretrained On** | Wikipedia + BookCorpus |

#### Fine-tuning Configuration

```python
# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # Recommended for BERT fine-tuning
EPOCHS = 3

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

#### Karakteristik

✅ **Kelebihan:**
- **State-of-the-art accuracy** untuk text classification
- **Bidirectional context understanding** yang sangat baik
- **Transfer learning** dari knowledge base yang luas
- Menangkap semantic dan konteks mendalam
- Performa terbaik di antara ketiga model

❌ **Kekurangan:**
- Training time sangat lama (~1-3 jam)
- Resource komputasi tinggi (butuh GPU untuk training efisien)
- Model size besar (~440 MB)
- Inference lebih lambat (~200-300ms per sample)
- Membutuhkan memory besar saat training

#### Performance Metrics

- **Test Accuracy**: **81.91%** ⭐ (Highest)
- **Validation Accuracy**: 83.47%
- **Training Time**: ~1-3 jam (GPU), ~5-8 jam (CPU)
- **Inference Speed**: ~200-300ms per sample
- **Model Size**: ~440 MB
- **Total Parameters**: ~110 Million

---

### 3️⃣ DistilBERT (Distilled Pretrained Model)

**DistilBERT Base Uncased - Efficient BERT Alternative**

#### Arsitektur Model

```
Input Text (max 128 tokens)
    ↓
DistilBERT Tokenizer (WordPiece)
    ↓
Input IDs + Attention Mask
    ↓
DistilBERT Model (6 Transformer Layers) - 50% dari BERT
│   ├── Multi-Head Self-Attention (12 heads)
│   ├── Feed-Forward Network
│   └── Layer Normalization + Residual
    ↓
[CLS] Token Representation (768-dim)
    ↓
Dropout (0.1)
    ↓
Classification Head (Dense 7 units, Softmax)
```

#### Model Specifications

| Specification | Value |
|---------------|-------|
| **Base Model** | distilbert-base-uncased |
| **Transformer Layers** | 6 (50% reduction) |
| **Hidden Size** | 768 (same as BERT) |
| **Attention Heads** | 12 |
| **Parameters** | ~66 Million (40% smaller) |
| **Max Sequence Length** | 128 tokens |
| **Vocabulary Size** | 30,522 (same as BERT) |
| **Distilled From** | BERT Base |

#### Knowledge Distillation

DistilBERT menggunakan teknik **knowledge distillation**:
- Student model (DistilBERT) belajar dari teacher model (BERT)
- Mempertahankan 97% performa BERT
- Mengurangi size 40% dan meningkatkan speed 60%

#### Fine-tuning Configuration

```python
# Same as BERT
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
```

#### Karakteristik

✅ **Kelebihan:**
- **Best balance** antara accuracy dan efficiency
- 60% **lebih cepat** dari BERT
- 40% **lebih kecil** dari BERT
- Mempertahankan ~97% performa BERT
- **Ideal untuk production deployment**
- Resource requirement lebih rendah dari BERT

❌ **Kekurangan:**
- Sedikit lebih rendah dari BERT dalam akurasi (~1-2%)
- Masih membutuhkan resource lebih dari LSTM
- Training time tetap cukup lama (~30-90 menit GPU)

#### Performance Metrics

- **Test Accuracy**: **80.67%** ⭐ (Second best)
- **Validation Accuracy**: 82.40%
- **Training Time**: ~30-90 menit (GPU), ~3-5 jam (CPU)
- **Inference Speed**: ~100-150ms per sample (60% faster than BERT)
- **Model Size**: ~260 MB (40% smaller than BERT)
- **Total Parameters**: ~66 Million

---

### 📊 Model Comparison Matrix

| Aspek | LSTM v2 | BERT | DistilBERT |
|-------|---------|------|------------|
| **Architecture** | 3-layer BiLSTM | 12-layer Transformer | 6-layer Transformer |
| **Parameters** | ~2-3M | ~110M | ~66M |
| **Pretrained** | ❌ No | ✅ Yes | ✅ Yes (Distilled) |
| **Training Time** | 30-50 min | 1-3 hours | 30-90 min |
| **Inference Speed** | ⚡ Fast (~50ms) | 🐢 Slow (~250ms) | 🚀 Medium (~120ms) |
| **Accuracy** | ~72-78% | ⭐ 81.91% | ⭐ 80.67% |
| **Model Size** | 💾 Small (15MB) | 💾 Large (440MB) | 💾 Medium (260MB) |
| **Resource Need** | 💻 Low (CPU ok) | 💻 High (GPU needed) | 💻 Medium (GPU better) |
| **Use Case** | Resource-limited | Maximum accuracy | Production deployment |

---

## 📈 Hasil Evaluasi dan Analisis

### 📊 Tabel Analisis Perbandingan Model

Berikut adalah ringkasan performa dan hasil analisis ketiga model yang telah dilatih dan dievaluasi pada dataset mental health classification:

| **Nama Model** | **Akurasi** | **Hasil Analisis** |
|----------------|-------------|-------------------|
| **LSTM (Bidirectional LSTM v2)** | **72.72%** | Model baseline dengan arsitektur 3-layer Bidirectional LSTM yang dilatih dari awal tanpa pretrained weights. Menggunakan 20K vocabulary dan 128 sequence length dengan 256-dimensional embeddings. Training time ~35 menit di CPU. Performa baik untuk model non-pretrained dengan resource komputasi rendah (15MB model size, ~50ms inference). Cocok untuk deployment di edge devices atau aplikasi dengan resource terbatas. Akurasi 72.72% menunjukkan pembelajaran sequential patterns yang cukup baik, namun terbatas dalam pemahaman semantic dan konteks mendalam. |
| **BERT (bert-base-uncased)** | **81.91%** ⭐ | Model pretrained transformer dengan 12 layers dan 110M parameters yang di-fine-tune untuk klasifikasi mental health. Menggunakan transfer learning dari knowledge base Wikipedia + BookCorpus. Memberikan **akurasi tertinggi (81.91%)** dengan pemahaman konteks bidirectional yang superior. Training time ~90 menit di CPU. Performa terbaik dengan F1-score macro 0.82 dan konsisten di semua kategori. Model size 440MB dengan inference ~250ms. **Rekomendasi: Pilihan terbaik untuk akurasi maksimal** dalam aplikasi yang memprioritaskan performa prediksi dan memiliki resource komputasi memadai. |
| **DistilBERT (distilbert-base-uncased)** | **80.67%** ⭐ | Versi ringan dari BERT hasil knowledge distillation dengan 6 layers (50% dari BERT) dan 66M parameters (40% lebih kecil). Mempertahankan **98.5% performa BERT** (hanya 1.24% lebih rendah) dengan keuntungan signifikan: training 33% lebih cepat (~60 menit), inference 60% lebih cepat (~120ms), dan model size 40% lebih kecil (260MB). F1-score macro 0.81 dengan performa konsisten. **Rekomendasi: Best balance antara accuracy dan efficiency**, ideal untuk production deployment yang membutuhkan keseimbangan antara akurasi tinggi dan kecepatan inference. |

### 💡 Kesimpulan Perbandingan

**Improvement Analysis:**
- **LSTM → DistilBERT**: +7.95% absolute improvement (+10.9% relative)
- **LSTM → BERT**: +9.19% absolute improvement (+12.6% relative)  
- **DistilBERT → BERT**: +1.24% absolute improvement (+1.5% relative)

**Rekomendasi Penggunaan:**
- ✅ **Maksimum Akurasi**: Gunakan **BERT** (81.91%)
- ✅ **Production Deployment**: Gunakan **DistilBERT** (80.67%, fast & efficient)
- ✅ **Resource Terbatas**: Gunakan **LSTM** (72.72%, lightweight & fast)

---

### Metrik Evaluasi

Setiap model dievaluasi menggunakan metrik standar untuk multi-class classification:

#### 1. **Classification Report**
- **Accuracy**: Proporsi prediksi benar dari total prediksi
- **Precision**: Ketepatan prediksi positif untuk setiap kelas
- **Recall**: Kemampuan model mendeteksi kelas positif
- **F1-Score**: Harmonic mean dari precision dan recall

#### 2. **Confusion Matrix**
- Visualisasi prediksi benar vs salah untuk setiap kelas
- Identifikasi kelas yang sering di-misclassify
- Analisis error patterns

#### 3. **Training Curves**
- **Loss Curve**: Training loss vs Validation loss
- **Accuracy Curve**: Training accuracy vs Validation accuracy
- **Deteksi Overfitting**: Gap antara training dan validation metrics

---

### 📊 Hasil Training Model

#### LSTM Model (Improved v2)

**Test Set Performance:**
```
Test Accuracy: 72.72%
Test Samples: 14,104
```

**Training Summary:**
- Epochs Trained: 15 (early stopped dari max 40)
- Best Validation Accuracy: 74.62%
- Final Training Accuracy: 89.49%
- Training Time: ~35 menit (CPU mode, M4 Pro)

**Classification Report (Ringkasan):**
| Metric | Score |
|--------|-------|
| Macro Avg Precision | 0.73 |
| Macro Avg Recall | 0.73 |
| Macro Avg F1-Score | 0.73 |
| Weighted Avg F1-Score | 0.73 |

**Analisis:**
- ✅ Model baseline yang solid dengan 3-layer BiLSTM
- ✅ Training stabil tanpa overfitting signifikan
- ⚠️ Accuracy 72% cukup baik untuk non-pretrained model
- ⚠️ Beberapa kelas (Personality Disorder) sulit dibedakan dari Normal/Depression

---

#### BERT Model

**Test Set Performance:**
```
Test Accuracy: 81.91% ⭐ (HIGHEST)
Test Samples: 14,104
```

**Training Summary:**
- Epochs Trained: 3
- Best Validation Accuracy: 83.47%
- Final Training Accuracy: 89.49%
- Training Time: ~90 menit (CPU mode, M4 Pro)

**Classification Report (Ringkasan):**
| Metric | Score |
|--------|-------|
| Macro Avg Precision | 0.82 |
| Macro Avg Recall | 0.82 |
| Macro Avg F1-Score | 0.82 |
| Weighted Avg F1-Score | 0.82 |

**Analisis:**
- ✅ **Akurasi tertinggi** di antara ketiga model
- ✅ Pemahaman konteks bidirectional yang superior
- ✅ Transfer learning sangat efektif untuk domain mental health
- ✅ Performa konsisten across semua kategori
- 🎯 **Model terbaik untuk akurasi maksimal**

---

#### DistilBERT Model

**Test Set Performance:**
```
Test Accuracy: 80.67% ⭐ (SECOND BEST)
Test Samples: 14,104
```

**Training Summary:**
- Epochs Trained: 3
- Best Validation Accuracy: 82.40%
- Final Training Accuracy: 87.13%
- Training Time: ~60 menit (CPU mode, M4 Pro)

**Classification Report (Ringkasan):**
| Metric | Score |
|--------|-------|
| Macro Avg Precision | 0.81 |
| Macro Avg Recall | 0.81 |
| Macro Avg F1-Score | 0.81 |
| Weighted Avg F1-Score | 0.81 |

**Analisis:**
- ✅ **Best balance** antara accuracy dan efficiency
- ✅ Hanya ~1.24% lebih rendah dari BERT
- ✅ Training 33% lebih cepat dari BERT
- ✅ Inference ~60% lebih cepat dari BERT
- 🎯 **Model terbaik untuk production deployment**

---

### 📊 Tabel Perbandingan Komprehensif

| Aspek | LSTM v2 | DistilBERT | BERT |
|-------|---------|------------|------|
| **🎯 Test Accuracy** | 72.72% | 80.67% | **81.91%** ⭐ |
| **📈 Validation Accuracy** | 74.62% | 82.40% | 83.47% |
| **🏋️ Training Time** | **~35 min** ⚡ | ~60 min | ~90 min |
| **⚡ Inference Speed** | **~50ms** ⚡ | ~120ms | ~250ms |
| **💾 Model Size** | **15 MB** 💾 | 260 MB | 440 MB |
| **🔢 Parameters** | ~2-3M | 66M | 110M |
| **📊 F1-Score (Macro)** | 0.73 | 0.81 | **0.82** ⭐ |
| **💻 Resource Need** | **Low (CPU)** | Medium | High (GPU better) |
| **✅ Pretrained** | ❌ No | ✅ Yes | ✅ Yes |

---

### 🔍 Analisis Perbandingan Detail

#### Accuracy Improvement Analysis

**LSTM → DistilBERT:**
- Improvement: **+7.95%** absolute
- Relative improvement: **+10.9%**
- Trade-off: 17x model size, 2.4x inference time

**DistilBERT → BERT:**
- Improvement: **+1.24%** absolute  
- Relative improvement: **+1.5%**
- Trade-off: 1.7x model size, 2.1x inference time

**LSTM → BERT:**
- Improvement: **+9.19%** absolute
- Relative improvement: **+12.6%**
- Trade-off: 29x model size, 5x inference time

#### Category-wise Performance

Best performing categories (all models):
1. ✅ **Suicidal** - Highest precision/recall (distinctive language patterns)
2. ✅ **Normal** - Well-separated from mental health conditions
3. ✅ **Depression** - Clear depressive language indicators

Challenging categories:
1. ⚠️ **Personality Disorder** - Often confused with Depression/Normal
2. ⚠️ **Anxiety vs Stress** - Overlapping symptoms and language
3. ⚠️ **Bipolar** - Requires context across mood swings

---

### 🎯 Kesimpulan dan Rekomendasi

#### Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **Maximum Accuracy** | **BERT** | 81.91% accuracy, best F1-scores |
| **Production Deployment** | **DistilBERT** | Best balance: 80.67% accuracy, fast inference |
| **Resource-Constrained** | **LSTM** | Low resource, fast, 72.72% acceptable |
| **Real-time Mobile App** | **LSTM** or **DistilBERT** | Small size, fast inference |
| **Batch Processing** | **BERT** | Accuracy prioritized over speed |
| **Research/Academia** | **All three** | Comprehensive comparison |

#### Key Insights

1. **Transfer Learning Impact**: 
   - Pretrained models (BERT/DistilBERT) outperform LSTM by **8-9% absolute**
   - Transfer learning dari general language understanding sangat efektif

2. **DistilBERT Sweet Spot**:
   - Mempertahankan **98.5% performa BERT** (80.67% vs 81.91%)
   - **60% lebih cepat** inference
   - **40% lebih kecil** model size
   - **Pilihan terbaik untuk production**

3. **LSTM Competitiveness**:
   - **72.72% accuracy** cukup baik untuk baseline non-pretrained
   - Dengan optimization (3-layer BiLSTM, 20K vocab, 256 embedding), LSTM dapat competitive
   - Ideal untuk edge devices dan resource terbatas

4. **Domain-Specific Challenges**:
   - Mental health text classification challenging karena overlap symptoms
   - Context understanding (BERT/DistilBERT) crucial untuk disambiguasi
   - Beberapa kategori (Personality Disorder) inherently sulit

---

### 📁 Visualisasi Hasil

Semua hasil evaluasi disimpan di folder **`results/`**:

#### Training History Plots
- `02_lstm_training_history.png` - LSTM accuracy/loss curves
- `03_bert_training_history.png` - BERT training progression
- `04_distilbert_training_history.png` - DistilBERT training curves

#### Confusion Matrices
- `02_lstm_confusion_matrix.png` - LSTM prediction patterns
- `03_bert_confusion_matrix.png` - BERT classification matrix
- `04_distilbert_confusion_matrix.png` - DistilBERT matrix

#### Classification Reports
- `02_lstm_classification_report.txt` - LSTM detailed metrics
- `03_bert_classification_report.txt` - BERT per-class performance
- `04_distilbert_classification_report.txt` - DistilBERT breakdown

#### Results JSON
- `02_lstm_results.json` - LSTM training summary & hyperparameters
- `03_bert_results.json` - BERT complete results
- `04_distilbert_results.json` - DistilBERT metrics & config

---

### 🚀 Future Improvements

Potential enhancements untuk meningkatkan performa:

1. **Data Augmentation**: Back-translation, paraphrasing untuk expand training data
2. **Ensemble Methods**: Combine predictions dari ketiga model
3. **Class Weighting**: Handle subtle class imbalance issues
4. **Fine-tuning Strategy**: Layer-wise learning rates, gradual unfreezing
5. **Larger Models**: BERT-large, RoBERTa untuk accuracy boost
6. **Domain Adaptation**: Pretrain lebih lanjut pada mental health corpus
7. **Multi-task Learning**: Auxiliary tasks (sentiment, emotion) sebagai additional signals

---

## 🌐 Panduan Menjalankan Sistem Website Secara Lokal

### 📋 Prerequisites

Pastikan sistem Anda memenuhi requirements berikut:

- **Python**: Versi 3.11 atau lebih tinggi
- **pip**: Python package manager (biasanya included dengan Python)
- **RAM**: Minimal 8GB (16GB recommended untuk training BERT)
- **Storage**: Minimal 2GB free space
- **OS**: Windows, macOS, atau Linux

---

### 🚀 Langkah-Langkah Instalasi

#### 1️⃣ Clone atau Download Repository

**Option A: Menggunakan Git**
```bash
git clone https://github.com/rullbachtiar2207/UAP_Pembelajaran_Mesin_B_202210370311046
cd UAP-Mesin-Learning
```

**Option B: Download ZIP**
1. Download repository sebagai ZIP file
2. Extract ke folder pilihan Anda
3. Buka terminal/command prompt di folder tersebut

---

#### 2️⃣ Buat Virtual Environment (Highly Recommended)

Virtual environment mengisolasi dependencies proyek agar tidak konflik dengan Python global.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

Setelah aktif, Anda akan melihat `(venv)` di command prompt.

---

#### 3️⃣ Install Dependencies

Install semua package yang diperlukan menggunakan requirements.txt:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Packages yang akan diinstall:**
- tensorflow==2.13.0
- transformers==4.57.3
- streamlit==1.52.2
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- nltk

**Estimasi waktu**: 5-10 menit (tergantung koneksi internet)

---

#### 4️⃣ Verifikasi Instalasi

Cek apakah semua package terinstall dengan benar:

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

Output yang diharapkan:
```
TensorFlow: 2.13.0
Transformers: 4.57.3
Streamlit: 1.52.2
```

---

#### 5️⃣ Prepare Dataset (Jika Belum Ada)

Jika folder `data/processed/` belum ada atau kosher:

1. Pastikan file `Combined Data 2.csv` ada di root folder
2. Jalankan notebook preprocessing:
   ```bash
   jupyter notebook notebooks/01_preprocessing.ipynb
   ```
3. Run semua cell untuk generate processed data

**Alternatif**: Dataset sudah di-preprocess dan tersimpan di folder `data/processed/`

---

#### 6️⃣ Verifikasi Model Files

Pastikan semua model sudah ada di folder `models/`:

```bash
ls models/
```

**Expected output:**
```
bert_model/
distilbert_model/
label_encoder.pkl
lstm_model.h5
lstm_tokenizer.pkl
```

Jika model belum ada, train terlebih dahulu menggunakan notebooks:
- `02_lstm_model.ipynb` → Generate `lstm_model.h5`
- `03_bert_model.ipynb` → Generate `bert_model/`
- `04_distilbert_model.ipynb` → Generate `distilbert_model/`

---

### 🎬 Menjalankan Aplikasi Streamlit

#### Langkah 1: Pastikan Virtual Environment Aktif

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

---

#### Langkah 2: Navigate ke Folder Proyek

```bash
cd /path/to/UAP
```

---

#### Langkah 3: Run Streamlit App

```bash
streamlit run app/app.py
```

**Alternatif (jika di root folder):**
```bash
cd app
streamlit run app.py
```

---

#### Langkah 4: Akses Aplikasi

Setelah command di atas, Streamlit akan:
1. Start local server
2. Otomatis membuka browser
3. Menampilkan aplikasi di `http://localhost:8501`

**Output di terminal:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

---

### 🎯 Menggunakan Aplikasi

#### 1. Pilih Model

Di **sidebar kiri**, pilih salah satu model:
- 🔷 **LSTM Neural Network** - Fast, lightweight
- 🔶 **BERT Transformer** - Highest accuracy
- 🔸 **DistilBERT Optimized** - Balanced performance

#### 2. Input Text

Di **text area utama**, ketik atau paste teks dalam **bahasa Inggris**:

**Contoh Input:**
```
I feel so anxious and worried all the time. I can't stop thinking about what might go wrong.
```

```
I've been feeling really down lately. Nothing seems to make me happy anymore.
```

```
Life is great! I'm feeling very positive and energized about everything.
```

#### 3. Analyze

Klik tombol **"🔍 ANALYZE TEXT"**

#### 4. Lihat Hasil

Aplikasi akan menampilkan:

**A. Prediction Result Card:**
- 🎯 Kategori yang diprediksi (e.g., "ANXIETY")
- 📊 Confidence score (e.g., 87.5%)
- 😰 Emoji representation

**B. Probability Distribution:**
- Bar chart untuk semua 7 kategori
- Percentage untuk setiap kelas
- Sorted dari tertinggi ke terendah

---

### 🔧 Troubleshooting

#### ❌ Error: "No module named 'tensorflow'"

**Solusi:**
```bash
pip install tensorflow==2.13.0
```

#### ❌ Error: "ModuleNotFoundError: No module named 'streamlit'"

**Solusi:**
```bash
pip install streamlit==1.52.2
```

#### ❌ Error: "Model file not found"

**Solusi:**
1. Pastikan di folder `models/` ada file:
   - `lstm_model.h5`
   - `bert_model/` (folder)
   - `distilbert_model/` (folder)
   - `label_encoder.pkl`
   - `lstm_tokenizer.pkl`

2. Jika belum ada, jalankan training notebooks terlebih dahulu

#### ❌ Port 8501 already in use

**Solusi 1: Gunakan port lain**
```bash
streamlit run app/app.py --server.port 8502
```

**Solusi 2: Kill process yang menggunakan port**
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8501 | xargs kill -9
```

#### ❌ Streamlit app sangat lambat

**Penyebab**: Model BERT/DistilBERT loading pertama kali

**Solusi**: 
- Tunggu beberapa detik saat pertama kali load model
- Model akan di-cache dan lebih cepat untuk request berikutnya
- Untuk performance lebih baik, gunakan LSTM model

#### ❌ Warning: TensorFlow Metal/GPU not detected

**Ini NORMAL** untuk setup ini. Aplikasi running di **CPU mode** yang sudah stable.

**Jika ingin GPU (optional):**
- Pastikan GPU compatible (NVIDIA CUDA)
- Install TensorFlow GPU version
- Untuk Mac M1/M2/M3: tensorflow-metal sudah discontinued

---

### 🛑 Menghentikan Aplikasi

Untuk stop Streamlit server:

1. Kembali ke terminal yang menjalankan Streamlit
2. Tekan **`Ctrl + C`**
3. Tunggu beberapa detik hingga server shutdown

---

### 🌍 Deploy ke Internet (Optional)

Jika ingin aplikasi accessible secara online:

#### Option 1: Streamlit Community Cloud (Free)

1. Push repository ke GitHub
2. Login ke [share.streamlit.io](https://share.streamlit.io)
3. Deploy dengan klik "New app"
4. Select repository, branch, dan file path (app/app.py)
5. Tunggu deployment (~5 menit)

**Limitations:**
- Model size limit (1GB)
- Resource terbatas (CPU only)
- BERT/DistilBERT mungkin timeout

**Recommendation**: Deploy dengan LSTM model only untuk Streamlit Cloud

#### Option 2: Heroku (Paid/Free Tier Limited)

1. Create `Procfile`:
   ```
   web: streamlit run app/app.py --server.port $PORT
   ```

2. Deploy ke Heroku:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

3. Ensure model files included atau download from cloud storage

---

### 📱 Akses dari Perangkat Lain di Network

Untuk akses aplikasi dari smartphone/tablet di WiFi yang sama:

1. Jalankan Streamlit dengan network URL:
   ```bash
   streamlit run app/app.py --server.address 0.0.0.0
   ```

2. Lihat **Network URL** di terminal:
   ```
   Network URL: http://192.168.1.100:8501
   ```

3. Buka URL tersebut di browser perangkat lain di network yang sama

---

### 🎓 Tips untuk Demonstrasi

#### Persiapan Demo:

1. **Test Run** sebelum demo:
   ```bash
   streamlit run app/app.py
   ```
   Pastikan semua model load dengan benar

2. **Prepare Sample Texts** untuk setiap kategori:
   - Anxiety: "I'm constantly worrying and can't relax..."
   - Depression: "I feel hopeless and nothing brings me joy..."
   - Suicidal: "I don't want to live anymore..."
   - Normal: "Life is great and I'm feeling wonderful!"

3. **Demo Flow**:
   - Start dengan LSTM → Show fast inference
   - Switch ke BERT → Show higher accuracy
   - Switch ke DistilBERT → Show balanced performance
   - Highlight probability distribution untuk setiap model

4. **Network Setup**:
   - Ensure stable internet untuk load models
   - Close other heavy applications
   - Test di browser yang akan digunakan

---

### 🔒 Security Notes

**PENTING:**
- ⚠️ Aplikasi ini **HANYA untuk edukasi dan demonstrasi**
- ⚠️ **BUKAN alat diagnosis medis profesional**
- ⚠️ Jangan deploy ke public tanpa disclaimer yang jelas
- ⚠️ Data input user **tidak disimpan** secara default
- ⚠️ Model predictions **tidak 100% akurat**

---

### 📞 Support

Jika mengalami masalah saat menjalankan aplikasi:

1. **Check Error Message**: Baca error di terminal dengan teliti
2. **Check README**: Pastikan follow semua steps
3. **Check Dependencies**: Verify semua package terinstall
4. **Check Model Files**: Ensure models ada di folder `models/`
5. **Contact**: Buka issue di GitHub atau hubungi asisten praktikum

---

### ✅ Quick Start Checklist

Sebelum demo, pastikan:

- [ ] Python 3.11+ installed
- [ ] Virtual environment created dan activated
- [ ] All dependencies installed (`requirements.txt`)
- [ ] Dataset ada di `data/processed/`
- [ ] All models trained dan saved di `models/`
- [ ] Streamlit running di `http://localhost:8501`
- [ ] Tested dengan sample texts
- [ ] Browser compatible (Chrome/Firefox/Safari)
- [ ] Network stable
- [ ] Prepared demo script

---

**🎉 Selamat! Aplikasi siap digunakan dan di-demo!**

---

## 🚀 Cara Menjalankan Proyek (Complete Guide)

### 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.11+ | 3.11.14 |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 2 GB free | 5 GB free |
| **CPU** | Dual-core | Quad-core+ |
| **GPU** | Not required | NVIDIA CUDA (optional) |
| **OS** | Windows/macOS/Linux | macOS M1+ or Ubuntu 20.04+ |

---

### 1️⃣ Clone Repository

```bash
git clone https://github.com/rullbachtiar2207/UAP_Pembelajaran_Mesin_B_202210370311046
cd UAP
```

---

### 2️⃣ Setup Virtual Environment

**macOS/Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements include:**
- tensorflow==2.13.0 (CPU mode)
- transformers==4.57.3
- streamlit==1.52.2
- pandas, numpy, scikit-learn
- matplotlib, seaborn

---

### 4️⃣ Prepare Data

Pastikan struktur folder:
```
UAP/
├── Combined Data 2.csv          # Raw dataset
├── data/
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── metadata.pkl
```

**Jika belum ada**, jalankan preprocessing notebook:
```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```

---

### 5️⃣ Train Models (Jika Belum Ada)

#### Train LSTM Model:
```bash
jupyter notebook notebooks/02_lstm_model.ipynb
```
- Run cells 1-16 untuk training
- Estimasi: ~35-50 menit (CPU)
- Output: `models/lstm_model.h5`, `models/lstm_tokenizer.pkl`

#### Train BERT Model:
```bash
jupyter notebook notebooks/03_bert_model.ipynb
```
- Run cells 1-15 untuk training
- Estimasi: ~1-3 jam (CPU), ~20-40 menit (GPU)
- Output: `models/bert_model/`

#### Train DistilBERT Model:
```bash
jupyter notebook notebooks/04_distilbert_model.ipynb
```
- Run cells 1-15 untuk training
- Estimasi: ~30-90 menit (CPU), ~15-30 menit (GPU)
- Output: `models/distilbert_model/`

---

### 6️⃣ Run Streamlit App

```bash
streamlit run app/app.py
```

**Aplikasi akan terbuka di:** `http://localhost:8501`

---

### 🎯 Quick Start (Jika Model Sudah Ada)

```bash
# 1. Activate venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 2. Run app
streamlit run app/app.py

# 3. Open browser
# http://localhost:8501
```

---

## 📂 Struktur Proyek

```
UAP/
│
├── 📄 Combined Data 2.csv                    # Dataset asli dari Kaggle (94K samples)
├── 📄 README.md                              # Dokumentasi lengkap (file ini)
├── 📄 requirements.txt                       # Python dependencies
├── 📄 .gitignore                            # Git ignore rules
├── 📄 LSTM_IMPROVEMENTS.md                  # Dokumentasi improvements LSTM
│
├── 📁 app/                                  # Streamlit Application
│   ├── app.py                               # Main Streamlit app ⭐
│   └── verify_setup.py                      # Verification script
│
├── 📁 notebooks/                            # Jupyter Notebooks untuk training
│   ├── 01_preprocessing.ipynb               # Data cleaning & splitting
│   ├── 02_lstm_model.ipynb                  # LSTM training & evaluation
│   ├── 03_bert_model.ipynb                  # BERT fine-tuning
│   └── 04_distilbert_model.ipynb            # DistilBERT fine-tuning
│
├── 📁 data/                                 # Datasets
│   └── processed/                           # Cleaned & split data
│       ├── train.csv                        # Training set (65,816 samples)
│       ├── val.csv                          # Validation set (14,104 samples)
│       ├── test.csv                         # Test set (14,104 samples)
│       └── metadata.pkl                     # Label info & statistics
│
├── 📁 models/                               # Trained Models ⭐
│   ├── lstm_model.h5                        # LSTM saved model (~15MB)
│   ├── lstm_tokenizer.pkl                   # LSTM tokenizer
│   ├── label_encoder.pkl                    # Shared label encoder
│   ├── bert_model/                          # BERT model directory (~440MB)
│   │   ├── config.json
│   │   ├── tf_model.h5
│   │   └── ...
│   └── distilbert_model/                    # DistilBERT directory (~260MB)
│       ├── config.json
│       ├── tf_model.h5
│       └── ...
│
└── 📁 results/                              # Training Results & Visualizations
    ├── 01_label_distribution.png            # Dataset class distribution
    ├── 02_lstm_training_history.png         # LSTM training curves
    ├── 02_lstm_confusion_matrix.png         # LSTM confusion matrix
    ├── 02_lstm_classification_report.txt    # LSTM detailed metrics
    ├── 02_lstm_results.json                 # LSTM summary JSON
    ├── 02_lstm_history.json                 # LSTM training history
    ├── 03_bert_training_history.png         # BERT training curves
    ├── 03_bert_confusion_matrix.png         # BERT confusion matrix
    ├── 03_bert_classification_report.txt    # BERT metrics
    ├── 03_bert_results.json                 # BERT summary
    ├── 03_bert_history.json                 # BERT history
    ├── 04_distilbert_training_history.png   # DistilBERT curves
    ├── 04_distilbert_confusion_matrix.png   # DistilBERT matrix
    ├── 04_distilbert_classification_report.txt
    ├── 04_distilbert_results.json
    └── 04_distilbert_history.json
```

### 📊 File Sizes

| Component | Size | Description |
|-----------|------|-------------|
| **Dataset** | ~50 MB | Combined Data 2.csv |
| **LSTM Model** | ~15 MB | lstm_model.h5 + tokenizer |
| **BERT Model** | ~440 MB | Full transformer model |
| **DistilBERT** | ~260 MB | Distilled transformer |
| **Total Project** | ~800 MB | Including all models |

---

## 🛠️ Tech Stack

### Machine Learning & Deep Learning
- **TensorFlow 2.13**: Framework deep learning utama
- **Keras**: High-level API untuk building neural networks
- **Transformers (Hugging Face)**: Pretrained models (BERT, DistilBERT)
- **Scikit-learn**: Preprocessing, metrics, dan utilities

### NLP & Text Processing
- **NLTK**: Tokenization, stopwords, lemmatization
- **Transformers Tokenizers**: BERT dan DistilBERT tokenizers

### Data Processing & Visualization
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib**: Plotting dan visualisasi
- **Seaborn**: Statistical visualizations

### Web Framework
- **Streamlit**: Framework untuk membuat web app interaktif

---

## 📝 Catatan Penting

### Training Tips

1. **GPU Recommendation**: 
   - LSTM: CPU sudah cukup
   - BERT/DistilBERT: Strongly recommended menggunakan GPU
   - Google Colab menyediakan free GPU

2. **Memory Management**:
   - BERT membutuhkan ~4-8GB RAM
   - Reduce batch size jika out of memory

3. **Hyperparameter Tuning**:
   - Learning rate: 2e-5 untuk BERT/DistilBERT optimal
   - Epochs: 3-5 untuk pretrained models sudah cukup
   - LSTM bisa membutuhkan 10-20 epochs

### Deployment Tips

1. **Model Optimization**:
   - Gunakan DistilBERT untuk production (lebih cepat)
   - Convert model ke TensorFlow Lite untuk mobile
   - Quantization untuk mengurangi model size

2. **Streamlit Deployment** (Optional):
   - Deploy ke [Streamlit Cloud](https://streamlit.io/cloud)
   - Deploy ke Heroku atau AWS
   - Butuh file `requirements.txt` dan `app.py`

---

## 🔍 Troubleshooting

### Issue: Model loading error
**Solusi**: Pastikan file model ada di folder `models/` dengan nama yang benar

### Issue: Out of memory saat training BERT
**Solusi**: Kurangi batch size dari 16 menjadi 8 atau 4

### Issue: NLTK data not found
**Solusi**: Download NLTK data sesuai langkah 4 di bagian instalasi

### Issue: TensorFlow GPU not detected
**Solusi**: Install TensorFlow GPU version atau gunakan Google Colab

---

## 📚 Referensi

### Papers & Articles
1. **LSTM**: Hochreiter & Schmidhuber (1997) - Long Short-Term Memory
2. **BERT**: Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers
3. **DistilBERT**: Sanh et al. (2019) - DistilBERT, a distilled version of BERT

### Libraries Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io)
- [NLTK Documentation](https://www.nltk.org/)

### Dataset
- [Kaggle - Sentiment Analysis for Mental Health](https://www.kaggle.com/)

---

## ⚡ Quick Start Summary

```bash
# 1. Clone dan setup
git clone <repo-url>
cd UAP
pip install -r requirements.txt

# 2. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# 3. Training model
jupyter notebook notebooks/Mental_Health_Classification.ipynb

# 4. Run aplikasi
streamlit run app.py
```

---

<div align="center">

**🧠 Klasifikasi Kondisi Kesehatan Mental Berbasis Teks 🧠**

*Proyek UAP - Praktikum Machine Learning*

**Remember: This is for educational purposes only, not for medical diagnosis!**

---

Made with ❤️ for Learning Machine Learning

</div>
