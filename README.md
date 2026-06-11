# SMS Spam Detector Final Project

Project akhir full-stack untuk klasifikasi SMS spam menggunakan dataset asli Kaggle, backend Python FastAPI, frontend React + Tailwind CSS, dan machine learning berbasis Multinomial Naive Bayes serta Logistic Regression.

## Ringkasan

- **Topik:** Deteksi SMS spam dan ham.
- **Dataset:** SMS Spam Collection Dataset dari Kaggle.
- **Target:** `ham = 0`, `spam = 1`.
- **Algoritma 1:** Multinomial Naive Bayes.
- **Algoritma 2:** Logistic Regression.
- **Compare:** McNemar Test + Weighted Model Ranking.
- **Evaluasi:** Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, Log Loss, Brier Score, Confusion Matrix.
- **Output:** Probabilitas spam, label prediksi, risk level, hasil batch prediction, export CSV/JSON.

## Struktur Project

```text
sms-spam-detector-final/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── ml_service.py
│   │   ├── text_processing.py
│   │   ├── schemas.py
│   │   └── config.py
│   ├── data/
│   │   └── README.md
│   ├── artifacts/
│   ├── scripts/
│   │   └── download_kaggle_dataset.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
├── docs/
│   ├── PROJECT_REPORT.md
│   └── SLIDE_OUTLINE.md
├── docker-compose.yml
├── Makefile
└── README.md
```

## Dataset

Download dataset asli dari Kaggle:

```text
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
```

File yang dibutuhkan:

```text
spam.csv
```

Kolom yang dipakai:

| Kolom | Fungsi |
|---|---|
| `v1` | Label: ham atau spam |
| `v2` | Isi SMS mentah |

Letakkan file di:

```text
backend/data/spam.csv
```

Atau upload melalui dashboard web pada bagian **Dataset asli Kaggle**.

## Cara Menjalankan Tanpa Docker

### 1. Jalankan Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend API:

```text
http://localhost:8000
```

Swagger API docs:

```text
http://localhost:8000/docs
```

### 2. Jalankan Frontend

Buka terminal baru:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## Cara Menjalankan Dengan Docker

```bash
docker compose up --build
```

Frontend:

```text
http://localhost:5173
```

Backend:

```text
http://localhost:8000
```

## Cara Download Dataset Menggunakan Kaggle API

1. Buat API token di akun Kaggle.
2. Simpan `kaggle.json` di `~/.kaggle/kaggle.json`.
3. Jalankan:

```bash
python backend/scripts/download_kaggle_dataset.py
```

Script akan mengunduh dataset ke:

```text
backend/data/spam.csv
```

## Alur Penggunaan Aplikasi

1. Buka web app di `http://localhost:5173`.
2. Upload `spam.csv` dari Kaggle.
3. Klik **Train Model**.
4. Lihat hasil evaluasi Naive Bayes dan Logistic Regression.
5. Lihat hasil McNemar Test.
6. Masukkan SMS baru untuk prediksi.
7. Gunakan batch prediction untuk file CSV berisi banyak pesan.
8. Download `metrics.json` dan `test_predictions.csv` sebagai bukti hasil project.

## Endpoint API Penting

| Endpoint | Method | Fungsi |
|---|---|---|
| `/api/health` | GET | Cek API aktif |
| `/api/dataset/status` | GET | Cek status dataset |
| `/api/dataset/upload` | POST | Upload `spam.csv` |
| `/api/train` | POST | Training dua model |
| `/api/metrics` | GET | Ambil hasil evaluasi |
| `/api/predict` | POST | Prediksi satu SMS |
| `/api/batch-predict` | POST | Prediksi banyak SMS |
| `/api/download/metrics` | GET | Download metrik JSON |
| `/api/download/predictions` | GET | Download prediksi CSV |

## Contoh Request Prediksi

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"Congratulations! You won a free cash prize. Claim now!","threshold":0.5}'
```

## Catatan Akademik

Dataset tidak dibuat manual dan tidak diubah menjadi data dummy. Dataset yang digunakan adalah dataset publik dari Kaggle. Project ini menggunakan label asli `ham` dan `spam`, lalu melakukan preprocessing, training model, evaluasi, dan prediksi probabilitas secara reproducible.
