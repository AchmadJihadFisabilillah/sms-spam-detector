# Laporan Project Akhir

## Judul

**Deteksi SMS Spam Menggunakan Multinomial Naive Bayes dan Logistic Regression Berbasis Dataset Asli Kaggle**

## BAB I - Pendahuluan

### 1.1 Latar Belakang

Perkembangan teknologi komunikasi digital menyebabkan pertukaran pesan menjadi semakin cepat dan masif. Salah satu masalah yang muncul adalah banyaknya pesan spam yang berisi promosi tidak relevan, penipuan, tautan mencurigakan, atau instruksi yang berpotensi merugikan pengguna. Oleh karena itu, diperlukan sistem klasifikasi otomatis yang mampu membedakan pesan spam dan bukan spam.

Project ini membangun aplikasi berbasis web untuk mendeteksi SMS spam menggunakan dataset asli dari Kaggle. Algoritma yang digunakan adalah Multinomial Naive Bayes dan Logistic Regression. Kedua algoritma dibandingkan menggunakan metrik evaluasi klasifikasi serta McNemar Test untuk mengetahui apakah perbedaan performa keduanya signifikan secara statistik.

### 1.2 Rumusan Masalah

1. Bagaimana membangun aplikasi web untuk klasifikasi SMS spam menggunakan dataset asli Kaggle?
2. Bagaimana performa Multinomial Naive Bayes dalam mendeteksi SMS spam?
3. Bagaimana performa Logistic Regression dalam mendeteksi SMS spam?
4. Algoritma mana yang lebih baik berdasarkan accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, Log Loss, Brier Score, weighted ranking, dan McNemar Test?

### 1.3 Tujuan Penelitian

1. Menggunakan dataset asli SMS Spam Collection dari Kaggle.
2. Membuat pipeline preprocessing teks untuk data SMS.
3. Membangun model klasifikasi menggunakan Multinomial Naive Bayes.
4. Membangun model pembanding menggunakan Logistic Regression.
5. Membandingkan kedua model dengan metrik evaluasi dan McNemar Test.
6. Mengimplementasikan hasil model dalam aplikasi web berbasis FastAPI, React, dan Tailwind CSS.

### 1.4 Manfaat Penelitian

1. Membantu memahami penerapan text classification pada data nyata.
2. Memberikan contoh implementasi machine learning end-to-end.
3. Menghasilkan aplikasi yang dapat memprediksi probabilitas SMS termasuk spam.
4. Memberikan perbandingan objektif antara dua algoritma machine learning.

## BAB II - Landasan Teori

### 2.1 Text Classification

Text classification adalah proses mengelompokkan teks ke dalam kategori tertentu. Pada project ini, teks SMS diklasifikasikan ke dalam dua kategori, yaitu ham dan spam.

### 2.2 TF-IDF

TF-IDF adalah metode representasi teks yang mengubah kata menjadi nilai numerik berdasarkan tingkat kepentingannya dalam dokumen dan seluruh kumpulan data.

### 2.3 Multinomial Naive Bayes

Multinomial Naive Bayes adalah algoritma probabilistik yang banyak digunakan untuk klasifikasi teks karena mampu memodelkan frekuensi kemunculan kata pada setiap kelas.

### 2.4 Logistic Regression

Logistic Regression adalah algoritma klasifikasi yang menghitung probabilitas suatu data masuk ke kelas tertentu. Pada project ini, Logistic Regression digunakan sebagai model pembanding.

### 2.5 Confusion Matrix

Confusion matrix menampilkan jumlah prediksi benar dan salah dari setiap kelas, terdiri dari True Positive, True Negative, False Positive, dan False Negative.

### 2.6 McNemar Test

McNemar Test adalah uji statistik untuk membandingkan dua model klasifikasi pada dataset pengujian yang sama. Uji ini membantu menentukan apakah perbedaan performa dua model signifikan secara statistik.

## BAB III - Metodologi

### 3.1 Dataset

Dataset yang digunakan adalah SMS Spam Collection Dataset dari Kaggle. Dataset berisi pesan SMS berbahasa Inggris yang telah diberi label sebagai ham atau spam.

### 3.2 Variabel Penelitian

| Variabel | Keterangan |
|---|---|
| message | Isi pesan SMS |
| label | Kelas SMS: ham atau spam |
| label_encoded | Label numerik: ham = 0, spam = 1 |

### 3.3 Tahapan Penelitian

1. Pengumpulan dataset Kaggle.
2. Upload dataset ke aplikasi web.
3. Data cleaning dan deduplikasi.
4. Label encoding.
5. Text preprocessing.
6. Feature engineering berbasis token teks.
7. TF-IDF vectorization.
8. Split data training dan testing.
9. Training Multinomial Naive Bayes.
10. Training Logistic Regression.
11. Evaluasi model.
12. McNemar Test.
13. Implementasi prediksi pada web app.

### 3.4 Preprocessing

Preprocessing meliputi:

1. Mengubah teks menjadi lowercase.
2. Menghapus simbol yang tidak diperlukan.
3. Mengganti URL menjadi token khusus.
4. Mengganti angka panjang menjadi token khusus.
5. Menghapus spasi berlebih.
6. Menambahkan token fitur seperti keyword spam dan panjang pesan.

### 3.5 Modeling

Model pertama adalah Multinomial Naive Bayes dengan TF-IDF unigram dan bigram. Model kedua adalah Logistic Regression dengan class balancing.

### 3.6 Evaluasi

Metrik evaluasi yang digunakan:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. ROC-AUC
6. PR-AUC
7. Log Loss
8. Brier Score
9. Confusion Matrix
10. McNemar Test

## BAB IV - Hasil dan Pembahasan

### 4.1 Hasil Dataset

Hasil dataset ditampilkan pada dashboard aplikasi berupa total data dan distribusi label ham/spam.

### 4.2 Hasil Model

Aplikasi menampilkan hasil evaluasi dari dua model. Model terbaik ditentukan berdasarkan weighted final score:

```text
Final Score = 0.30 ROC-AUC + 0.25 F1-score + 0.20 Precision + 0.15 Recall + 0.10 Calibration Score
```

### 4.3 Hasil McNemar Test

McNemar Test digunakan untuk melihat apakah perbedaan performa Naive Bayes dan Logistic Regression signifikan secara statistik. Jika p-value < 0.05, maka perbedaan performa dianggap signifikan.

### 4.4 Implementasi Web

Aplikasi web terdiri dari:

1. Upload dataset.
2. Training model.
3. Dashboard evaluasi.
4. Prediksi SMS baru.
5. Batch prediction.
6. Export metrik dan hasil prediksi.

## BAB V - Kesimpulan dan Saran

### 5.1 Kesimpulan

Project ini berhasil membangun aplikasi web untuk mendeteksi SMS spam menggunakan dataset asli dari Kaggle. Dua algoritma, yaitu Multinomial Naive Bayes dan Logistic Regression, berhasil diimplementasikan dan dibandingkan dengan metrik evaluasi lengkap serta McNemar Test. Aplikasi juga mampu menghasilkan probabilitas spam untuk pesan baru.

### 5.2 Saran

Pengembangan berikutnya dapat menambahkan model deep learning seperti BERT, multilingual spam detection, deployment cloud, autentikasi user, dan monitoring performa model secara berkala.
