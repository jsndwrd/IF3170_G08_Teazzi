# IF3170_G08_Teazzi

Tugas Besar 2 IF3170 Inteligensi Artifisial · Kelompok 08

Proyek ini memprediksi pembatalan reservasi penginapan. Target `is_canceled` bernilai `1` jika reservasi dibatalkan dan `0` jika tidak. Metrik utama adalah **macro F1**, yaitu rata-rata F1 dari kedua kelas.

## Mulai dari sini

Jalankan perintah berikut dari root repo di macOS atau Linux.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook src/main.ipynb
```

Di VS Code, buka [`src/main.ipynb`](src/main.ipynb) dan pilih kernel Python dari `.venv`. Data sudah tersedia di [`data/`](data/), dengan 27.373 reservasi berlabel di `train.csv` dan 6.844 reservasi untuk diprediksi di `test.csv`.

## Alur notebook

1. **Data Understanding dan Preparation** — jelajahi distribusi target dan kualitas data, lalu lakukan stratified split 80:20. Fit preprocessor pada split latih dan terapkan ke validation.
2. **Modeling and Validation** — latih KNN, CART, Logistic Regression, SVM, dan Naive Bayes yang diimplementasikan dari nol. Notebook menampilkan skor holdout, 5-fold CV, dan hasil tuning.
3. **Comparison with scikit-learn** — bandingkan model dengan baseline scikit-learn pada split yang sama.
4. **Saving and Loading Models** — simpan lima model dasar di `output/validation_models/` dan bandingkan prediksi sampel sebelum dan sesudah model dimuat ulang.
5. **Kaggle Submission dan Error Analysis** — buat CSV prediksi final dan tinjau kesalahan pada validation dengan confusion matrix dan classification report.

Jalankan cell secara berurutan. **Run All** juga menjalankan tuning dan eksperimen bonus, serta memperbarui beberapa grafik dan CSV eksperimen di `output/`.

## Struktur repo

| Lokasi | Isi |
| --- | --- |
| [`src/main.ipynb`](src/main.ipynb) | Alur utama dari eksplorasi data hingga submission. |
| [`src/models/`](src/models/) | Implementasi model dari nol. |
| [`src/preprocessing.py`](src/preprocessing.py) | Split, pembersihan, konstruksi fitur, dan transformasi. |
| [`src/validation.py`](src/validation.py) | Evaluasi, tuning, analisis kesalahan, dan ekspor submission. |
| [`output/`](output/) | Prediksi, model tersimpan, grafik, dan arsip eksperimen. |
| [`docs/main.pdf`](docs/main.pdf) | Laporan tugas besar. Sumber dan petunjuk build ada di [`docs/`](docs/README.md). |

## Anggota kelompok

1. Jason Edward Salim (13524034)
2. Stefani Angeline Oroh (13524064)
3. Bryan Pratama Putra Hendra (13524067)
4. Angelina Andra Alanna (13524079)
5. Michael James Liman (13524106)
