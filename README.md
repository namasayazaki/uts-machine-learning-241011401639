# UTS Machine Learning – Linear Regression & Logistic Regression

- **Nama  :** Zaki Zainul Muttaqin
- **NIM   :** 241011401639
- **Kelas :** Reg B
- **random_state :** 39 (2 digit terakhir NIM)

## Isi Repositori
| File | Keterangan |
|---|---|
| `tugas_UTS.ipynb` | Notebook lengkap (semua soal A1–A6, B1–B6, C, dan Bonus) beserta output |
| `Ads_Data.csv` | Data biaya iklan vs penjualan (40 data) |
| `Email_Data.csv` | Data jumlah link, kata promo, dan label spam (60 data) |

Model dan metrik (train_test_split, LinearRegression, LinearRegressionGD, LogisticRegression, MAE/MSE/RMSE/R², accuracy, confusion matrix, precision, recall, F1) diimplementasikan sendiri dengan NumPy. scikit-learn hanya dipakai pada soal Bonus.

## Ringkasan Hasil

### Bagian A – Linear Regression (data Ads)
- Persamaan (normal equation): ŷ = 29,4296 + 8,9456 × BiayaIklan
- Data training/testing: 32 / 8

| Data | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Training | 4,6829 | 40,5038 | 6,3643 | 0,9387 |
| Testing | 4,0044 | 19,8542 | 4,4558 | 0,9395 |

- Gradient descent (5000 iterasi): lr 0,001 terlalu lambat (belum konvergen), lr 0,02 konvergen sama dengan normal equation, lr 0,1 divergen (overflow). Batas lr stabil ≈ 0,0268.
- Prediksi: 4,5 juta → 69,68 juta; 7 juta → 92,05 juta; 12 juta → 136,78 juta (ekstrapolasi, kurang dapat dipercaya).

### Bagian B – Logistic Regression (data Email)
- Persamaan: P(Spam) = σ(−8,7309 + 0,8172 × JumlahLink + 0,7414 × KataPromo)
- Data training/testing: 45 / 15
- Data testing (threshold 0,5): accuracy 0,9333; precision 0,8333; recall 1,0; F1 0,9091; confusion matrix [[9, 1], [0, 5]]

| Threshold | FP | FN | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| 0,3 | 5 | 0 | 0,5000 | 1,0 | 0,6667 |
| 0,5 | 1 | 0 | 0,8333 | 1,0 | 0,9333 |
| 0,7 | 0 | 0 | 1,0000 | 1,0 | 1,0000 |

- Threshold yang dipilih: **0,7** (FP lebih merugikan daripada FN).
- Prediksi email baru (threshold 0,7): E1 Bukan Spam (0,0016), E2 Bukan Spam (0,4509), E3 Spam (0,9976), E4 Spam (0,7424).

### Bonus – scikit-learn
- Linear Regression: identik dengan implementasi NumPy.
- Logistic Regression: koefisien default scikit-learn berbeda karena regularisasi L2 (C=1); dengan C=1e10 hasilnya hampir sama dengan NumPy.
