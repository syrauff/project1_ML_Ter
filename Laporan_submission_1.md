# Laporan Proyek Machine Learning - Syahrul Ridho R Rauf

## Domain Proyek

Proyek ini berfokus pada analisis harga rumah berdasarkan berbagai fitur seperti lokasi, ukuran, jumlah kamar, dan lainnya. Tujuan utama adalah untuk membangun model prediksi harga rumah yang dapat membantu pengguna atau perusahaan dalam mengambil keputusan terkait investasi atau pembelian properti.

### Latar Belakang
Pasar real estat sering kali sangat dinamis dengan banyak faktor yang mempengaruhi harga.  Pertumbuhan  kebutuhan  akan  hunian yang  layak  seiring  dengan  meningkatnya  populasi menjadikan  sektor  ini  sebagai  salah  satu  investasi yang strategis. Pemanfaatan sebuah bangunan atau gedung sebagai tempat tinggal merupakan ide yang sangat baik.

Dengan menggunakan pendekatan berbasis data, prediksi harga rumah dapat dilakukan lebih akurat dibandingkan dengan metode tradisional. Penerapan pendekatan machine learning dalam menanganinya merupakan hal yang sudah mulai digunakan saat ini. Dengan menggunakan beberapa algoritma yang relevan dan membandingkannya sehingga mengetahui mana algortima yang sebaiknya digunakan dalam hal ini.  

[Prediksi Harga Rumah menggunakan Machine Learning Algoritma Regresi Linier](http://jurnal.unidha.ac.id/index.php/jteksis/article/view/1732) 

## Business Understanding

### Problem Statements
Berikut adalah beberapa problem statements yang diangkat:
- Bagaimana memprediksi harga rumah berdasarkan fitur yang ada?
- Fitur mana yang paling memengaruhi harga rumah?

### Goals

Berdasarkan problem statements yang ada, berikut adalah goals yang diharapkan dalam project ini:

- Membangun model prediksi harga rumah dengan performa tinggi.
- Mengidentifikasi fitur yang memiliki pengaruh besar terhadap harga rumah.

### Solution statements

- Menggunakan beberapa algoritma regresi seperti K-Nearest Neighbors (KNN), Random Forest, AdaBoost, dan Linear Regression.
- Membandingkan performa model berdasarkan Mean Squared Error (MSE) untuk memilih model terbaik.
- Melakukan scaling data numerik untuk memastikan performa optimal dari algoritma.

## Data Understanding

Dataset yang digunakan berasal kaggle dengan nama dataset `House price prediction`. Dari folder zip dataset yang didownload, data yangdigunakanadalah file dengan nama output.csv. Dataset yang digunakan berisi informasi mengenai properti, seperti jumlah kamar, luas tanah, dan harga. Dataset terdiri dari 4600 sampel.

Kondisi dataset yang telah didownload hanya perlu beberapa perbaikan, missing value dan duplikat data tidak terdapat pada dataset. Perbaikan yang diperlukan hanya pada tipe data pada tipe data.  

[Kaggle](https://www.kaggle.com/datasets/shree1992/housedata/data).

### Variabel-variabel pada House price prediction dataset adalah sebagai berikut:


- Dataset yang digunakan dalam proyek ini berisi 4600 data properti dengan 18 kolom, yaitu:

- date (object): Tanggal pencatatan data.

- price (float64): Harga properti dalam mata uang tertentu.

- bedrooms (float64): Jumlah kamar tidur pada properti.

- bathrooms (float64): Jumlah kamar mandi pada properti.

- sqft_living (int64): Luas bangunan (dalam square feet).

- sqft_lot (int64): Luas tanah (dalam square feet).

- floors (float64): Jumlah lantai pada properti.

- waterfront (int64): Indikator apakah properti memiliki akses ke tepi air (1: Ya, 0: Tidak).

- view (int64): Indeks yang menunjukkan kualitas pemandangan properti.

- condition (int64): Kondisi properti (1: sangat buruk, 5: sangat baik).

- sqft_above (int64): Luas bangunan di atas tanah (dalam square feet).

- sqft_basement (int64): Luas basement (dalam square feet).

- yr_built (int64): Tahun properti dibangun.

- yr_renovated (int64): Tahun properti terakhir direnovasi.

- street (object): Nama jalan lokasi properti.

- city (object): Nama kota lokasi properti.

- statezip (object): Kombinasi kode negara bagian dan kode pos lokasi properti.

- country (object): Nama negara lokasi properti.


## Data Preparation

### Langkah-langkah Persiapan Data:
1. **Mengonversi tipe data**:
   Kolom `date` dari yang sebelumnya *object* menjadi *datetime64*.
2. **Menghapus data yang tidak valid**:
   Beberapa sampel data di kolom `price` bernilai 0, hal ini tidak wajar karena harga rumah untuk data ini bersifat gratis.
3. **Handling outlier**:
   Outlier didefinisikan sebagai nilai yang berada di bawah batas bawah (Q1 - 1.5 * IQR) atau di atas batas atas (Q3 + 1.5 * IQR). Setelah nilai-nilai tersebut diidentifikasi, outlier baris yang mengandung outlier dari dataset dihapus. Setelah penghapusan data tersisa 3422.
4. **Encoding data**:
   Metode yang digunakan adalah one-hot encoding, yang secara otomatis membuat kolom baru untuk setiap kategori unik di dalam kolom kategorikal. Pada kode berikut, kita menggunakan pd.get_dummies() dari library Pandas
5. **Scaling Data Numerik**:
   Data numerik diskalakan menggunakan `StandardScaler` untuk memastikan distribusi data seragam:
   ```python
   scaler = StandardScaler()
   scaler.fit(X_train)
   X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
   ```
6. **Split Dataset**:
   Dataset dibagi menjadi data pelatihan (80%) dan data pengujian (20%) untuk evaluasi model:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
   ```

## Modeling

Beberapa algoritma yang digunakan untuk membangun model adalah:

1. **K-Nearest Neighbors (KNN)**:
   - KNN adalah algoritma berbasis instance yang memprediksi nilai target berdasarkan kedekatan data baru dengan data-data *neighbors* terdekatnya. Kedekatan ini diukur menggunakan metrik jarak, seperti Euclidean Distance. 
   - Parameter utama: `n_neighbors=19`. Jumlah *neighbors* terdekat yang digunakan untuk membuat prediksi. Nilai ini dipilih untuk menghindari overfitting (terlalu sedikit *neighbors*) atau underfitting (terlalu banyak *neighbors*).
   - Prediksi dilakukan dengan menghitung rata-rata harga (atau target) dari 19 tetangga terdekat untuk data baru.

2. **Random Forest (RF)**:
   - Random Forest adalah algoritma ensemble berbasis bagging yang membangun banyak pohon keputusan (decision trees) dan menggabungkan hasil prediksinya. Dengan memilih subset acak dari fitur dan data, Random Forest meningkatkan generalisasi model dan mengurangi overfitting.
   - Parameter `n_estimators=200`: Jumlah pohon keputusan dalam ensemble. Semakin banyak pohon, semakin stabil prediksi, namun dengan tambahan biaya komputasi. Parameter `max_depth=16`: Kedalaman maksimum pohon keputusan. Parameter ini membantu mengontrol kompleksitas model dan mengurangi risiko overfitting.
   - Algoritma membagi data ke dalam beberapa subset, melatih pohon pada subset tersebut, dan menggabungkan prediksi dari semua pohon (misalnya, dengan rata-rata dalam kasus regresi) untuk membuat hasil akhir.

3. **AdaBoost Regressor**:
   - AdaBoost (Adaptive Boosting) adalah algoritma boosting yang membangun model secara iteratif, di mana model baru fokus pada data yang sulit diprediksi oleh model sebelumnya. 
   - Parameter utama: `learning_rate=0.05`. Mengontrol kontribusi setiap model lemah (weak learner) terhadap hasil akhir. Nilai yang lebih kecil membuat proses pembelajaran lebih lambat tetapi stabil.
   - Model membangun serangkaian weak learners (biasanya pohon keputusan dangkal) dan menggabungkan prediksinya dengan bobot tertentu untuk meningkatkan akurasi secara keseluruhan.

4. **Linear Regression**:
   - Linear Regression adalah model regresi dasar yang mencoba menemukan hubungan linier antara fitur dan target.
   - Menggunakan parameter default, yang berarti tidak ada pengaturan khusus diterapkan pada model.
   - Model ini menghitung nilai koefisien linier yang meminimalkan Mean Squared Error (MSE) antara prediksi dan nilai target sebenarnya. Linear Regression sering digunakan sebagai baseline untuk membandingkan performa dengan model lain.

## Evaluation

**Proses Evaluasi:**
Proses evaluasi dilakukan menggunakan metrik Mean Squared Error (MSE), yang mengukur rata-rata kuadrat kesalahan antara nilai prediksi dan nilai aktual. Nilai MSE yang lebih kecil menunjukkan performa model yang lebih baik. Proses evaluasi dilakukan untuk data pelatihan (train) dan data pengujian (test).

Metrik evaluasi yang digunakan adalah Mean Squared Error (MSE):
```python
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting', 'LR'])
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
```

**Hasil Evaluasi:**

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train</th>
      <th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KNN</th>
      <td>21262995.144448</td>
      <td>48553166.185254</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>2636168.939598</td>
      <td>74487690.747625</td>
    </tr>
    <tr>
      <th>Boosting</th>
      <td>20632960.616099</td>
      <td>47761339.131888</td>
    </tr>
    <tr>
      <th>LR</th>
      <td>18103.620197</td>
      <td>7799739810.355667</td>
    </tr>
  </tbody>
</table>
</div>

- KNN: Memiliki performa cukup baik, tetapi masih terdapat gap antara data pelatihan dan pengujian.
- Random Forest (RF): MSE data pelatihan sangat rendah, namun pada data pengujian meningkat signifikan, yang menunjukkan kemungkinan overfitting.
- AdaBoost (Boosting): Memberikan hasil stabil dengan MSE yang lebih rendah dibandingkan KNN pada data pengujian.
- Linear Regression (LR): Meskipun MSE pada data pelatihan sangat rendah, model ini memiliki performa buruk pada data pengujian, menunjukkan model tidak mampu menangkap hubungan non-linier dengan baik.

- Model terbaik berdasarkan MSE pada data pengujian adalah **Random Forest**, dengan nilai MSE paling rendah.

**Prediksi:**
Prediksi harga dilakukan untuk satu sampel:
```python
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true': y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
```

**Kesimpulan:**
- **Problem statements:**

  - Model yang dibangun mampu memprediksi harga rumah berdasarkan fitur yang ada. 
  - Pemahaman fitur mana yang paling memengaruhi harga rumah dapat digunakan untuk strategi bisnis yang lebih baik.

- **Goals:**

  - Goals ini tercapai dengan AdaBoost, yang memiliki performa paling stabil berdasarkan evaluasi MSE. Untuk performa yang tinggi tidak bisa dipastikan, model mengeluarkan prediksi yang tidak sama tetapi memberikan prediksi yang mendekati hasil aslinya.
  - Fitur yang memiliki pengaruh besar terhadap harga rumah adalah variabel `sqft_living`. Kemudian disusul oleh `sqft_above` dan `bath_rooms`

- **Solution statements:**

  Dampak yang didapatkan dari menguji beberapa algoritma untuk prediksi harga rumah adalah mengetahui algoritma apa yang efektif terhadap prediksi harga rumah. Prediksi harga rumah akan lebih baik dilakukan dengan algoritma AdaBooster, meskipun predksi yang dilakukan tidak tepar, tetapi model dapat memberikan *output* yang mendekati hasil. 
  
**---Ini adalah bagian akhir laporan---**