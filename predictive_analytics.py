# %% [markdown]
# # Predictive Analytics - House Price Prediction 
# 
# 

# %% [markdown]
# #### Import Library yang dibutuhkan
# Import library  yang diperlukan untuk analisis dan pemodelan data.- Import pustaka dan modul yang diperlukan untuk analisis dan pemodelan data.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# %% [markdown]
# #### Membaca dataset
# 
# - Mendefinisikan path file dataset (`output.csv`) yang akan digunakan.
# - Membaca dataset menggunakan `pd.read_csv()` dan menyimpannya dalam variabel `df_house`.
# - Menampilkan beberapa baris pertama dari dataset untuk melihat struktur data.

# %%
path_to_file = './data/output.csv'
df_house = pd.read_csv(path_to_file)

df_house

# %% [markdown]
# ### Menampilkan informasi tentang dataset

# %%
df_house.info()

# %% [markdown]
# ### Memperbaiki type data
# 
# Mengonversi kolom `date` pada dataset menjadi tipe data `datetime` menggunakan `pd.to_datetime()` untuk mempermudah analisis berbasis waktu.
# 

# %%
df_house['date'] = pd.to_datetime(df_house['date'])

# %%
df_house.info()

# %% [markdown]
# ### Melihat Statistik data
# 
# Menampilkan statistik deskriptif dataset, seperti rata-rata (mean), standar deviasi (std), nilai minimum, kuartil, dan nilai maksimum untuk setiap kolom numerik.

# %%
df_house.describe()

# %% [markdown]
# ### Memeriksa data yang kemungkinan tidak valid
# 
# Menampilkan baris dalam dataset di mana nilai `price` sama dengan 0. Ini berguna untuk mengidentifikasi entri yang tidak valid atau tidak relevan. 
# 

# %%
df_house.loc[(df_house['price']==0)]

# %% [markdown]
# ### Menghapus data yang tidak valid berdasarkan statistik
# 
# - Menghapus baris dari dataset di mana nilai `price` kurang dari atau sama dengan 0.
# - Menampilkan ukuran dataset setelah penghapusan untuk memastikan jumlah baris telah berkurang.

# %%
df_house = df_house[df_house['price'] > 0]

# %%
df_house.shape

# %% [markdown]
# ### Mengidentifikasi outliers
# 
# - Mengidentifikasi kolom numerik dalam dataset.
# - Menghitung kuartil pertama (Q1), kuartil ketiga (Q3), dan rentang interkuartil (IQR) untuk setiap kolom numerik.
# - Menghapus outlier berdasarkan aturan IQR (nilai di luar rentang [Q1 - 1.5*IQR, Q3 + 1.5*IQR]).
# - Menampilkan ukuran dataset setelah outlier dihapus untuk memastikan perubahan.

# %%
numeric_columns = df_house.select_dtypes(include=['float64', 'int64']).columns

Q1 = df_house[numeric_columns].quantile(0.25)
Q3 = df_house[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Hapus outliers
df_house = df_house[~((df_house[numeric_columns] < (Q1 - 1.5 * IQR)) | 
                       (df_house[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Cek ukuran dataset setelah menghapus outliers
df_house.shape

# %%
df_house.describe()

# %% [markdown]
# ### Visualisasi Dataset 
# - Memisahkan kolom dataset menjadi fitur kategorikal dan numerikal.
# - Menyiapkan grid untuk membuat beberapa plot dalam satu figure.
# - Melakukan iterasi pada setiap fitur kategorikal dan membuat bar chart yang menampilkan distribusi nilai di masing-masing kolom.
# - Jika suatu fitur tidak ada, subplot dibiarkan kosong.

# %%
categorical_features = df_house.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = df_house.select_dtypes(include=['number']).columns.tolist()

# Setup untuk grid plot
num_features = len(categorical_features)
rows = (num_features + 2) // 3  
fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 6))  
axes = axes.flatten()  

# Loop melalui setiap feature dan buat bar chart
for i, feature in enumerate(categorical_features):
    if feature in df_house.columns:
        count = df_house[feature].value_counts()
        percent = 100 * df_house[feature].value_counts(normalize=True)
        df = pd.DataFrame({'Jumlah Sampel': count, 'Persentase': percent.round(1)})

        # Plot bar chart pada subplot yang sesuai
        count.plot(kind='bar', title=feature, ax=axes[i])
        axes[i].set_ylabel('Jumlah Sampel')
    else:
        # Kosongkan subplot jika feature tidak ada
        axes[i].axis('off')

# %% [markdown]
# #### Membuat fungsi visualisasi data berdasarkan price
# 
# - Fungsi untuk membuat bar chart berdasarkan kolom kategorikal tertentu dalam dataset.
# - Memvalidasi apakah kolom yang diberikan ada di dataset, lalu menghitung distribusi jumlah sampel dan persentasenya.
# - Membuat plot dengan opsi rotasi label untuk meningkatkan keterbacaan.

# %%
def plot_categorical_feature(df, feature, rotate=45):
    
    if feature in df.columns:
        # Hitung jumlah sampel dan persentase
        count = df[feature].value_counts()
        percent = 100 * df[feature].value_counts(normalize=True)
        feature_df = pd.DataFrame({'Jumlah Sampel': count, 'Persentase': percent.round(1)})

        # Plot bar chart
        plt.figure(figsize=(10, 6))
        count.plot(kind='bar', color='skyblue')
        plt.title(f'Distribusi Fitur Kategorikal: {feature}', fontsize=14)
        plt.xlabel('Kategori', fontsize=12)
        plt.ylabel('Jumlah Sampel', fontsize=12)
        plt.xticks(rotation=rotate, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Fitur '{feature}' tidak ditemukan dalam DataFrame.")

# %% [markdown]
# #### Visualisasi data kota dan kodepos berdasarkan jumlah sample
# 
# - Menggunakan fungsi `plot_categorical_feature` untuk membuat bar chart distribusi dari fitur kategorikal `city` dan `statezip`.
# - Label pada `statezip` diputar 90 derajat untuk keterbacaan lebih baik.

# %%
plot_categorical_feature(df_house, 'city')

# %%
plot_categorical_feature(df_house, 'statezip', 90)

# %% [markdown]
# #### Melihat distribusi data
# 
# - Membuat histogram untuk semua kolom numerik dalam dataset.
# - Histogram menunjukkan distribusi data untuk masing-masing kolom.

# %%
df_house.hist(bins=50, figsize=(20,15))
plt.show()

# %%
categorical_features_selected = ['statezip', 'city']

# %% [markdown]
# #### Melihat Visualisasi data
# 
# - Membuat bar chart menggunakan `seaborn.catplot` untuk menunjukkan rata-rata `price` terhadap kolom `statezip` dan `city`.
# - Label sumbu-x diputar 90 derajat untuk keterbacaan lebih baik.

# %%
# Perulangan untuk membuat plot
for col in categorical_features_selected:
    sns.catplot(
        x=col, y="price", kind="bar", 
        dodge=False, height=4, aspect=3, 
        data=df_house, hue=col, palette="Set3"
    )
    plt.title(f"Rata-rata 'price' Relatif terhadap {col}")
    plt.xticks(rotation=90)

# %%
sns.pairplot(df_house, diag_kind = 'kde')

# %% [markdown]
# #### Korelasi antar fitur numerik
# 
# - Membuat heatmap korelasi untuk kolom numerik dalam dataset.
# - Heatmap menunjukkan hubungan linier antara fitur-fitur numerik.

# %%
plt.figure(figsize=(10, 8))
correlation_matrix = df_house[numerical_features].corr().round(2)
 

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

# %% [markdown]
# #### One-Hot Encoding
# 
# - Mengonversi kolom kategorikal menjadi representasi numerik menggunakan metode one-hot encoding.
# - Menggunakan `drop_first=True` untuk menghindari multikolinearitas dengan membuang salah satu kategori.

# %%

df_house=pd.get_dummies(df_house,drop_first=True)

# %% [markdown]
# #### Menampilkan beberapa fitur yang memiliki sedikit korelasi  
# 
# - Membuat pair plot untuk beberapa fitur numerik untuk mengeksplorasi hubungan antar fitur.
# - Ukuran marker diatur kecil untuk mempermudah interpretasi data jika ada banyak titik.

# %%
sns.pairplot(df_house[['bedrooms','bathrooms','sqft_living', 'sqft_above']], plot_kws={"s": 4});

# %% [markdown]
# #### Mengatur split data 
# 
# - Membagi dataset menjadi data pelatihan dan data uji. 
# - Fitur `price` dan `date` dihapus dari fitur input (`X`).
# 

# %%
X = df_house.drop(["price", "date"],axis =1)
y = df_house["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# %%
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

# %% [markdown]
# #### Menyimpan model-model dalam Data Frame
# 
# Membuat DataFrame untuk menyimpan nilai Mean Squared Error (MSE) dari berbagai model regresi.

# %%
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting', 'LinearRegression'])

# %% [markdown]
# #### Melatih beberapa model
# 
# - Melatih model K-Nearest Neighbors dan menyimpan nilai MSE data pelatihan ke DataFrame `models`.
# - Melatih model Random Forest dan menyimpan nilai MSE data pelatihan ke DataFrame `models`.
# - Melatih model AdaBoost Regressor dan menyimpan nilai MSE data pelatihan ke DataFrame `models`.
# - Melatih model Linear Regression dan menyimpan nilai MSE data pelatihan ke DataFrame `models`. 

# %%
knn = KNeighborsRegressor(n_neighbors=19)
knn.fit(X_train, y_train)
 
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

# %%
param_grid = {'n_estimators': [50, 100, 150, 200, 300]}

rf = RandomForestRegressor(max_depth=16, random_state=55, n_jobs=-1)

grid = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid.fit(X_train, y_train)

print("Optimal n_estimators:", grid.best_params_['n_estimators'])
print("Best CV MSE:", -grid.best_score_)

# %%
RF = RandomForestRegressor(n_estimators=200, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)   

# %%
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

# %%
lr=LinearRegression()
lr.fit(X_train, y_train)
models.loc['train_mse','LinearRegression'] = mean_squared_error(y_pred=lr.predict(X_train), y_true=y_train)

# %% [markdown]
# #### Data Normalization
# 
# - Normalisasi data pada fitur numerik menggunakan `StandardScaler` dari `sklearn`. 
# - Skala data dilatih berdasarkan data training, lalu diterapkan pada data test untuk memastikan distribusi data seragam antara training dan testing.
# 

# %%
scaler = StandardScaler()
scaler.fit(X_train)

# %%
numerical_features = [feature for feature in numerical_features if feature != 'price']
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# %% [markdown]
# #### Model Evaluation Menggunakan Mean Squared Error (MSE)
# 
# Cell ini mengevaluasi performa setiap model menggunakan metrik Mean Squared Error (MSE) pada data training dan testing. Hasil MSE disimpan dalam DataFrame `mse` untuk analisis lebih lanjut. Hasil dibagi dalam ribuan untuk kemudahan interpretasi.
# 

# %%
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting', 'LR'])
 
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting, 'LR': lr}
 
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 
mse

# %% [markdown]
# #### Visualisasi performa model
# 
# Visualisasi performa model menggunakan bar chart horizontal. MSE untuk data testing diurutkan dari nilai tertinggi ke terendah untuk memberikan gambaran jelas model mana yang memiliki error terendah pada data testing.
# 

# %%
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

# %% [markdown]
# #### Prediksi dengan data random
# 
# Prediksi pada satu sampel data dari dataset test menggunakan semua model yang telah dilatih. Hasil prediksi dari setiap model dibandingkan dengan nilai aktual (y_true). Output disajikan dalam bentuk DataFrame untuk kemudahan interpretasi.
# 

# %%
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)


