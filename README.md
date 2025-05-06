Project ini dibuat sebagai bagian dari Ujian Tengah Semester (UTS)
Tujuan dari program Python ini adalah membangun dan mengevaluasi model klasifikasi berbasis pohon keputusan (Decision Tree) untuk memprediksi apakah sebuah ponsel 
memiliki fitur 4G/LTE berdasarkan spesifikasi lainnya (seperti RAM, ROM, sistem operasi, dll), menggunakan dataset dari Kaggle.

Library yang diimpor meliputi:
  kagglehub: untuk mengunduh dataset dari Kaggle.
  pandas: untuk manipulasi data (CSV ke DataFrame).
  sklearnt-learn: untuk preprocessing, modeling, evaluasi, dan visualisasi.
  matplotlib: untuk menampilkan pohon keputusan dalam bentuk gambar.
  
Lalu pada bagian:
path = kagglehub.dataset_download("pratikgarai/mobile-phone-specifications-and-prices")
data_path = path + "/ndtv_data_final.csv"
df = pd.read_csv(data_path)
Dataset spesifikasi dan harga ponsel diunduh dari Kaggle dan dibaca ke dalam DataFrame df

Pada bagian:
rint(df.head())
print(df.info())


Menampilkan beberapa baris awal dan informasi umum kolom dataset, termasuk tipe data dan missing values.
Lalu di bagian:
df = df.drop(columns=['Phone Name'])
df = df.dropna()

Menghapus kolom tidak berguna dan mengatasi nilai yang hilang (missing value)

Pada bagian:
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])
    Mengubah kolom yang bertipe objek (biasanya string seperti 'Android', 'Yes', 'No') menjadi angka agar bisa diproses oleh model machine learning.

Lalu dibagian:
x = df.drop(columns=['Price'])      
y = df['4G/ LTE']
Fitur (X) adalah semua kolom kecuali 'Price'.
Target (Y) adalah kolom '4G/ LTE' yang ingin diprediksi.

Pada bagian:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)

Dataset dibagi menjadi data latih dan uji (80% : 20%), model pohon keputusan dilatih menggunakan data latih.

Bagian ini:
accuracy = dtree.score(x_test, y_test)
print(f"Akurasi model: {accuracy*100:.2f}%")

Mengukur akurasi model pada data uji.

Lalu pada bagian ini:
plt.figure(figsize=(50, 30))
tree.plot_tree(dtree, filled=True)
plt.show()

Menampilkan struktur pohon keputusan yang dibentuk model.
