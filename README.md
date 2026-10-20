# UTS Graf Sains Data: Rekomendasi Item dengan Neo4j dan KNN

| | |
|---|---|
| Nama | `Marshanda Claudia I` |
| NIM | `23031554014` |
| Kelas | `2023A` |
| GitHub | [audy2204](https://github.com/audy2204) |
---

## 1. Ringkasan Proyek

Proyek ini membangun graf interaksi e-commerce dari dataset **RetailRocket** (Roman Zykov, Kaggle) di **Neo4j**, melakukan *Vector Space Modelling* (VSM) dan proyeksi graf, menghitung similarity, lalu membuat sistem rekomendasi **KNN (user-based collaborative filtering)** yang dievaluasi dengan **Hit@5** dan **NDCG@5**.

Alur kerja:

```
events.csv (Kaggle)
   -> filter periode P1 dan P2 (Python)        -> data/p1/events.csv, data/p2/events.csv
   -> impor P1 ke Neo4j (import_events.py)     -> graf Visitor -> Item
   -> VSM User dan Item, pilih proyeksi        -> proyeksi Item-Item
   -> cosine similarity (Python) + Cypher      -> relasi SIMILAR_TO di graf
   -> KNN user + evaluasi pada P2              -> Hit@5, NDCG@5
```

## 2. Struktur Repository

```
UTS-Graf-Sains-Data/
├── README.md
├── requirements.txt
├── .gitignore
├── import_events.py                 # impor events.csv ke Neo4j secara batch
├── 2.1 Wrangling dan EDA.ipynb      # soal 2.1 nomor 1 sampai 8
├── 2.2 Analytics.ipynb              # soal 2.2 nomor 1 sampai 2
└── data/
    ├── p1/events.csv                # data periode P1 (training)
    └── p2/events.csv                # data periode P2 (testing)
```

> File `data/events.csv` asli tidak disertakan karena berukuran besar (lihat bagian 4).

## 3. Dataset dan Periode

Sumber: [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). Data asli memuat 2.756.101 baris interaksi dari 3 Mei 2015 sampai 18 September 2015 (1.407.580 visitor dan 235.061 item).

Fitur: `timestamp` (Unix epoch, milidetik), `visitorid`, `event` (`view`, `addtocart`, `transaction`), `itemid`, dan `transactionid`.

| Fold | Periode | Baris | view | addtocart | transaction |
|---|---|---|---|---|---|
| **P1** (training) | 14 Mei 2015 sampai 14 Juli 2015 | 1.274.033 | 1.232.775 | 31.026 | 10.232 |
| **P2** (testing) | 14 Juli 2015 sampai 14 September 2015 | 1.207.800 | 1.166.474 | 31.274 | 10.052 |

Filter memakai batas `[start, end)` sehingga tidak ada baris yang masuk ke dua periode. P1 selalu mendahului P2 untuk menghindari *data leakage*.

## 4. Cara Menjalankan

**Prasyarat:** Python 3.10 atau lebih baru, Neo4j (Desktop atau Server) yang aktif di `neo4j://localhost:7687`, dan database bernama `neo4j`.

1. **Buat dan aktifkan virtual environment**
   ```bash
   python -m venv myvenv
   myvenv\Scripts\activate          # Windows
   # source myvenv/bin/activate     # Linux / macOS
   ```
2. **Pasang dependensi**
   ```bash
   pip install -r requirements.txt
   ```
3. **Unduh dataset asli** dari Kaggle dan letakkan sebagai `data/events.csv`:
   ```bash
   curl -L -o ecommerce-dataset.zip https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset
   ```
   Ekstrak `events.csv` dari zip ke folder `data/`. Jika hanya ingin mereproduksi hasil, `data/p1` dan `data/p2` sudah tersedia di repo dan langkah ini boleh dilewati.
4. **Jalankan notebook secara berurutan**
   1. `2.1 Wrangling dan EDA.ipynb` (membuat `data/p1`, `data/p2`, mengisi graf Neo4j, dan menghitung similarity).
   2. `2.2 Analytics.ipynb` (KNN dan evaluasi; membaca `data/p1` dan `data/p2`).

   Notebook akan meminta password Neo4j lewat `getpass`, sehingga password tidak tersimpan di kode.

> **Peringatan:** pada notebook 2.1, jika `RUN_IMPORT = True`, sel impor menjalankan `MATCH (n) DETACH DELETE n` yang **menghapus seluruh isi database** sebelum impor ulang. Ubah menjadi `False` jika graf P1 sudah terisi, dan jangan gunakan pada *shared database*.

> **Catatan versi driver:** kode memakai `driver.execute_query(...)` yang membutuhkan paket `neo4j` versi 5.5 atau lebih baru, sehingga pin `neo4j==4.2.1` pada contoh di naskah soal tidak dipakai di `requirements.txt` ini.

## 5. Hasil dan Penjelasan

### 2.1 Wrangling dan EDA

**No. 1 dan 2: Periode dan filter.** Dua periode dipilih seperti pada bagian 3, difilter pada kolom `timestamp`, lalu disimpan ke `data/p1/events.csv` dan `data/p2/events.csv`. Jumlah relasi di Neo4j (1.274.033) sama dengan jumlah baris P1.

**No. 3: Komposisi subgraph.**

```cypher
MATCH (n) RETURN labels(n)[0] AS jenis_simpul, count(*) AS jumlah ORDER BY jumlah DESC
MATCH ()-[r]->() RETURN type(r) AS jenis_relasi, count(*) AS jumlah ORDER BY jumlah DESC
```

`labels(n)[0]` mengambil label pertama tiap simpul, dan `count(*)` menghitung simpul per label (Cypher otomatis melakukan *group by* pada kolom non-agregat).

| Simpul | Jumlah | Relasi | Jumlah |
|---|---|---|---|
| Visitor | 650.722 | VIEWED | 1.232.775 |
| Item | 168.452 | ADDED_TO_CART | 31.026 |
| **Total** | **819.174** | BOUGHT | 10.232 |

Strukturnya adalah graf bipartit berarah (Visitor -> Item) dengan tiga jenis relasi.

**No. 4: VSM User.** Bobot interaksi: view = 1, addtocart = 3, bought = 5 (diambil bobot tertinggi per pasangan User-Item). Matriks sparse User x Item berukuran **650.722 x 168.452** dengan 986.731 elemen non-nol (density 0,0009%, sparsity 99,9991%). Rata-rata 1,52 elemen non-nol per user (median 1, maksimum 1.565), dan 80% user hanya punya satu item. Plot networkx menampilkan graf bipartit dari 5 user acak (seed 42): biru = User, oranye = Item.

**No. 5: VSM Item.** Transpos dari matriks di atas, berukuran **168.452 x 650.722**, dengan rata-rata 5,86 elemen non-nol per item (median 2, maksimum 1.302). Item dengan satu interaksi hanya 41,5%.

**No. 6: Proyeksi terbaik.**

| Kriteria | User-User | Item-Item |
|---|---|---|
| Jumlah simpul | 650.722 | **168.452** |
| Sisi maksimum n(n-1)/2 | 2,1 x 10^11 | **1,4 x 10^10** |
| Estimasi sisi (batas atas) | 21,78 juta | **11,14 juta** |
| Vektor dengan 1 elemen | 80,0% | **41,5%** |
| Simpul P2 yang sudah ada di P1 | 9,85% | **88,13%** |

Proyeksi **Item-Item** dipilih karena lebih efisien (simpul dan sisi lebih sedikit), lebih efektif (vektor lebih padat sehingga cosine lebih bermakna), dan lebih stabil terhadap waktu (88,13% item P2 sudah ada di P1, sedangkan user hanya 9,85%).

**No. 7: Similarity pada proyeksi Item-Item.** Matriks dibuat biner, dinormalisasi L2 per baris, lalu cosine similarity dihitung dengan perkalian matriks sparse per *chunk* 500 baris (hanya item berderajat >= 5, yaitu 44.937 item), dan diambil top-5 tetangga tiap item. Hasilnya 172.604 pasangan.

(a) Record acuan dipilih acak dan reproducible dengan `numpy.random.default_rng(42)` dari item berderajat >= 20, yaitu **Item 45237**.

```cypher
MATCH (a:Item {itemId: $ref})<-[:VIEWED|ADDED_TO_CART|BOUGHT]-(h:Visitor)
WITH a, collect(DISTINCT h) AS ha
UNWIND ha AS h
MATCH (h)-[:VIEWED|ADDED_TO_CART|BOUGHT]->(b:Item) WHERE b <> a
WITH a, ha, b, count(DISTINCT h) AS common
MATCH (b)<-[:VIEWED|ADDED_TO_CART|BOUGHT]-(hb:Visitor)
WITH a, ha, b, common, count(DISTINCT hb) AS nb WHERE nb >= $min_deg
RETURN b.itemId AS tetangga, common, nb,
       toFloat(common) / sqrt(size(ha) * nb) AS cosine
ORDER BY cosine DESC, tetangga ASC LIMIT 5
```

Query ini mencari visitor yang mengakses item acuan, mencari item lain yang diakses visitor yang sama (`common`), menghitung total visitor tiap item tersebut (`nb`), lalu menghitung cosine biner `common / sqrt(|A| x |B|)`.

| Tetangga | common | nb | cosine |
|---|---|---|---|
| 331381 | 10 | 21 | 0,412393 |
| 348804 | 5 | 9 | 0,314970 |
| 15153 | 5 | 12 | 0,272772 |
| 431232 | 5 | 13 | 0,262071 |
| 33425 | 3 | 5 | 0,253546 |

Hasil Cypher dicocokkan dengan hasil Python: cocok 5 dari 5, selisih maksimum 3,9 x 10^-7.

(b) Similarity ditulis kembali ke graf sebagai relasi `SIMILAR_TO`:

```cypher
UNWIND $rows AS r
MATCH (a:Item {itemId: r.a}), (b:Item {itemId: r.b})
MERGE (a)-[s:SIMILAR_TO]->(b)
SET s.score = r.score, s.metode = 'cosine-biner', s.periode = 'P1'
```

`MERGE` mencegah relasi terduplikasi jika sel dijalankan ulang. Query pembacaan kembali menampilkan lima tetangga dan skor yang sama persis dengan tabel di atas.

**No. 8: Sebelum dan sesudah similarity.**

```cypher
MATCH ()-[r]->() RETURN type(r) AS relasi, count(*) AS jumlah ORDER BY relasi
```

| Relasi | Sebelum | Sesudah |
|---|---|---|
| ADDED_TO_CART | 31.026 | 31.026 |
| BOUGHT | 10.232 | 10.232 |
| VIEWED | 1.232.775 | 1.232.775 |
| **SIMILAR_TO** | **0** | **172.604** |

Graf asli tidak berubah; hanya bertambah relasi `SIMILAR_TO` antar Item. Visualisasi networkx di notebook membandingkan relasi Visitor -> Item 45237 (sebelum) dengan Item 45237 -> 5 tetangga beserta skor cosine (sesudah).

### 2.2 Analytics

**No. 1a: Pemilihan user dan skrip KNN.** Kandidat user adalah mereka yang memiliki event `addtocart` atau `transaction` di P2, minimal 3 item di P1, dan minimal 1 item baru di P2 (442 user). Pemilihan memakai `numpy.random.default_rng(42).choice(...)` pada daftar kandidat yang diurutkan, sehingga selalu menghasilkan **user 163049** (19 item di P1; item relevan di P2: 215051).

Skrip KNN bersifat user-based:
1. Matriks User x Item (bobot 1/3/5) dinormalisasi L2, lalu cosine similarity user target ke semua user dihitung dengan perkalian sparse.
2. Dari 1.000 user terdekat dipilih **k = 5 tetangga** yang punya minimal 2 item dan menyumbang minimal 1 item baru bagi target. Filter ini diperlukan karena KNN mentah menghasilkan tetangga yang tidak berguna (contoh: user 555036 dengan cosine 0,77 tetapi hanya 2 item dan 0 item baru).
3. Setiap item milik tetangga yang belum pernah dilihat target diberi skor `sum(cosine x bobot interaksi)`.
4. Item diurutkan dari skor tertinggi (seri diputus dengan itemId terkecil), dan jika kurang dari 5 dilengkapi item terpopuler.

| Tetangga | Cosine | Jumlah item | Item baru bagi target |
|---|---|---|---|
| 939751 | 0,7337 | 3 | 2 |
| 65270 | 0,6269 | 10 | 1 |
| 1029863 | 0,6114 | 14 | 3 |
| 182312 | 0,6100 | 9 | 1 |
| 1312606 | 0,5921 | 13 | 3 |

**No. 1b: 3 item yang diiklankan (k = 5).**

| Peringkat | Item | Skor |
|---|---|---|
| 1 | 269085 | 1,3258 |
| 2 | 9877 | 1,2035 |
| 3 | 322341 | 1,2035 |

Dua item berikutnya pada top-5 adalah 142324 dan 82377.

**No. 2: Hit@5 dan NDCG@5.** Relevansi biner: sebuah item relevan jika dibeli atau dimasukkan ke keranjang pada P2 dan belum pernah muncul di P1.

- Hit@k = 1 jika ada minimal satu item relevan di top-k, selain itu 0.
- NDCG@k = DCG / IDCG, dengan DCG = sum(rel_i / log2(i + 1)).

Untuk user 163049, item relevan 215051 tidak masuk top-5, sehingga **Hit@5 = 0** dan **NDCG@5 = 0**. Agar tidak bergantung pada satu user, evaluasi tambahan pada 200 user acak (seed 42) menghasilkan **Hit@5 rata-rata = 0,05** dan **NDCG@5 rata-rata = 0,0274**. Nilai yang rendah ini wajar karena data sangat sparse dan hanya sekitar 9,85% user P2 yang sudah ada di P1 (*cold-start*), sehingga KNN berbasis user sulit menebak satu item relevan di antara ratusan ribu item.
