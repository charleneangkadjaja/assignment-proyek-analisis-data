# E-Commerce Data Analysis Project

## Deskripsi Proyek
Dashboard ini menampilkan hasil analisis data e-commerce Brazil (Olist, 2016–2018) yang mencakup:
- Tren penjualan bulanan
- Produk terlaris dan rating pelanggan
- Pola waktu transaksi (hari & jam)
- Distribusi market per negara bagian
- Segmentasi pelanggan (RFM Analysis)
- Rekomendasi strategis berbasis data

## Struktur Direktori
submission
├── dashboard
│   ├── main_data.csv        # Dataset utama (hasil cleaning & join)
│   └── dashboard.py         # Aplikasi Streamlit
├── data
│   ├── orders_dataset.csv
│   ├── order_items_dataset.csv
│   ├── customers_dataset.csv
│   ├── products_dataset.csv
│   ├── product_category_name_translation.csv
│   ├── order_reviews_dataset.csv
│   └── geolocation_dataset.csv
├── notebook.ipynb           # Jupyter Notebook analisis lengkap
├── README.md
├── requirements.txt
└── url.txt


## Setup 
1. Virtual Environment (venv): `pyhton -m venv venv` dan `venv\Scripts\Activate.ps1`
2. Upgrade pip: `python -m pip install --upgrade pip`
3. Install Library Utama : `pip install pandas numpy matplotlib seaborn streamlit babel folium branca ipython`
4. Run Dashboard: `streamlit run dashboard/dashboard.py`

## Run dashboard
cd submission
streamlit run dashboard/dashboard.py

## Insight Utama
* Kategori produk terlaris
* Pola pembelian berdasarkan waktu
* Distribusi pelanggan per wilayah