# Berikan tampilan rekomendasi strategis yang dapat dilakukan oleh pihak e-commerce untuk meningkatkan penjualan dan kepuasan pelanggan berdasarkan hasil analisis data yang telah dilakukan. 
# Rekomendasi ini harus didukung oleh visualisasi yang jelas dan mudah dipahami, serta disertai dengan penjelasan singkat mengenai alasan di balik setiap rekomendasi yang diberikan.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import os



# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard E-commerce Analysis", layout="wide", initial_sidebar_state="expanded")

PALETTE = {
    'primary'   :'#1565C0',
    'secondary' :'#5C9BD6',
    'success'   : '#2E7D32',
    'warning'   : '#E65100',
    'danger'    : '#C62828',
    'neutral'   : '#9E9E9E',
    'bg'        : '#FAFAFA',
}

SEG_COLORS = {
    'Champions'          : '#1B5E20',
    'Loyal Customers'    : '#4CAF50',
    'Potential Loyalists': '#8BC34A',
    'New Customers'      : '#2196F3',
    'At Risk'            : '#FF9800',
    'Churned'            : '#F44336',
    'Needs Attention'    : '#9C27B0',
}

plt.rcParams.update({
    'figure.facecolor' : PALETTE['bg'],
    'axes.facecolor'   : PALETTE['bg'],
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.35,
    'grid.linestyle'   : '--',
})

# Load & Preprocessing Data
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)  # folder dashboard/
    file_path = os.path.join(base_path, 'main_data.csv')
    df = pd.read_csv(file_path)
    
    datetime_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
        'shipping_limit_date'
    ]

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['freight_ratio'] = (df['freight_value'] / df['price'].replace(0, np.nan)).round(3)
    return df

# Build RFM
@st.cache_data
def build_rfm(df):
    now = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg(
        Recency = ('order_purchase_timestamp', lambda x: (now - x.max()).days),
        Frequency = ('order_id', 'nunique'),
        Monetary = ('price', 'sum')
    ).reset_index()
    
    rfm['R_score'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    
    def segment(row):
        if row['RFM_score'] >= 13:
            return 'Champions'
        elif row['RFM_score'] >= 10:
            return 'Loyal Customers'
        elif row['RFM_score'] >= 7:
            return 'Potential Loyalists'
        elif row['R_score'] >= 4  and row['F_score'] <= 2:
            return 'New Customers'
        elif row['M_score'] <= 2  and row['F_score'] >= 3:
            return 'At Risk'
        elif row['R_score'] <= 2:
            return 'Churned'
        else:
            return 'Needs Attention'
    
    rfm['Segment'] = rfm.apply(segment, axis=1)
    return rfm
  
df = load_data()



# Sidebar untuk Filter
with st.sidebar:
    st.title("E-Commerce Brazil Dashboard")
    st.caption("Dataset 2016 - 2018")
    st.subheader("Filter Data")
    
    min_date=df['order_purchase_timestamp'].min().date()
    max_date=df['order_purchase_timestamp'].max().date()
    
    data_range = st.date_input(
        "Rentang Waktu",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(data_range) == 2:
        start_date = pd.to_datetime(data_range[0])
        end_date = pd.to_datetime(data_range[1])
    else:
        start_date = pd.to_datetime(min_date)
        end_date = pd.to_datetime(max_date)
        
    all_state = sorted(df['customer_state'].dropna().unique())
    sel_state = st.multiselect("Pilih State", all_state, default=all_state)
    
    all_cats = sorted(df['product_category'].dropna().unique())
    sel_cats = st.multiselect("Pilih Kategori Produk", all_cats, default=all_cats)
    
    st.markdown("---")
    st.caption("Charlene Manuella Angkadjaja")
    st.caption("Dicoding · ID: CDCC244D6X2377")
    
    
# Filter Data
filtered_df = df[
    (df['order_purchase_timestamp'] >= start_date) &
    (df['order_purchase_timestamp'] <= end_date) &
    (df['customer_state'].isin(sel_state)) &
    (df['product_category'].isin(sel_cats))
].copy()

# Header
st.title("E-commerce Dashboard")
st.caption(f"Data periode: {start_date.date()} s/d {end_date.date()}  ·  {len(filtered_df):,} transaksi")
st.markdown("---")


# Metrics KPI
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📦 Total Pesanan",       f"{filtered_df['order_id'].nunique():,}")
k2.metric("💰 Total Revenue",       f"R$ {filtered_df['price'].sum():,.0f}")
k3.metric("⭐ Avg Review Score",     f"{filtered_df['review_score'].mean():.2f}")
k4.metric("🚚 Avg Delivery (hari)", f"{filtered_df['delivery_days'].mean():.1f}")
k5.metric("📦 Avg Freight Ratio",   f"{filtered_df['freight_ratio'].mean():.2f}")
st.markdown("---")


# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Tren Penjualan",
    "Produk & Rating",
    "Pola Transaksi",
    "Geospatial",
    "RFM & Segmentasi",
    "Rekomendasi Strategis"
])

# Tab 1 - Tren Penjualan
with tab1:
    st.subheader("Tren Penjualan")
    
    monthly = filtered_df.groupby('year_month').agg(
            total_orders    = ('order_id', 'nunique'),
            total_revenue   = ('price', 'sum'),
            avg_score       = ('review_score', 'mean')
        ).reset_index().sort_values('year_month')    
    
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Jumlah Pesanan per Bulan**")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(range(len(monthly)), monthly['total_orders'],
                marker='o', color=PALETTE['primary'], linewidth=2.5,
                markersize=5, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(range(len(monthly)), monthly['total_orders'], alpha=0.12, color=PALETTE['primary'])
        peak_idx = monthly['total_orders'].idxmax()
        peak_pos = monthly.index.get_loc(peak_idx)
        ax.scatter(peak_pos, monthly.loc[peak_idx, 'total_orders'],color=PALETTE['danger'], s=120, zorder=5)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(monthly['year_month'], rotation=45, ha='right', fontsize=8)
        
        ax.set_ylabel("Jumlah Pesanan (unit)", fontsize=10)
        ax.set_xlabel("Bulan", fontsize=10)
        ax.set_title(f"Puncak: {monthly.loc[peak_idx, 'year_month']} ({monthly.loc[peak_idx, 'total_orders']} pesanan)", fontsize=10, color=PALETTE['danger'])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        plt.tight_layout()
        st.pyplot(fig)

    with col_b:
        st.markdown("**Total Revenue per Bulan**")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(range(len(monthly)), monthly['total_revenue'], color=PALETTE['secondary'], edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(monthly['year_month'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Revenue (R$)", fontsize=10)
        ax.set_xlabel("Bulan", fontsize=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'R${x/1e6:.1f}M'))
        plt.tight_layout()
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("Tren per Kategori (Top 5 Terlaris)")
    top_cats = filtered_df.groupby('product_category')['order_id'].nunique().sort_values(ascending=False).head(5).index.tolist()
    
    cat_monthly = (
        filtered_df[filtered_df['product_category'].isin(top_cats)]
        .groupby(['year_month','product_category'])
        .agg(total_orders=('order_id','nunique')).reset_index()
        .sort_values('year_month')
    )
    ym_list     = sorted(cat_monthly['year_month'].unique())
    ym_idx      = {ym: i for i, ym in enumerate(ym_list)}
    colors_cat  = ['#1565C0','#E65100','#2E7D32','#6A1B9A','#00838F']
 
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cat in enumerate(top_cats):
        d    = cat_monthly[cat_monthly['product_category'] == cat]
        xpos = [ym_idx[ym] for ym in d['year_month']]
        ax.plot(xpos, d['total_orders'], marker='o', linewidth=2, label=cat,
                color=colors_cat[i], markersize=4, markerfacecolor='white', markeredgewidth=1.5)
    ax.set_xticks(range(len(ym_list)))
    ax.set_xticklabels(ym_list, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Jumlah Pesanan (unit)", fontsize=10)
    ax.set_xlabel("Bulan", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    plt.tight_layout()
    st.pyplot(fig)
 
 
    st.markdown("---")
    st.subheader("Analisis Lanjutan: Lonjakan November")
    monthly['bulan'] = monthly['year_month'].str[5:7].astype(int)
    nov_data = monthly[monthly['bulan'] == 11]
    peak_month_str = monthly.loc[monthly['total_orders'].idxmax(), 'year_month']
 
    peak_cat = (
        filtered_df[filtered_df['year_month'] == peak_month_str]
        .groupby('product_category')
        .agg(total_orders=('order_id','nunique'))
        .sort_values('total_orders', ascending=False).head(5).reset_index()
    )
 
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.markdown("**Pesanan di Setiap Bulan November**")
        fig, ax = plt.subplots(figsize=(7, 4))
        b = ax.bar(nov_data['year_month'], nov_data['total_orders'],
                   color=PALETTE['primary'], edgecolor='white', width=0.5)
        for bar in b:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                    f'{int(bar.get_height()):,}', ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel("Jumlah Pesanan (unit)", fontsize=10)
        ax.set_xlabel("Tahun–Bulan", fontsize=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        plt.tight_layout()
        st.pyplot(fig)

    with col_n2:
        st.markdown(f"**Top 5 Kategori Pendorong di {peak_month_str}**")
        palette_cat2 = [PALETTE['warning'] if i == 0 else '#FFAB76' for i in range(len(peak_cat))]
        fig, ax = plt.subplots(figsize=(7, 4))
        bh = ax.barh(peak_cat['product_category'][::-1], peak_cat['total_orders'][::-1],
                     color=palette_cat2[::-1], edgecolor='white', height=0.6)
        for bar in bh:
            ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
                    f'{int(bar.get_width()):,}', va='center', fontsize=9, fontweight='bold')
        ax.set_xlabel("Jumlah Pesanan (unit)", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        plt.tight_layout(); st.pyplot(fig)
        
        
# Tab 2 - Produk & Rating
with tab2:
    st.subheader("Top 10 Kategori Terlaris & Rating")
    prod = (
        filtered_df.groupby('product_category').agg(
            total_orders      = ('order_id',      'nunique'),
            total_revenue     = ('price',          'sum'),
            avg_review_score  = ('review_score',  'mean'),
            avg_freight_ratio = ('freight_ratio', 'mean')
        ).sort_values('total_orders', ascending=False).head(10).reset_index()
    )
    prod['avg_review_score'] = prod['avg_review_score'].round(2)
    prod['rank_orders'] = prod['total_orders'].rank(ascending=False)
    prod['rank_rating'] = prod['avg_review_score'].rank(ascending=False)
    prod['gap_score']   = (prod['rank_orders'] - prod['rank_rating']).round(1)


    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Jumlah Order & Rating Rata-rata**")
        fig, ax1 = plt.subplots(figsize=(8, 5))
        palette_p = [PALETTE['primary'] if i == 0 else PALETTE['secondary'] for i in range(len(prod))]
        bars = ax1.bar(range(len(prod)), prod['total_orders'], color=palette_p, edgecolor='white')
        for bar in bars:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+60,
                     f'{int(bar.get_height()):,}', ha='center', fontsize=7.5, color=PALETTE['primary'])
        ax1.set_xticks(range(len(prod)))
        ax1.set_xticklabels(prod['product_category'], rotation=40, ha='right', fontsize=8)
        ax1.set_ylabel("Jumlah Order (unit)", color=PALETTE['primary'], fontsize=10)
        ax1.tick_params(axis='y', labelcolor=PALETTE['primary'])
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        ax2 = ax1.twinx()
        ax2.plot(range(len(prod)), prod['avg_review_score'], color=PALETTE['warning'],
                 marker='o', linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
        for i, (_, row) in enumerate(prod.iterrows()):
            ax2.annotate(f"{row['avg_review_score']:.2f}", xy=(i, row['avg_review_score']),
                         xytext=(0,8), textcoords='offset points',
                         fontsize=7.5, ha='center', color=PALETTE['warning'])
        ax2.set_ylabel("Rating Rata-rata (1–5)", color=PALETTE['warning'], fontsize=10)
        ax2.tick_params(axis='y', labelcolor=PALETTE['warning'])
        ax2.set_ylim(3.0, 5.0)
        ax1.set_title("Top 10 Kategori Terlaris & Rating\nSumber: Olist Dataset 2016–2018",
                      fontsize=11, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig)
 
    with col_p2:
        st.markdown("**Gap Score: Popularitas vs Kepuasan**")
        top10_gap  = prod.sort_values('gap_score', ascending=False)
        colors_gap = [PALETTE['danger'] if g > 0 else PALETTE['success'] for g in top10_gap['gap_score']]
        fig, ax = plt.subplots(figsize=(8, 5))
        bh = ax.barh(top10_gap['product_category'][::-1], top10_gap['gap_score'][::-1],
                     color=colors_gap[::-1], edgecolor='white', height=0.65)
        ax.axvline(x=0, color='#333', linewidth=1)
        for bar, val in zip(bh, top10_gap['gap_score'][::-1]):
            offset = 0.2 if val >= 0 else -0.2
            ha     = 'left' if val >= 0 else 'right'
            ax.text(bar.get_width()+offset, bar.get_y()+bar.get_height()/2,
                    f'{val:+.1f}', va='center', ha=ha, fontsize=9, fontweight='bold',
                    color=PALETTE['danger'] if val > 0 else PALETTE['success'])
        ax.set_xlabel("Gap Score (rank popularitas − rank rating)", fontsize=10)
        ax.set_ylabel("Kategori Produk", fontsize=10)
        ax.set_title("Merah = populer tapi rating rendah\nHijau = rating tinggi tapi kurang populer",
                     fontsize=10, pad=10)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=PALETTE['danger'], label='Populer, rating rendah'),
                            Patch(facecolor=PALETTE['success'], label='Rating tinggi, kurang populer')],
                  fontsize=8, framealpha=0.85)
        plt.tight_layout()
        st.pyplot(fig)
 
    st.markdown("---")
    st.subheader("Hambatan Ongkos Kirim per Kategori")
    freight_cat = (
        filtered_df.groupby('product_category').agg(
            total_items       = ('order_item_id', 'count'),
            avg_price         = ('price',          'mean'),
            avg_freight       = ('freight_value',  'mean'),
            avg_freight_ratio = ('freight_ratio',  'mean'),
        ).dropna().reset_index()
    )
    freight_cat['avg_freight_ratio'] = freight_cat['avg_freight_ratio'].round(3)
    top_freight = freight_cat.nlargest(10, 'avg_freight_ratio')
 
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**Top 10 Kategori — Freight-to-Price Ratio Tertinggi**")
        colors_fr = [PALETTE['danger'] if r > 0.3 else '#EF9A9A' for r in top_freight['avg_freight_ratio']]
        fig, ax = plt.subplots(figsize=(8, 5))
        bh2 = ax.barh(top_freight['product_category'][::-1], top_freight['avg_freight_ratio'][::-1],
                      color=colors_fr[::-1], edgecolor='white', height=0.65)
        ax.axvline(x=0.3, color='#333', linestyle='--', linewidth=1.5, label='Batas risiko (>0.3)')
        for bar, val in zip(bh2, top_freight['avg_freight_ratio'][::-1]):
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=8.5, fontweight='bold',
                    color=PALETTE['danger'] if val > 0.3 else '#555')
        ax.set_xlabel("Freight-to-Price Ratio", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_title("Merah gelap = di atas batas risiko 0.3", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
 
    with col_f2:
        st.markdown("**Kategori dengan Freight Ratio > 0.3 (Prioritas Subsidi)**")
        risiko = freight_cat[freight_cat['avg_freight_ratio'] > 0.3].sort_values('avg_freight_ratio', ascending=False)
        st.dataframe(
            risiko[['product_category','avg_freight_ratio','total_items','avg_price','avg_freight']]
            .rename(columns={
                'product_category' : 'Kategori',
                'avg_freight_ratio': 'Freight Ratio',
                'total_items'      : 'Total Item',
                'avg_price'        : 'Avg Price (R$)',
                'avg_freight'      : 'Avg Freight (R$)'
            })
            .style.format({'Freight Ratio':':{:.3f}', 'Avg Price (R$)':'R${:.2f}', 'Avg Freight (R$)':'R${:.2f}'}),
            use_container_width=True
        )
        
        
# Tab 3 - Pola Transaksi
with tab3:
    st.subheader("Waktu Puncak Transaksi Pelanggan")
 
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_dist  = filtered_df['order_day_of_week'].value_counts().reindex(day_order).fillna(0)
    hour_dist = filtered_df['order_hour'].value_counts().sort_index()
    peak_day  = day_dist.idxmax()
    peak_hour = hour_dist.idxmax()
 
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Pesanan per Hari dalam Seminggu**")
        colors_day = [PALETTE['primary'] if d == peak_day else PALETTE['secondary'] for d in day_dist.index]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars_d = ax.bar(day_dist.index, day_dist.values, color=colors_day, edgecolor='white')
        for bar in bars_d:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+80,
                    f'{int(bar.get_height()):,}', ha='center', fontsize=8.5)
        ax.set_ylabel("Jumlah Pesanan (unit)", fontsize=10)
        ax.set_xlabel("Hari", fontsize=10)
        ax.tick_params(axis='x', rotation=25)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        ax.annotate(f'Puncak: {peak_day}',
                    xy=(list(day_dist.index).index(peak_day), day_dist.max()),
                    xytext=(0,20), textcoords='offset points',
                    fontsize=9, color=PALETTE['danger'], fontweight='bold', ha='center')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_t2:
        st.markdown("**Pesanan per Jam dalam Sehari**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hour_dist.index, hour_dist.values,marker='o', color=PALETTE['danger'],
                linewidth=2.5, markersize=4, markerfacecolor='white', markeredgewidth=1.5)
        ax.fill_between(hour_dist.index, hour_dist.values, alpha=0.15, color=PALETTE['danger'])
        ax.axvspan(10, 16, alpha=0.08, color='orange', label='Jam sibuk (10–16)')
        ax.axvline(x=peak_hour, color=PALETTE['danger'], linestyle='--', linewidth=1.5,
                   label=f'Puncak jam {peak_hour}:00')
        ax.set_xlabel("Jam (format 24 jam)", fontsize=10)
        ax.set_ylabel("Jumlah Pesanan (unit)", fontsize=10)
        ax.set_xticks(range(0, 24))
        ax.legend(fontsize=9, framealpha=0.85)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        plt.tight_layout(); st.pyplot(fig)
 
    st.info(f"Hari tersibuk: **{peak_day}**  |  Jam tersibuk: **{peak_hour}:00**  →  Window optimal untuk push notification & flash sale")
    
    st.markdown("---")
    st.subheader("Perbandingan Pola Jam: SP vs State Lainnya")
    sp_hr    = filtered_df[filtered_df['customer_state']=='SP']['order_hour'].value_counts().sort_index()
    nonsp_hr = filtered_df[filtered_df['customer_state']!='SP']['order_hour'].value_counts().sort_index()
    sp_pct    = sp_hr    / sp_hr.sum()    * 100
    nonsp_pct = nonsp_hr / nonsp_hr.sum() * 100
 
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(sp_pct.index, sp_pct.values, marker='o', label='São Paulo (SP)',
            color=PALETTE['primary'], linewidth=2.5, markersize=4,
            markerfacecolor='white', markeredgewidth=1.5)
    ax.plot(nonsp_pct.index, nonsp_pct.values, marker='s', label='State Lainnya',
            color=PALETTE['danger'], linewidth=2.5, markersize=4,
            linestyle='--', markerfacecolor='white', markeredgewidth=1.5)
    ax.fill_between(sp_pct.index, sp_pct.values, nonsp_pct.values, alpha=0.07, color=PALETTE['primary'])
    ax.set_xlabel("Jam (format 24 jam)", fontsize=10)
    ax.set_ylabel("Persentase Pesanan (%)", fontsize=10)
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=11, framealpha=0.85)
    plt.tight_layout(); st.pyplot(fig)
    st.caption(f"Jam puncak SP: {sp_pct.idxmax()}:00 ({sp_pct.max():.1f}%)  |  State Lainnya: {nonsp_pct.idxmax()}:00 ({nonsp_pct.max():.1f}%)")    


# Tab 4 - Geospatial Analysis
with tab4:
    st.subheader("Distribusi Market Pelanggan per State Brazil")
    state_market = (
        filtered_df.groupby('customer_state').agg(
            total_customers = ('customer_id', 'nunique'),
            total_revenue    = ('price', 'sum'),
            avg_review_score = ('review_score', 'mean'),
        ).reset_index().sort_values('total_customers', ascending=False)
    )
    
    state_market['market_share'] = (state_market['total_customers'] / state_market['total_customers'].sum() * 100).round(2)
    
    col_geo1, col_geo2 = st.columns(2)
    with col_geo1:
        st.markdown("**Top 10 State - Jumlah Pelanggan**")
        top10 = state_market.head(10)
        colors_s = [PALETTE['warning'] if i == 0 else '#D3D3D3' for i in range(len(top10))]
        fig, ax = plt.subplots(figsize=(7, 4))
        bh = ax.barh(top10['customer_state'][::-1], top10['total_customers'][::-1],
                     color=colors_s[::-1], edgecolor='white', height=0.6)
        for bar in bh:
            ax.text(bar.get_width()+150, bar.get_y()+bar.get_height()/2,
                    f'{int(bar.get_width()):,}', va='center', fontsize=8)
        ax.set_xlabel("Jumlah Pelanggan (orang)", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_geo2:
        st.markdown("**Market Share Pelanggan (Donut)**")
        top5 = state_market.head(5).copy()
        oth  = state_market.iloc[5:]['total_customers'].sum()
        pie_data = pd.concat([top5[['customer_state','total_customers']],
                               pd.DataFrame([{'customer_state':'Lainnya','total_customers':oth}])],
                              ignore_index=True)
        colors_pie = ['#5E69DE','#769FCD','#B9D7EA','#D6E6F2','#E8F1F5','#F0F0F0']
        explode    = [0.08 if i == 0 else 0 for i in range(len(pie_data))]
        fig, ax = plt.subplots(figsize=(7, 5))
        _, _, autotexts = ax.pie(
            pie_data['total_customers'], labels=pie_data['customer_state'],
            autopct='%1.1f%%', startangle=140, colors=colors_pie, explode=explode,
            wedgeprops={'edgecolor':'white','linewidth':1.5}
        )
        for at in autotexts: at.set_fontweight('bold'); at.set_fontsize(9)
        ax.add_artist(plt.Circle((0,0), 0.55, fc='white'))
        ax.set_title("Dominasi Market Share (%)", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
 
    st.markdown("---")
    st.subheader("Sebaran Lokasi Pelanggan (Peta Interaktif)")
    map_data = filtered_df[['geolocation_lat','geolocation_lng']].dropna().sample(n=min(30000, len(filtered_df)), random_state=42)
    map_data.columns = ['lat','lon']
    st.map(map_data, zoom=3)
    
    st.markdown("---")
    st.subheader("Analisis Lanjutan: Potensi Ekspansi (Exclude SP/RJ/MG)")
    all_periods = sorted(filtered_df['year_month'].unique())
    mid  = all_periods[len(all_periods)//2]
    fh   = filtered_df[filtered_df['year_month'] <  mid]
    sh   = filtered_df[filtered_df['year_month'] >= mid]
    o1   = fh.groupby('customer_state')['order_id'].nunique().rename('orders_first')
    o2   = sh.groupby('customer_state')['order_id'].nunique().rename('orders_second')
    grow = pd.concat([o1, o2], axis=1).fillna(0)
    grow['growth_pct'] = ((grow['orders_second']-grow['orders_first'])
                          /grow['orders_first'].replace(0,np.nan)*100).round(1)
    exp = (grow[~grow.index.isin(['SP','RJ','MG'])]
           .sort_values('growth_pct', ascending=False).head(10).reset_index())
 
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_exp = [PALETTE['success'] if g > 0 else PALETTE['danger'] for g in exp['growth_pct']]
    bh3 = ax.barh(exp['customer_state'][::-1], exp['growth_pct'][::-1],
                  color=colors_exp[::-1], edgecolor='white', height=0.65)
    ax.axvline(x=0, color='#333', linewidth=1.2)
    for bar, val in zip(bh3, exp['growth_pct'][::-1]):
        offset = 1 if val >= 0 else -1
        ha_    = 'left' if val >= 0 else 'right'
        ax.text(bar.get_width()+offset, bar.get_y()+bar.get_height()/2,
                f'{val:+.1f}%', va='center', ha=ha_, fontsize=9, fontweight='bold',
                color=PALETTE['success'] if val > 0 else PALETTE['danger'])
    ax.set_xlabel("Pertumbuhan (%)", fontsize=10)
    ax.set_ylabel("Negara Bagian (State)", fontsize=10)
    ax.set_title("Membandingkan paruh pertama vs paruh kedua periode dataset", fontsize=10, pad=10)
    from matplotlib.patches import Patch as MPatch
    ax.legend(handles=[MPatch(facecolor=PALETTE['success'], label='Pertumbuhan positif'),
                        MPatch(facecolor=PALETTE['danger'],  label='Pertumbuhan negatif')],
              fontsize=9, framealpha=0.85)
    plt.tight_layout()
    st.pyplot(fig)
 
 
# Tab 5 - RFM & Segmentasi
with tab5:
    st.subheader("RFM Analysis - Segmentasi Pelanggan")
    st.caption("Recency - Frequency - Monetary - berdasarkan data terfilter")
    
    rfm = build_rfm(filtered_df)
    seg_summary = (
        rfm.groupby('Segment').agg(
            count        = ('customer_id','count'),
            avg_recency  = ('Recency',    'mean'),
            avg_frequency= ('Frequency',  'mean'),
            avg_monetary = ('Monetary',   'mean')
        ).round(2).reset_index()
    )
    seg_summary['pct'] = (seg_summary['count'] / seg_summary['count'].sum() * 100).round(1)
    seg_plot = seg_summary.sort_values('count', ascending=False)
 
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Distribusi Segmen Pelanggan**")
        colors_list = [SEG_COLORS.get(s,'#888') for s in seg_plot['Segment']]
        fig, ax = plt.subplots(figsize=(7, 6))
        _, _, autotexts = ax.pie(
            seg_plot['count'], labels=seg_plot['Segment'],
            autopct='%1.1f%%', colors=colors_list, startangle=90, pctdistance=0.82,
            wedgeprops={'linewidth':2.5,'edgecolor':'white'}
        )
        ax.add_artist(plt.Circle((0,0), 0.55, fc='white'))
        ax.text(0, 0, f"{seg_plot['count'].sum():,}\nPelanggan",
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        for at in autotexts:
            at.set_fontsize(0.5)
            at.set_fontweight('bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_r2:
        st.markdown("**Peta Segmen: Recency vs Monetary**")
        fig, ax = plt.subplots(figsize=(7, 6))
        for _, row in seg_plot.iterrows():
            ax.scatter(row['avg_recency'], row['avg_monetary'],
                       s=row['avg_frequency']*900+150,
                       color=SEG_COLORS.get(row['Segment'],'#888'),
                       alpha=0.85, edgecolors='white', linewidth=1.5)
            ax.annotate(row['Segment'], (row['avg_recency'], row['avg_monetary']),
                        textcoords='offset points', xytext=(6,6), fontsize=8.5, fontweight='bold')
        ax.set_xlabel("Avg Recency (Hari Sejak Pembelian Terakhir)", fontsize=10)
        ax.set_ylabel("Avg Monetary — Total Belanja (R$)", fontsize=10)
        ax.set_title("Ukuran bubble = rata-rata Frequency", fontsize=10, pad=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'R${x:,.0f}'))
        plt.tight_layout() 
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Clustering: Segmentasi Perilaku Belanja")
    rfm['recency_bin']   = pd.cut(rfm['Recency'], bins=[0,30,90,180,np.inf],
                                   labels=['Fresh (0–30 hr)','Recent (31–90 hr)','Moderate (91–180 hr)','Lama (>180 hr)'])
    rfm['monetary_bin']  = pd.qcut(rfm['Monetary'], q=4,
                                    labels=['Low Spender','Mid Spender','High Spender','Premium Spender'])
    rfm['frequency_bin'] = rfm['Frequency'].apply(
        lambda x: 'Single Purchase' if x == 1 else ('Returning (2–3x)' if x <= 3 else 'Loyal (4x+)'))
 
    cluster_summary = (rfm.groupby(['recency_bin','monetary_bin'], observed=True)
                       .agg(pelanggan=('customer_id','count'), avg_freq=('Frequency','mean'), avg_spend=('Monetary','mean'))
                       .round(2).reset_index())
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Heatmap Recency × Spending Level**")
        hm_data = cluster_summary.pivot(index='recency_bin', columns='monetary_bin', values='pelanggan').fillna(0)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(hm_data, annot=True, fmt='.0f', cmap='YlGn', ax=ax, linewidths=0.8,
                    cbar_kws={'label':'Jumlah Pelanggan'}, annot_kws={'fontsize':10,'fontweight':'bold'})
        ax.set_xlabel("Spending Level", fontsize=10)
        ax.set_ylabel("Recency", fontsize=10)
        ax.tick_params(axis='x', rotation=25, labelsize=9)
        ax.tick_params(axis='y', rotation=0,  labelsize=9)
        plt.tight_layout(); st.pyplot(fig)
 
    with col_c2:
        st.markdown("**Frekuensi Pembelian vs Rata-rata Spending**")
        freq_order   = ['Single Purchase','Returning (2–3x)','Loyal (4x+)']
        freq_summary = (rfm.groupby('frequency_bin')
                        .agg(count=('customer_id','count'), avg_monetary=('Monetary','mean'))
                        .reset_index())
        freq_summary['frequency_bin'] = pd.Categorical(freq_summary['frequency_bin'],
                                                        categories=freq_order, ordered=True)
        freq_summary = freq_summary.sort_values('frequency_bin')
 
        fig, ax = plt.subplots(figsize=(7, 5))
        bar_col = ['#B8D9A4','#6DB35A','#2E7D32']
        bars2   = ax.bar(freq_summary['frequency_bin'].astype(str), freq_summary['count'],
                         color=bar_col, edgecolor='white', width=0.5, zorder=2)
        for bar in bars2:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                    f'{int(bar.get_height()):,}', ha='center', fontsize=10, fontweight='bold', color=PALETTE['success'])
        ax_t = ax.twinx()
        ax_t.plot(freq_summary['frequency_bin'].astype(str), freq_summary['avg_monetary'],
                  'D--', color=PALETTE['danger'], linewidth=2.5, markersize=11,
                  markerfacecolor='white', markeredgewidth=2, label='Avg Spend (R$)', zorder=3)
        for i, (_, row) in enumerate(freq_summary.iterrows()):
            ax_t.annotate(f"R${row['avg_monetary']:,.0f}", xy=(i, row['avg_monetary']),
                          xytext=(0,12), textcoords='offset points',
                          fontsize=9, color=PALETTE['danger'], ha='center', fontweight='bold')
        ax_t.set_ylabel("Rata-rata Total Spend (R$)", fontsize=10, color=PALETTE['danger'])
        ax_t.tick_params(axis='y', labelcolor=PALETTE['danger'])
        ax_t.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'R${x:,.0f}'))
        ax.set_xlabel("Segmen Frekuensi Pembelian", fontsize=10)
        ax.set_ylabel("Jumlah Pelanggan (orang)", fontsize=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        ax_t.legend(fontsize=9, loc='upper left', framealpha=0.85)
        plt.tight_layout(); st.pyplot(fig)
 
    st.markdown("---")
    st.subheader("Ringkasan Segmen RFM")
    st.dataframe(
        seg_summary.sort_values('count', ascending=False)
        .rename(columns={'Segment':'Segmen','count':'Jumlah Pelanggan',
                         'avg_recency':'Avg Recency (Hari)','avg_frequency':'Avg Frequency',
                         'avg_monetary':'Avg Monetary (R$)','pct':'%'})
        .style.format({'Avg Recency (Hari)':'{:.0f}','Avg Monetary (R$)':'R${:,.2f}','%':'{:.1f}%'}),
        use_container_width=True
    )


# Tab 6 - Rekomendasi Strategis
with tab6:
    st.subheader("Rekomendasi Strategis")
    st.caption("Berdasarkan hasil analisis data E-Commerce 2016-2018")
    st.markdown("---")
    
    rekomendasi = [{
            "no"    :"01",
            "judul" : "Subsidi Ongkos Kirim untuk Kategori Berisiko Tinggi",
            "dasar" : "Freight-to-price ratio > 0.3 terbukti menjadi hambatan pembelian.",
            "aksi"  : "Berikan gratis ongkir atau subsidi parsial untuk kategori dengan freight ratio > 0.3, terutama di state yang pertumbuhannya tinggi (SC, BA, GO).",
            "kpi"  : "Target: turunkan freight ratio < 0.25 untuk top-3 kategori berisiko",
        },
        {
            "no"    :"02",
            "judul": "Flash Sale di Hari Senin Pukul 14.00–16.00",
            "dasar": "Analisis pola waktu menunjukkan Senin adalah hari tersibuk dan jam 14:00–16:00 adalah window puncak transaksi secara konsisten.",
            "aksi" : "Jadwalkan promo kilat, push notification, dan email blast tepat di window waktu ini untuk memaksimalkan konversi.",
            "kpi"  : "Target: tingkatkan konversi di window tersebut sebesar 15–20%",
        },
        {
            "no"    :"03",
            "judul" :"Program Loyalitas untuk Potential Loyalists",
            "dasar": "RFM analysis menunjukkan segmen Potential Loyalists adalah kelompok terbesar yang belum dimaksimalkan — mereka sudah aktif tapi belum konsisten.",
            "aksi" : "Buat program poin reward, diskon khusus pelanggan berulang, dan notifikasi personal untuk mendorong mereka naik ke segmen Champions.",
            "kpi"  : "Target: konversi 20% Potential Loyalists menjadi Loyal Customers dalam 6 bulan",
        },
        {
            "no"    :"04",
            "judul" : "Kampanye Win-Back untuk At Risk & Churned",
            "dasar": "Segmen At Risk dan Churned memiliki Recency tinggi — mereka pernah aktif tapi sudah lama tidak bertransaksi.",
            "aksi" : "Kirim email reaktivasi dengan diskon eksklusif, reminder produk yang pernah mereka beli, dan batas waktu penawaran yang jelas.",
            "kpi"  : "Target: reaktivasi 10% pelanggan Churned dalam 3 bulan",
        },
        {
            "no"    :"05",
            "judul" : "Ekspansi Pemasaran ke SC dan BA",
            "dasar": "SC dan BA adalah satu-satunya state yang muncul di KEDUA analisis: sudah punya basis pelanggan (3.247–3.546) DAN masih tumbuh >210%.",
            "aksi" : "Alokasikan anggaran iklan digital secara spesifik ke SC dan BA. Pertimbangkan partnership dengan seller lokal di kedua state ini.",
            "kpi"  : "Target: tingkatkan pelanggan SC dan BA sebesar 30% dalam 1 tahun",
        },
        {
            "no"    :"06",
            "judul" :"Optimalkan bed_bath_table & health_beauty",
            "dasar": "Keduanya mendominasi penjualan sekaligus memiliki rating > 4.0. Ini kombinasi ideal untuk anchor category.",
            "aksi" : "Jadikan featured category di halaman utama, bundling lintas kategori, dan tingkatkan stok menjelang November (puncak penjualan).",
            "kpi"  : "Target: pertahankan posisi #1 dan #2 dengan margin rating > 4.0",
        },
    ]
    
    for rec in rekomendasi:
        with st.expander(f"Rekomendasi {rec['no']}: {rec['judul']}", expanded=False):
            c1, c2, c3 = st.columns([2,2,1])
            with c1:
                st.markdown("**Dasar Analisis**")
                st.info(rec['dasar'])
            with c2:
                st.markdown("**Aksi yang Disarankan**")
                st.success(rec['aksi'])
            with c3:
                st.markdown("**Target KPI**")
                st.warning(rec['kpi'])
 
    st.markdown("---")
    st.subheader("Visualisasi Pendukung Rekomendasi")
 
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("**Review Score Distribution**")
        rv = filtered_df['review_score'].dropna().apply(lambda x: int(round(x))).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_rv = [PALETTE['danger'] if s <= 2 else (PALETTE['neutral'] if s == 3 else PALETTE['success']) for s in rv.index]
        bars_rv   = ax.bar(rv.index.astype(str), rv.values, color=colors_rv, edgecolor='white', width=0.6)
        for bar in bars_rv:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                    f'{int(bar.get_height()):,}', ha='center', fontsize=9)
        ax.set_xlabel("Review Score", fontsize=10)
        ax.set_ylabel("Jumlah Ulasan (unit)", fontsize=10)
        ax.set_title(f"Avg: {filtered_df['review_score'].mean():.2f} — Mayoritas pelanggan puas (score 4–5)", fontsize=10, pad=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        plt.tight_layout(); st.pyplot(fig)
 
    with col_v2:
        st.markdown("**Delivery Days Distribution**")
        dlv = filtered_df['delivery_days'].dropna()
        dlv = dlv[dlv <= 60]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(dlv, bins=30, color=PALETTE['secondary'], edgecolor='white', linewidth=0.5)
        ax.axvline(dlv.mean(), color=PALETTE['danger'], linestyle='--', linewidth=2,
                   label=f'Rata-rata: {dlv.mean():.1f} hari')
        ax.axvline(dlv.median(), color=PALETTE['success'], linestyle='--', linewidth=2,
                   label=f'Median: {dlv.median():.1f} hari')
        ax.set_xlabel("Hari Pengiriman", fontsize=10)
        ax.set_ylabel("Jumlah Pesanan (unit)", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        plt.tight_layout(); st.pyplot(fig)
 
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:gray;font-size:13px;'>"
        "📊 E-Commerce Brazil Dashboard · Charlene Manuella Angkadjaja · Dicoding 2024"
        "</div>",
        unsafe_allow_html=True
    )