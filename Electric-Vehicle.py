import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Judul aplikasi
st.title('Website Analisis Dataset')

# Deskripsi
st.write("""
## Selamat Datang Di Dashboard Interaktif!
Aplikasi ini memungkinkan Anda untuk memasukkan data dan melihat visualisasi secara real-time.
Silakan gunakan kontrol di sisi kiri untuk memasukkan Jumlah Nilai K Dan Upload data bertipe CSV.
""")

# Footer
st.write("""
---
**Dibuat oleh [Muhammad Ibrahim Sulaiman Dawud Abdullah]** 
"""
"""
**Jurusan = Teknik Informatika 21**
"""
"""
**NIM = 21103041025**
""")

# Upload dataset
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# Tampilkan data
    st.header('Dataframe')
    st.write(df)

    # Tampilkan statistik dasar
    st.header('Statistik Dasar')
    st.write(df.describe())

    # Analisis data
    st.header('Analisis Data')

    # Pilih kolom untuk visualisasi
    numeric_columns = df.select_dtypes(['float64', 'int64']).columns
    selected_column = st.selectbox('Pilih kolom untuk histogram', numeric_columns)
    
    if selected_column:
        st.subheader(f'Histogram dari {selected_column}')
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], ax=ax, kde=True)
        st.pyplot(fig)

    # Scatter plot
    st.header('Scatter Plot')
    x_axis = st.selectbox('Pilih kolom untuk sumbu X', numeric_columns)
    y_axis = st.selectbox('Pilih kolom untuk sumbu Y', numeric_columns)
    
    if x_axis and y_axis:
        st.subheader(f'Scatter plot dari {x_axis} vs {y_axis}')
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
        st.pyplot(fig)

    # Heatmap korelasi
    st.header('Heatmap Korelasi')
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax)
    st.pyplot(fig)

# Dataset Kendaraan Listrik Original
df = pd.read_csv('ElectricVehicle_PopulationFiltering.csv')
st.header("Original Dataset")
st.write(df)

# Dataset Preprocessing
df = pd.read_csv('data_filteringfinish4.csv')
st.header("Preprocessing Dataset")
st.write(df)

# Dataset One Hot Encoding
df = pd.read_csv('data_filteringfinish4.csv')
st.header("Data Encoded Boolean")
st.write(df)

# Proses Clustering
st.header("Hasil Grafik Menentukan Jumlah Cluster dengan Elbow dan Hasil Cluster")
st.write(df)
df_cols = df = pd.read_csv('data_filteringfinish4.csv')
clusters=[]
for i in range(1,11):
    km =KMeans(n_clusters=i).fit(df_cols)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')

n_clust = 3
kmean = KMeans(n_clusters=n_clust).fit(df_cols)
df_cols['Labels'] = kmean.labels_

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

# Dataset Kendaraan Listrik Clustering
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Model Year', y='Electric Range', hue='Labels', size='Labels', palette=sns.color_palette('hls', n_clust), data=df_cols)

for label in df_cols['Labels'].unique():
    mean_model_year = df_cols[df_cols['Labels'] == label]['Model Year'].mean()
    mean_electric_range = df_cols[df_cols['Labels'] == label]['Electric Range'].mean()
    plt.annotate(label,
                 (mean_model_year, mean_electric_range),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='black')

plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()


# Define df_cols as the columns of the DataFrame
df_cols = df = pd.read_csv('data_filteringfinish4.csv')
df_cols = df.columns.tolist()
st.title("DataFrame Column Selector")
st.write("Select the columns you want to display:")
selected_cols = st.multiselect("Columns", df_cols, default=df_cols)
filtered_df = df[selected_cols]
st.write("Filtered DataFrame:")
st.dataframe(filtered_df)

# Set Option Sidebar
st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2,10,3,1)
