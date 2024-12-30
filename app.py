import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io

# Load Data
def load_data():
    data = pd.read_csv('water_potability1.csv')
    return data

# Preprocess Data
def preprocess_data(data):
    # Mengisi nilai yang hilang
    data["ph"].fillna(value=data["ph"].mean(), inplace=True)
    data["Sulfate"].fillna(value=data["Sulfate"].mean(), inplace=True)
    data["Trihalomethanes"].fillna(value=data["Trihalomethanes"].mean(), inplace=True)

    # Memisahkan fitur dan label
    X = data.drop("Potability", axis=1).values
    y = data["Potability"].values

    # Normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Membagi data menjadi train dan test (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=3)

    return X_train, X_test, y_train, y_test, scaler, data

# Train Model
def train_model(X_train, y_train, K=9):
    model = KNeighborsClassifier(K)
    model.fit(X_train, y_train)
    return model

# Save Model
def save_model(model, scaler, filename='model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump({'model': model, 'scaler': scaler}, file)

# Load Model
def load_model(filename='model.pkl'):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data['model'], data['scaler']

# Sidebar Navigation
st.sidebar.title("Water Potability Classification")
st.sidebar.write("Nama: Muhammad Hablul Warid Ghazali")
st.sidebar.write("NIM: 220411100019")
options = st.sidebar.radio("Pilih Halaman:", ['Analisis Data', 'Preprocessing', 'Modeling', 'Klasifikasi'])

# Main Pages
if options == 'Analisis Data':
    st.title("Klasifikasi Kualitas Air Minum Menggunakan Metode KNN")
    data = load_data()
    st.write("### Data Awal:")
    st.dataframe(data.head())

    st.write("### Informasi Dataset:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("### Statistik Deskriptif:")
    st.dataframe(data.describe())

elif options == 'Preprocessing':
    st.title("Preprocessing Data")
    data = load_data()
    st.write("### Sebelum Preprocessing")
    st.dataframe(data.isnull().sum())

    _, _, _, _, _, data_filled = preprocess_data(data)

    st.write("### Setelah Pengisian Missing Value")
    st.dataframe(data_filled.head())

    X_train, X_test, y_train, y_test, scaler, _ = preprocess_data(data)

    st.write("### Setelah Normalisasi")
    normalized_df = pd.DataFrame(X_train, columns=data.columns[:-1])
    st.dataframe(normalized_df.head())
    st.write(f"X_train Shape: {X_train.shape}")
    st.write(f"X_test Shape: {X_test.shape}")

elif options == 'Modeling':
    st.title("Modeling")
    data = load_data()
    X_train, X_test, y_train, y_test, scaler, _ = preprocess_data(data)

    st.write("### Split Data: 80% Train, 20% Test")
    st.write(f"Jumlah Data Train: {X_train.shape[0]}")
    st.write(f"Jumlah Data Test: {X_test.shape[0]}")

    # Pilihan nilai K
    st.write("### Pilih Nilai K untuk KNN:")
    k_value = st.selectbox("Nilai K:", options=[3, 5, 7, 9], index=3)

    if st.button("Train Model"):
        model = train_model(X_train, y_train, K=k_value)

        y_pred = model.predict(X_test)

        st.write("### Evaluasi Model:")
        st.write(f"Nilai K yang digunakan: {k_value}")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        save_model(model, scaler)
        st.success("Model berhasil disimpan sebagai 'model.pkl'!")

elif options == 'Klasifikasi':
    st.title("Klasifikasi")
    model, scaler = load_model()

    st.write("### Input Data untuk Prediksi")
    ph = st.number_input("pH:", value=7.0, format="%.2f")
    hardness = st.number_input("Hardness:", value=200.0, format="%.2f")
    solids = st.number_input("Solids:", value=10000.0, format="%.2f")
    chloramines = st.number_input("Chloramines:", value=5.0, format="%.2f")
    sulfate = st.number_input("Sulfate:", value=300.0, format="%.2f")
    conductivity = st.number_input("Conductivity:", value=400.0, format="%.2f")
    organic_carbon = st.number_input("Organic Carbon:", value=10.0, format="%.2f")
    trihalomethanes = st.number_input("Trihalomethanes:", value=50.0, format="%.2f")
    turbidity = st.number_input("Turbidity:", value=4.0, format="%.2f")

    if st.button("Klasifikasikan"):
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.success("Air Layak Diminum")
        else:
            st.error("Air Tidak Layak Diminum")
