from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model ramping dan scaler yang baru dibuat di Colab
model = joblib.load('model_web_siswa.pkl')
scaler = joblib.load('scaler_web.pkl')

@app.route('/')
def index():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Ambil data dari form HTML (G1, G2, absences)
    fitur_input = [float(x) for x in request.form.values()]
    
    # 2. Ubah ke format array dan lakukan scaling (wajib)
    final_features = np.array([fitur_input])
    scaled_features = scaler.transform(final_features)
    
    # 3. Prediksi menggunakan model SVM
    prediction = model.predict(scaled_features)
    
    # 4. LOGIKA LABEL (Sesuai output Google Colab)
    # 0 = Gagal (Berisiko Gagal)
    # 1 atau 2 = Cukup/Pintar (Aman)
    if prediction[0] == 0:
        hasil = "BERISIKO GAGAL"
    else:
        hasil = "AMAN"
    
    return render_template('index.html', prediction_text=f'Hasil Prediksi: Siswa dinyatakan {hasil}')

if __name__ == "__main__":
    app.run(debug=True)