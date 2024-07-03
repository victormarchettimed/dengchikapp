from flask import Flask, request, jsonify
import pickle
import numpy as np

# Carregar o modelo
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Inicializar o aplicativo Flask
app = Flask(__name__)

# Definir o endpoint para predições
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = [
        float(data.get('ARTHRALGIA', 0)),
        float(data.get('RETRO_ORBITAL_PAIN', 0)),
        float(data.get('NAUSEA', 0)),
        float(data.get('EXANTHEM', 0)),
        float(data.get('ARTHRITIS', 0)),
        float(data.get('BACK_PAIN', 0)),
        float(data.get('MYALGIA', 0)),
        float(data.get('VOMITING', 0)),
        float(data.get('HEADACHE', 0)),
        float(data.get('FEVER', 0)),
        float(data.get('CONJUNCTIVITIS', 0)),
        float(data.get('PETECHIA', 0))
    ]
    
    # Convertendo os dados para um numpy array
    symptoms_array = np.array(symptoms).reshape(1, -1)
    
    # Fazer a predição
    prediction = model.predict_proba(symptoms_array)
    
    # Retornar a predição como JSON
    return jsonify({
        'dengue_probability': prediction[0][1],  # Probabilidade de Dengue (classe 1)
        'chikungunya_probability': prediction[0][0]  # Probabilidade de Chikungunya (classe 0)
    })

if __name__ == '__main__':
    app.run(debug=True)
