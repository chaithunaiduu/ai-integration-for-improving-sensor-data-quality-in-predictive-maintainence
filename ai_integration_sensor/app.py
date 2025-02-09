from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Define the sequence generation function
def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros = pd.DataFrame(np.zeros((seq_length-1, id_df.shape[1])), columns=id_df.columns)
    id_df = df_zeros.append(id_df, ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = [data_array[start:stop, :] for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements))]
    return np.array(lstm_array)

# Define the preprocessing pipeline
def preprocess_data(df):
    features_col_name = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                         's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    df[features_col_name] = imputer.fit_transform(df[features_col_name])
    
    # Feature Scaling
    scaler = MinMaxScaler()
    df[features_col_name] = scaler.fit_transform(df[features_col_name])
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Preprocess the data
        df = preprocess_data(df)
        
        # Generate sequences
        seq_length = 50
        features_col_name = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                             's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        X = gen_sequence(df, seq_length, features_col_name)
        
        # Make predictions
        y_prob = model.predict(X).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate failure probabilities
        failure_probs = y_prob[y_prob > 0.5]
        
        # Prepare the response
        response = {
            'predictions': y_pred.tolist(),
            'failure_probabilities': failure_probs.tolist(),
            'all_probabilities': y_prob.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)