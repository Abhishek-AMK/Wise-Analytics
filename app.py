from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
kmeans_model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Sample data to analyze clusters
sample_data = pd.DataFrame({
    'Recency': [10, 300, 5, 1, 150, 20, 400],
    'Frequency': [20, 2, 50, 1, 5, 30, 1],
    'MonetaryValue': [500, 50, 2000, 100, 1500, 300, 20]
})

def analyze_clusters():
    sample_data_log = np.log(sample_data[['Recency', 'Frequency', 'MonetaryValue']]).round(3)
    sample_data_normalized = scaler.transform(sample_data_log)
    cluster_labels = kmeans_model.predict(sample_data_normalized)
    sample_data['Cluster'] = cluster_labels
    cluster_summary = sample_data.groupby('Cluster').mean().round(2)
    print(cluster_summary)

@app.route('/')
def index():
    analyze_clusters()  # Analyze clusters on the home page load
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary_value = float(request.form['monetary_value'])

        # Ensure the input values are positive and non-zero
        if recency <= 0 or frequency <= 0 or monetary_value <= 0:
            return render_template('index.html', error="All input values must be positive and non-zero.")

        # Create a DataFrame for the input
        input_data = np.array([[recency, frequency, monetary_value]])
        print(f"Input Data: {input_data}")

        input_data_log = np.log(input_data).round(3)
        print(f"Log Transformed Data: {input_data_log}")

        input_data_normalized = scaler.transform(input_data_log)
        print(f"Normalized Data: {input_data_normalized}")
        
        # Predict the cluster
        cluster_label = kmeans_model.predict(input_data_normalized)[0]
        print(f"Predicted Cluster: {cluster_label}")

        # Define the segments based on cluster analysis
        segments = {
            0: "Frequent",
            1: "VIP",
            2: "Lost Customers"
        }

        segment = segments.get(cluster_label, "Unknown")

        return render_template('results.html', segment=segment)
    except ValueError:
        return render_template('index.html', error="Invalid input. Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)