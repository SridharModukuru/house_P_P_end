import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import joblib

file = 'price_model.sav'
# Load your trained model
loaded_model = joblib.load(file)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('form.html')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        # Extracting features from the form
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        condition = int(request.form['condition'])
        yr_built = int(request.form['yr_built'])
        city = int(request.form['city'])
        sqft = float(request.form['sqft'])

        # Assuming you have loaded the model successfully, you can predict the price
        data = {
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'condition': [condition],
            'yr_built': [yr_built],
            'city': [city],
            'sqft': [sqft]
        }
        data = pd.DataFrame(data)

        # Make predictions directly without scaling
        pred = loaded_model.predict(data)
        price = pred[0]

        # Format price with commas
        formatted_price = '{:,.2f}'.format(price)

        return render_template('result.html', price=formatted_price)

if __name__ == '__main__':
    app.run(port=8090)
