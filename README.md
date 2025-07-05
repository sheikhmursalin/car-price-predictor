# 🚗 Car Price Predictor

A machine learning web app built using **Python, Flask, and scikit-learn** that predicts the selling price of a used car based on its company, fuel type, manufacturing year, and kilometers driven.

---

### 🔮 Live Demo

> [👉 Click here to try it](https://car-price-predictor-rwv3.onrender.com) 

---

### 📸 Screenshot

---

## 📂 Project Structure

```
car-price-predictor/
├── app.py
├── train_model.py
├── requirements.txt
├── Procfile
├── .gitignore
├── data/
│   └── car_data.csv
├── models/
│   ├── car_price_model.pkl
│   └── meta.json
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   ├── css/styles.css
│   └── js/main.js
└── notebooks/
    └── Car Price Predictor.ipynb
```

---

## 📊 Features

- Predict selling price of used cars
- Takes user inputs:
  - Car company
  - Fuel type
  - Year of manufacture
  - KMs driven
- Trained with **Random Forest Regressor**
- Clean and responsive UI using **Bootstrap 5**
- Easy to deploy (Render, Heroku)

---

## ⚙️ Tech Stack

- Python 🦕
- Flask 🍥
- scikit-learn
- pandas & numpy
- HTML5, CSS3, Bootstrap
- Render or Heroku (deployment)

---

## 📦 Installation (Local)

```bash
# Clone the repository
git clone https://github.com/your-username/car-price-predictor.git
cd car-price-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # On Windows
# source venv/bin/activate     # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Train the model (only if not already saved)
python train_model.py

# Run the app
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🚀 Deployment (Render)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com)
3. Create → Web Service → Connect your repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app`
5. Add file: `Procfile` → `web: gunicorn app:app`

---

## 📊 Model Training

The model is trained using the `car_data.csv` dataset:

- Cleaned and preprocessed
- Feature engineered: `car_age = current_year - year`
- OneHotEncoded categorical features
- Trained on RandomForestRegressor
- Stored as:
  - `models/car_price_model.pkl` — trained model
  - `models/meta.json` — dropdown values

Use the Jupyter notebook for EDA & training:\
`notebooks/Car Price Predictor.ipynb`

---

## 🧠 Example Prediction

```python
sample = pd.DataFrame([{
  "company": "Maruti",
  "fuel_type": "Petrol",
  "kms_driven": 40000,
  "car_age": 6
}])

price = model.predict(sample)[0]
print(f"Predicted Price: ₹{round(price):,}")
```

---

## ✍️ Author

**Sheikh Mursaleen**\
Connect with me on [LinkedIn](https://linkedin.com/in/your-profile)\
Follow AI projects on [GitHub](https://github.com/your-username)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

### 🧠 Tip

You can convert the notebook into Python script for training with:

```bash
jupyter nbconvert --to script "Car Price Predictor.ipynb" --output train_model
```

