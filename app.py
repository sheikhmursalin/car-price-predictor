from flask import Flask, render_template, request
import pandas as pd, joblib, json, os

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(ROOT, "models", "car_price_model.pkl"))
meta  = json.load(open(os.path.join(ROOT, "models", "meta.json")))

@app.route("/")
def index():
    return render_template(
        "index.html",
        companies=meta["companies"],
        fuel_types=meta["fuel_types"],
        current_year=meta["trained_year"]
    )

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    try:
        year       = int(form["year"])
        kms        = int(form["kms_driven"])
        company    = form["company"]
        fuel       = form["fuel_type"]
        car_age    = meta["trained_year"] - year

        inp = pd.DataFrame([{
            "company": company,
            "fuel_type": fuel,
            "kms_driven": kms,
            "car_age": car_age
        }])

        price = model.predict(inp)[0]
        return render_template("result.html",
                               price=f"₹ {price:,.0f}",
                               company=company, year=year,
                               kms=kms, fuel=fuel)
    except Exception as e:
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
