# House Maxify — Prototype

A simple Flask web app to get property price estimates using an XGBoost model.

## Project Structure

- `app.py` — Flask server and routes
- `html/` — Templates (`index.html`, `form.html`, `result.html`)
- `css/` — Styling (`styles.css`)
- `js/` — Small client enhancements (`app.js`)
- `assets/` — Branding assets (logo)
- `model/house_price_xgb_advanced.joblib` — Trained model (place here)
- `data/kc_house_data_clean.csv` — Dataset (used for medians/examples)
- `data/leads.csv` — Created automatically to store submissions

## Setup

1. Create and activate a virtual environment (optional but recommended):

   - Windows: `python -m venv .venv && .venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure files exist:
   - Model: `model/house_price_xgb_advanced.joblib`
   - Dataset: `data/kc_house_data_clean.csv`

## Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### Run with Docker

1. Build the image:

   ```bash
   docker build -t house-maxify .
   ```

2. Run the container (bind port 5000 and mount local data/model if you want live updates):

   ```bash
   docker run --rm -p 5000:5000 \
     -v %cd%/data:/app/data \
     -v %cd%/model:/app/model \
     house-maxify
   ```

   On macOS/Linux, replace `%cd%` with `$(pwd)`.

## Notes

- The app infers the model’s feature list when possible and fills missing inputs with dataset medians.
- Submissions are appended to `data/leads.csv` with timestamp, contact, intent, and predicted range.
