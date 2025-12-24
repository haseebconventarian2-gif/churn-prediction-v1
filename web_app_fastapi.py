from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import os
import pickle
import traceback

app = FastAPI()

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
FEATURE_STORE = os.path.join(ARTIFACTS_DIR, "feature_store.csv")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessing.pkl")
MODEL_CANDIDATES = [
    os.path.join(ARTIFACTS_DIR, "best_model_tuned.pkl"),
    os.path.join(ARTIFACTS_DIR, "best_model.pkl")
]


@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
    <html>
      <head>
        <title>Churn Prediction (by customerID)</title>
        <style> body { font-family: Arial, sans-serif; margin:20px } .container{max-width:600px;margin:auto}</style>
      </head>
      <body>
        <div class="container">
          <h1>Churn Prediction</h1>
          <p>Enter a <strong>customerID</strong> to look up the record and get a churn prediction.</p>
          <form action="/predict" method="post">
            <label for="customerID">customerID</label>
            <input type="text" id="customerID" name="customerID" style="width:100%; padding:8px" required>
            <br/><br/>
            <button type="submit">Predict</button>
          </form>
          <hr/>
          <p>If prediction fails due to an environment/pickle mismatch, please regenerate the preprocessor by running the transformation step inside the same environment where you will run this app.</p>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


def load_preprocessor(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f), p
            except Exception:
                # try next candidate
                continue
    return None, None


@app.post("/predict")
async def predict(request: Request):
    form = await request.form()
    customer_id = form.get("customerID")
    if not customer_id:
        return HTMLResponse("<h3>Please provide customerID</h3>")

    # Load feature store and find the row
    if not os.path.exists(FEATURE_STORE):
        return HTMLResponse(f"<h3>Feature store not found at {FEATURE_STORE}</h3>")

    try:
        df = pd.read_csv(FEATURE_STORE)
    except Exception as e:
        return HTMLResponse(f"<h3>Unable to read feature store: {e}</h3>")

    row = df.loc[df['customerID'] == customer_id]
    if row.empty:
        return HTMLResponse(f"<h3>No record found for customerID: {customer_id}</h3>")

    # Prepare single-row DataFrame for preprocessing
    input_df = row.drop(columns=['customerID'], errors='ignore')

    # Load preprocessor
    if not os.path.exists(PREPROCESSOR_PATH):
        return HTMLResponse("<h3>Preprocessor not found. Please run data transformation to create `artifacts/preprocessing.pkl`.</h3>")

    try:
        preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    except Exception as e:
        tb = traceback.format_exc()
        msg = (
            f"<h3>Failed to load preprocessor (pickle unpickle error).</h3>"
            f"<pre>{str(e)}</pre>"
            f"<p>Traceback:</p><pre>{tb}</pre>"
            f"<p>Likely cause: binary/library version mismatch (numpy/scikit-learn). Recommended: regenerate `artifacts/preprocessing.pkl` inside the active environment by running the data transformation step.</p>"
        )
        return HTMLResponse(msg)

    # Apply preprocessing (preprocessor may expect a full set of columns)
    try:
        X = preprocessor.transform(input_df)
    except Exception as e:
        tb = traceback.format_exc()
        return HTMLResponse(f"<h3>Preprocessing failed:</h3><pre>{tb}</pre>")

    # Load model
    model, model_path = load_model()
    if model is None:
        return HTMLResponse("<h3>No model found in artifacts. Place `best_model_tuned.pkl` or `best_model.pkl` into the artifacts folder.</h3>")

    try:
        # predict_proba may not be available for all models
        proba = None
        try:
            proba = model.predict_proba(X)
        except Exception:
            pass

        pred = model.predict(X)
        pred_label = pred[0]
        prob_str = ""
        if proba is not None:
            # assume binary classification, take probability of positive class
            if proba.shape[1] == 2:
                prob = proba[0, 1]
                prob_str = f" (probability {prob:.3f})"
            else:
                prob_str = f" (probabilities: {proba[0].tolist()})"

        html = f"""
        <html>
        <head><title>Prediction Result</title></head>
        <body>
          <div style='font-family:Arial, sans-serif; margin:20px'>
            <h2>Prediction for customerID: {customer_id}</h2>
            <p><strong>Model:</strong> {os.path.basename(model_path)}</p>
            <p><strong>Predicted label:</strong> {pred_label}{prob_str}</p>
            <h3>Input record</h3>
            {input_df.to_html(index=False)}
            <p><a href="/">Back</a></p>
          </div>
        </body>
        </html>
        """
        return HTMLResponse(html)
    except Exception:
        tb = traceback.format_exc()
        return HTMLResponse(f"<h3>Prediction failed:</h3><pre>{tb}</pre>")


if __name__ == "__main__":
    import uvicorn
    import os

    # Allow changing socket via environment variables HOST and PORT
    # Example (PowerShell):
    # $env:PORT=8001; $env:HOST='127.0.0.1'; python web_app_fastapi.py
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "8000"))
    except ValueError:
        port = 8000

    uvicorn.run(app, host=host, port=port)
