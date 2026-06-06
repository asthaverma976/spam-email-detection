"""
Flask Web Application — Spam Email Detector
Routes:
    GET  /          → email input form
    POST /classify  → classify email and show result
"""

import os
import sys

from flask import Flask, render_template, request

# ─── Project path setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model.predict import predict_email

# ─── Optional MongoDB ────────────────────────────────────────────────────────
try:
    from database.mongo_handler import MongoHandler
    mongo = MongoHandler()
    mongo_available = mongo.connect()
except Exception:
    mongo = None
    mongo_available = False

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Render the email input form."""
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    """Classify the submitted email and render the result."""
    email_text = request.form.get("email_text", "").strip()

    if not email_text:
        return render_template("index.html")

    # Run prediction
    result = predict_email(email_text)

    # Optionally log to MongoDB
    if mongo_available and mongo is not None:
        try:
            mongo.insert_classification(
                email_text=email_text,
                prediction=result["prediction"],
                confidence=result["confidence"],
                preprocessed_text=result["preprocessed_text"],
            )
        except Exception as e:
            print(f"[MongoDB] Logging failed: {e}")

    return render_template(
        "result.html",
        prediction=result["prediction"],
        confidence=result["confidence"],
        email_text=email_text,
        preprocessed_text=result["preprocessed_text"],
    )


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
