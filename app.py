# app.py
"""
Single-file Flask backend for:
- Upload CSVs (saved to uploads/)
- Train Message model (text,label)
- Train URL model (url,label)
- Predict on Message / URL
- Save artifacts (pickle) and reload on startup

Usage:
    python app.py
Open browser: http://127.0.0.1:5000
"""
import os
import pickle
from io import StringIO

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------- Configuration ----------
APP_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(APP_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MSG_MODEL_FILE = os.path.join(APP_DIR, "msg_model.pkl")
MSG_VEC_FILE = os.path.join(APP_DIR, "msg_vectorizer.pkl")
URL_MODEL_FILE = os.path.join(APP_DIR, "url_model.pkl")
URL_VEC_FILE = os.path.join(APP_DIR, "url_vectorizer.pkl")

ALLOWED_EXT = {"csv"}

# ---------- Flask init ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
app.secret_key = "dev-secret-change-for-prod"

# ---------- In-memory artifacts ----------
msg_model = None
msg_vectorizer = None
url_model = None
url_vectorizer = None


# ---------- Utilities ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def safe_read_csv_file_storage(file_storage):
    """
    Read uploaded file (werkzeug FileStorage) robustly into pandas.DataFrame.
    """
    raw = file_storage.read()
    # reset pointer for potential re-use
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(text))


def load_artifacts():
    """Try to load pickled models/vectorizers from disk into memory."""
    global msg_model, msg_vectorizer, url_model, url_vectorizer
    try:
        if os.path.exists(MSG_MODEL_FILE) and os.path.exists(MSG_VEC_FILE):
            with open(MSG_MODEL_FILE, "rb") as f:
                msg_model = pickle.load(f)
            with open(MSG_VEC_FILE, "rb") as f:
                msg_vectorizer = pickle.load(f)
    except Exception:
        msg_model, msg_vectorizer = None, None

    try:
        if os.path.exists(URL_MODEL_FILE) and os.path.exists(URL_VEC_FILE):
            with open(URL_MODEL_FILE, "rb") as f:
                url_model = pickle.load(f)
            with open(URL_VEC_FILE, "rb") as f:
                url_vectorizer = pickle.load(f)
    except Exception:
        url_model, url_vectorizer = None, None


def train_message_model_from_df(df):
    """
    Expects df with columns: 'text' and 'label'
    Returns (clf, vectorizer, nrows, labels_list)
    """
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Message CSV must contain columns: 'text' and 'label'")
    df = df.dropna(subset=["text", "label"])
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xv = vec.fit_transform(X)
    clf = MultinomialNB()
    clf.fit(Xv, y)
    return clf, vec, len(df), sorted(set(y))


def train_url_model_from_df(df):
    """
    Expects df with columns: 'url' and 'label'
    Returns (clf, vectorizer, nrows, labels_list)
    """
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("URL CSV must contain columns: 'url' and 'label'")
    df = df.dropna(subset=["url", "label"])
    X = df["url"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    # char n-grams are a good baseline for URLs
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    Xv = vec.fit_transform(X)
    clf = MultinomialNB()
    clf.fit(Xv, y)
    return clf, vec, len(df), sorted(set(y))


def probs_line(model, X):
    """Return human friendly probability string if model supports predict_proba."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
        classes = list(model.classes_)
        pairs = sorted(zip(classes, p), key=lambda t: -t[1])
        return "Confidence → " + " • ".join([f"{c}: {v*100:.1f}%" for c, v in pairs])
    return None


def base_ctx(**extra):
    ctx = {
        "msg_model_ready": msg_model is not None and msg_vectorizer is not None,
        "url_model_ready": url_model is not None and url_vectorizer is not None,
        "msg_train_msg": None,
        "msg_train_log": None,
        "url_train_msg": None,
        "url_train_log": None,
        "msg_pred_error": None,
        "msg_prediction": None,
        "msg_is_phish": None,
        "msg_proba_line": None,
        "url_pred_error": None,
        "url_prediction": None,
        "url_is_phish": None,
        "url_proba_line": None,
        "last_message": None,
        "last_url": None,
    }
    ctx.update(extra)
    return ctx


# Load artifacts on startup if they exist
load_artifacts()

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **base_ctx())


@app.route("/upload", methods=["POST"])
def upload():
    """
    Generic uploader. Saves uploaded file into uploads/ (preserves filename).
    Use the Upload form on UI to store files like messages.csv or urls.csv.
    """
    if "file" not in request.files:
        return render_template("index.html", **base_ctx(msg_train_msg="No file part in request."))

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", **base_ctx(msg_train_msg="No selected file."))

    if not allowed_file(file.filename):
        return render_template("index.html", **base_ctx(msg_train_msg="Only CSV files allowed."))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    # rewind stream and save
    file.stream.seek(0)
    file.save(save_path)
    msg = f"Uploaded and saved: {filename} (-> uploads/)"
    return render_template("index.html", **base_ctx(msg_train_msg=msg))


@app.route("/train_message", methods=["POST"])
def train_message():
    """
    Train message model. Form may POST an uploaded CSV under 'message_dataset'
    or the app will look for uploads/messages.csv saved earlier.
    """
    global msg_model, msg_vectorizer
    f = request.files.get("message_dataset")
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], "messages.csv")
    ctx = {}

    try:
        if f and f.filename:
            df = safe_read_csv_file_storage(f)
        else:
            if not os.path.exists(saved_path):
                ctx["msg_train_msg"] = "⚠️ messages.csv not found in uploads/ and no file provided."
                return render_template("index.html", **base_ctx(**ctx))
            df = pd.read_csv(saved_path)

        model_tmp, vec_tmp, nrows, labels = train_message_model_from_df(df)

        # persist
        with open(MSG_MODEL_FILE, "wb") as fh:
            pickle.dump(model_tmp, fh)
        with open(MSG_VEC_FILE, "wb") as fh:
            pickle.dump(vec_tmp, fh)

        # load into memory
        msg_model, msg_vectorizer = model_tmp, vec_tmp
        load_artifacts()

        ctx["msg_train_msg"] = "✅ Message model trained & saved."
        ctx["msg_train_log"] = f"Rows: {nrows}  Labels: {labels}"
    except Exception as e:
        ctx["msg_train_msg"] = "❌ Message training failed."
        ctx["msg_train_log"] = f"{type(e).__name__}: {e}"

    return render_template("index.html", **base_ctx(**ctx))


@app.route("/train_url", methods=["POST"])
def train_url():
    """
    Train URL model. Form may POST an uploaded CSV under 'url_dataset'
    or the app will look for uploads/urls.csv saved earlier.
    """
    global url_model, url_vectorizer
    f = request.files.get("url_dataset")
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], "urls.csv")
    ctx = {}

    try:
        if f and f.filename:
            df = safe_read_csv_file_storage(f)
        else:
            if not os.path.exists(saved_path):
                ctx["url_train_msg"] = "⚠️ urls.csv not found in uploads/ and no file provided."
                return render_template("index.html", **base_ctx(**ctx))
            df = pd.read_csv(saved_path)

        model_tmp, vec_tmp, nrows, labels = train_url_model_from_df(df)

        with open(URL_MODEL_FILE, "wb") as fh:
            pickle.dump(model_tmp, fh)
        with open(URL_VEC_FILE, "wb") as fh:
            pickle.dump(vec_tmp, fh)

        url_model, url_vectorizer = model_tmp, vec_tmp
        load_artifacts()

        ctx["url_train_msg"] = "✅ URL model trained & saved."
        ctx["url_train_log"] = f"Rows: {nrows}  Labels: {labels}"
    except Exception as e:
        ctx["url_train_msg"] = "❌ URL training failed."
        ctx["url_train_log"] = f"{type(e).__name__}: {e}"

    return render_template("index.html", **base_ctx(**ctx))


@app.route("/predict_message", methods=["POST"])
def predict_message():
    ctx = {}
    msg = (request.form.get("message") or "").strip()
    ctx["last_message"] = msg

    if not (msg_model and msg_vectorizer):
        ctx["msg_pred_error"] = "Model not trained yet. Please train messages model first."
        return render_template("index.html", **base_ctx(**ctx))

    if not msg:
        ctx["msg_pred_error"] = "Please enter some text to check."
        return render_template("index.html", **base_ctx(**ctx))

    try:
        X = msg_vectorizer.transform([msg])
        pred = msg_model.predict(X)[0]
        ctx["msg_prediction"] = pred
        ctx["msg_is_phish"] = (str(pred).lower() == "phishing")
        ctx["msg_proba_line"] = probs_line(msg_model, X)
    except Exception as e:
        ctx["msg_pred_error"] = f"Prediction error: {type(e).__name__}: {e}"

    return render_template("index.html", **base_ctx(**ctx))


@app.route("/predict_url", methods=["POST"])
def predict_url():
    ctx = {}
    u = (request.form.get("url") or "").strip()
    ctx["last_url"] = u

    if not (url_model and url_vectorizer):
        ctx["url_pred_error"] = "URL model not trained yet. Please train urls model first."
        return render_template("index.html", **base_ctx(**ctx))

    if not u:
        ctx["url_pred_error"] = "Please enter a URL to check."
        return render_template("index.html", **base_ctx(**ctx))

    try:
        X = url_vectorizer.transform([u])
        pred = url_model.predict(X)[0]
        ctx["url_prediction"] = pred
        ctx["url_is_phish"] = (str(pred).lower() == "phishing")
        ctx["url_proba_line"] = probs_line(url_model, X)
    except Exception as e:
        ctx["url_pred_error"] = f"Prediction error: {type(e).__name__}: {e}"

    return render_template("index.html", **base_ctx(**ctx))


# ---------- Start ----------
if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)
