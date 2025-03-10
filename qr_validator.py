import os
import re
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import tldextract
import whois
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import logging
import cv2
from pyzbar.pyzbar import decode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load NPCI registered UPI handles from dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "npci_registered_upi_apps.xlsx")

def npci_registered_apps(csv_path):
    """Load NPCI registered UPI apps from an Excel file."""
    try:
        df = pd.read_excel(csv_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        logging.error(f"Error loading NPCI registered apps: {e}")
        return pd.DataFrame()

npci_apps = npci_registered_apps(CSV_PATH)

# Load MCC CSV file
MCC_CSV_PATH = os.path.join(BASE_DIR, "mcc.csv")

def load_mcc_data(csv_path):
    """Load MCC data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        logging.error(f"Error loading MCC data: {e}")
        return pd.DataFrame()

mcc_data = load_mcc_data(MCC_CSV_PATH)

# QR Code Extraction & Parsing
def extract_qr_text(image_path):
    """Extracts QR code data from an image."""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)

        for obj in decoded_objects:
            return obj.data.decode("utf-8")

        return None
    except Exception as e:
        logging.error(f"Error extracting QR code: {e}")
        return None

def parse_upi_qr(qr):
    """Parses the UPI QR code string into a dictionary."""
    qr_text = extract_qr_text(qr)
    qr_pattern = re.compile(r"upi://pay\?(.*)")
    match = qr_pattern.match(qr_text)

    if not match:
        return {"error": "Invalid UPI QR Code format"}

    query_params = match.group(1).split("&")
    parsed_data = {param.split("=")[0]: param.split("=")[1] if "=" in param else "" for param in query_params}

    return parsed_data

def validate_upi_id(upi_id, df):
    """Validates UPI ID against NPCI registered handles."""
    upi_pattern = r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+$"

    if "@" not in upi_id:
        return {"UPI ID": upi_id, "Valid": False, "Error": "Invalid format", "App": None, "Bank": None}

    handle = '@' + upi_id.split("@")[1]
    if handle in df['Handle Name'].values:
        app = df.loc[df['Handle Name'] == handle, 'TPAP'].values[0]
        bank = df.loc[df['Handle Name'] == handle, 'PSP Banks'].values[0]
        match = re.match(upi_pattern, upi_id)
        if not match:
            return {"UPI ID": upi_id, "Valid": False, "Error": "Invalid UPI ID format"}
        return {"UPI ID": upi_id, "Valid": True, "App": app, "Bank": bank, "Handle": handle}
    
    return {"UPI ID": upi_id, "Valid": False, "Error": "Handle not found", "App": None, "Bank": None}

def verify_mcc(mcc):
    if not mcc or mcc == "0000":
        return "Individual Receiver"

    descriptions = mcc_data[mcc_data['MCC'].astype(str) == str(mcc)]['Description'].tolist()

    if descriptions:
        return list(set(descriptions)) 
    else:
        return "Unknown MCC"

HEADERS = {"User-Agent": "Mozilla/5.0"}
def get_genuine_urls():
    try:
        response = requests.get("https://moz.com/top500", headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        sites = []
        for row in soup.select("table tbody tr"):
            cols = row.find_all("td")
            if len(cols) > 1:
                site = cols[1].text.strip()
                sites.append(f"https://{site}")

        return sites[:100]
    except Exception as e:
        print(f"Error scraping genuine URLs: {e}")
        return

def get_phishing_urls():
    try:
        response = requests.get("https://openphish.com/feed.txt")
        if response.status_code == 200:
            phishing_urls = response.text.split("\n")
            return phishing_urls[:100]
        else:
            print(f"Error: OpenPhish returned status {response.status_code}")
            return
    except Exception as e:
        print(f"Error fetching phishing URLs: {e}")
        return

def get_whois_features(url):
    """Extracts WHOIS-based domain features for phishing detection."""
    try:
        domain = urlparse(url).netloc
        domain_info = whois.whois(domain)

        creation_date = domain_info.creation_date
        expiry_date = domain_info.expiration_date

        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(expiry_date, list):
            expiry_date = expiry_date[0]

        domain_age = (datetime.datetime.now() - creation_date).days if creation_date else 730
        domain_age = max(domain_age, 730)

        days_to_expire = (expiry_date - datetime.datetime.now()).days if expiry_date else 365
        days_to_expire = max(days_to_expire, 180)

        private_registration = 1 if domain_info.registrant_name in [None, "REDACTED FOR PRIVACY"] else 0

        return [domain_age, days_to_expire, private_registration]
    except:
        return [730, 180, 0]

def extract_features(url):
    parsed_url = urlparse(url)

    url_length = len(url)
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = sum(url.count(c) for c in ['-', '_', '.', '=', '&', '?', '%', '@'])

    domain_info = tldextract.extract(url)
    num_subdomains = domain_info.subdomain.count(".")
    domain_length = len(domain_info.domain)

    https = 1 if parsed_url.scheme == "https" else 0
    shortener_domains = ["bit.ly", "t.co", "tinyurl.com", "is.gd", "shorte.st"]
    is_shortened = 1 if any(short in parsed_url.netloc for short in shortener_domains) else 0

    whois_features = get_whois_features(url)

    return [
        url_length, https, num_special_chars, num_digits, num_subdomains, domain_length,
        is_shortened, *whois_features
    ]

# Machine Learning Models
def train_models():
    """Trains machine learning models for phishing detection."""
    genuine_urls = get_genuine_urls()
    phishing_urls = get_phishing_urls()

    data, labels = [], []

    for url in genuine_urls:
        data.append(extract_features(url))
        labels.append(0)  # Safe

    for url in phishing_urls:
        data.append(extract_features(url))
        labels.append(1)  # Phishing

    df = pd.DataFrame(data, columns=[
        "URL_Length", "HTTPS", "Special_Chars", "Digits", "Subdomains", "Domain_Length",
        "Shortened", "Domain_Age", "Days_To_Expire", "Private_Registration"
        ])
    df["Label"] = labels

    X = df.drop(columns=["Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    xgb_model = XGBClassifier(eval_metric="logloss")
    xgb_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(genuine_urls + phishing_urls)
    X_seq = tokenizer.texts_to_sequences(genuine_urls + phishing_urls)
    X_seq = pad_sequences(X_seq, maxlen=100)
    y_seq = np.array([0] * len(genuine_urls) + [1] * len(phishing_urls))

    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    lstm_model = Sequential([
                Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32),
                LSTM(64, return_sequences=True),
                LSTM(32),
                Dense(16, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid")
            ])
    lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    lstm_model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=16, validation_split=0.2)

    return xgb_model, rf_model, iso_forest, scaler, lstm_model, X

xgb_model, rf_model, iso_forest, scaler, lstm_model, X = train_models()

# Phishing Prediction
def predict_phishing(url):
    """Predicts whether a URL is phishing or safe."""
    ml_features = extract_features(url)
    ml_features_df = pd.DataFrame([ml_features], columns=X.columns)

    rf_score = rf_model.predict(ml_features_df.to_numpy())[0]
    xgb_score = xgb_model.predict(ml_features_df)[0]
    iso_score = iso_forest.decision_function(ml_features_df.to_numpy())[0]

    tokenizer = Tokenizer()
    seq_input = tokenizer.texts_to_sequences([url])
    seq_input = pad_sequences(seq_input, maxlen=100)
    lstm_score = lstm_model.predict(seq_input)[0][0]

    final_score = (rf_score + xgb_score - iso_score + lstm_score) / 4
    return "Phishing" if final_score > 0.6 else "Safe"

# Risk Analysis
def calculate_risk_score(is_upi_valid, phishing_status, mcc_status):
    """Calculates a risk score (0-100) based on UPI ID, Phishing, and MCC data."""
    upi_risk = 0 if is_upi_valid else 50
    phishing_risk = 0 if phishing_status == "Safe" else 50
    mcc_risk = 0 if mcc_status != "Unknown MCC" else 30

    risk_score = upi_risk + phishing_risk + mcc_risk
    return risk_score

def predict_qr_code_safety(qr_data):
    """Predicts the safety of a QR code based on UPI ID, RefUrl, and MCC."""
    upi_id = qr_data.get("pa", "")
    ref_url = qr_data.get("refUrl", "")
    mcc = qr_data.get("mc", "")

    # Validate UPI ID
    validation_result = validate_upi_id(upi_id, npci_apps)
    is_upi_valid = validation_result["Valid"]
    upi_result = (
        f"UPI ID '{upi_id}' is **valid** and registered under NPCI."
        if is_upi_valid else
        f"UPI ID '{upi_id}' is **not registered** under NPCI. It could be fraudulent."
    )

    # Check Phishing Risk
    phishing_status = predict_phishing(ref_url) if ref_url else "Safe"
    phishing_result = (
        f"The reference URL '{ref_url}' is **potentially a phishing site**."
        if phishing_status == "Phishing" else
        f"The reference URL '{ref_url}' is **safe**."
    )

    # Verify MCC
    mcc_status = verify_mcc(mcc)
    if isinstance(mcc_status, list):
        mcc_description = ", ".join(mcc_status)
    else:
        mcc_description = mcc_status

    mcc_result = (
        f"The MCC '{mcc}' belongs to **{mcc_description}**."
        if mcc_status != "Unknown MCC" else
        f"The MCC '{mcc}' is **not recognized**. It could be suspicious."
    )

    # Calculate Risk Score
    risk_score = calculate_risk_score(is_upi_valid, phishing_status, mcc_status)

    # Generate Detailed Risk Analysis
    risk_level = (
        "**HIGH RISK QR CODE!**" if risk_score >= 70 else
        "**MODERATE RISK QR CODE.**" if 40 <= risk_score < 70 else
        "**SAFE QR CODE.**"
    )

    # Return Detailed Analysis as Text
    detailed_report = f"""
    **QR Code Analysis Report**
    --------------------------------------
    **UPI ID Analysis:**  
    {upi_result}

    **Phishing Link Analysis:**  
    {phishing_result}

    **Merchant Category Code (MCC) Analysis:**  
    {mcc_result}

    **Final Risk Score:**  
    **{risk_score}/100**  
    {risk_level}
    """
    
    return detailed_report