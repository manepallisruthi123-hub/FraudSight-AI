import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from xgboost import XGBClassifier

st.set_page_config(page_title="FraudSight AI", layout="wide", page_icon="🛡️")

# ==========================================================
# OCR imports (guarded — app must not crash if unavailable)
# ==========================================================
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ==========================================================
# Load model artifacts
# ==========================================================
@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model('xgb_fraud_model.json')
    threshold = joblib.load('threshold.pkl')
    explainer = shap.TreeExplainer(model)
    return model, threshold, explainer

model, threshold, explainer = load_artifacts()
sample_df = pd.read_csv('sample_transactions.csv')
EXPECTED_COLUMNS = sample_df.columns.tolist()

# ==========================================================
# Helper functions — core model
# ==========================================================
def get_risk_level(prob):
    if prob >= threshold:
        return "🔴 High Risk"
    elif prob >= threshold * 0.4:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"


def plain_language_result(prob, is_flagged):
    """Human-readable one-liner instead of raw jargon."""
    pct = int(round(prob * 100))
    if is_flagged:
        return f"⚠️ This looks suspicious — about a {pct}% chance of fraud. We'd recommend a manual review."
    elif prob >= threshold * 0.4:
        return f"🤔 This is borderline — around a {pct}% fraud likelihood. Worth a second look."
    else:
        return f"✅ This looks safe — only about a {pct}% chance of fraud."


def validate_columns(df):
    df_cols = set(df.columns) - {'Class'}
    expected_cols = set(EXPECTED_COLUMNS)
    missing = expected_cols - df_cols
    extra = df_cols - expected_cols
    return missing, extra


def generate_explanation(shap_values_row, feature_names, is_flagged):
    shap_series = pd.Series(shap_values_row, index=feature_names)
    top_features = shap_series.abs().sort_values(ascending=False).head(3)
    feature_list = ", ".join(top_features.index.tolist())
    if is_flagged:
        return (
            f"This transaction was classified as **High Risk** because features "
            f"**{feature_list}** significantly increased the fraud score. "
            f"The transaction shows patterns similar to previously flagged fraudulent transactions."
        )
    else:
        return (
            f"This transaction was classified as **Low Risk** because features "
            f"**{feature_list}** did not show unusual patterns. "
            f"No signs of previously flagged fraudulent behavior were detected."
        )


def show_feature_note():
    st.caption(
        "Note: The dataset anonymizes transaction attributes for privacy. "
        "Features V1–V28 are transformed variables rather than real-world "
        "labels such as merchant, device, or location."
    )


def draw_gauge(prob, threshold):
    """Semi-circle gauge chart."""
    fig, ax = plt.subplots(figsize=(4.2, 2.4), subplot_kw={'aspect': 'equal'})

    bands = [(0, 0.4 * threshold, '#2ecc71'), (0.4 * threshold, threshold, '#ffa500'), (threshold, 1.0, '#ff4b4b')]
    for start, end, color in bands:
        theta1 = 180 - (start * 180)
        theta2 = 180 - (end * 180)
        wedge = mpatches.Wedge((0, 0), 1, theta2, theta1, width=0.35, facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(wedge)

    needle_angle = np.radians(180 - (prob * 180))
    needle_len = 0.85
    ax.plot([0, needle_len * np.cos(needle_angle)], [0, needle_len * np.sin(needle_angle)], color='#1E2761', linewidth=3, solid_capstyle='round')
    ax.add_patch(mpatches.Circle((0, 0), 0.05, facecolor='#1E2761'))

    ax.text(0, -0.35, f"{prob*100:.1f}%", ha='center', va='center', fontsize=22, fontweight='bold', color='#1E2761')
    ax.text(0, -0.58, "fraud likelihood", ha='center', va='center', fontsize=10, color='#666666')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.7, 1.1)
    ax.axis('off')
    return fig


def show_single_alert(proba):
    if proba >= threshold:
        st.error(f"🚨 **FLAGGED — HIGH RISK** — Transaction flagged at {proba:.1%} risk. Immediate review recommended.")
        st.toast("🚨 High-risk transaction flagged!", icon="🚨")
    elif proba >= threshold * 0.4:
        st.warning(f"⚠️ Medium risk transaction ({proba:.1%}) — monitor for confirmation.")
        st.toast("⚠️ Medium-risk transaction flagged", icon="⚠️")
    else:
        st.toast("✅ Transaction looks legitimate", icon="✅")


def show_batch_alert(fraud_count, total, highest_risk):
    if fraud_count > 0:
        st.markdown(
            f"""
            <div style="background-color:#ffe6e6; padding:20px; border-radius:10px; border-left:6px solid #ff4b4b;">
                <h2 style="color:#cc0000; margin-top:0;">🚨 HIGH RISK ALERT</h2>
                <p style="font-size:18px; margin-bottom:5px;">
                    Out of <b>{total:,}</b> transactions checked, <b>{fraud_count}</b> look suspicious and need review.
                </p>
                <p style="font-size:18px; margin-bottom:5px;">
                    Riskiest one scored: <b>{highest_risk:.0%}</b> chance of fraud
                </p>
                <p style="font-size:16px; color:#555; margin-top:10px;">
                    <b>Recommendation:</b> Review these transactions before approval.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.toast(f"🚨 {fraud_count} suspicious transaction(s) flagged!", icon="🚨")
    else:
        st.markdown(
            f"""
            <div style="background-color:#e6ffed; padding:20px; border-radius:10px; border-left:6px solid #2ecc71;">
                <h2 style="color:#1e8449; margin-top:0;">✅ All Clear</h2>
                <p style="font-size:16px; color:#555;">
                    Checked all {total:,} transactions — none of them look suspicious.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.toast("✅ No suspicious transactions flagged in this batch", icon="✅")
        st.balloons()

# ==========================================================
# Helper functions — SMS / Receipt text pattern scanner (rule-based)
# ==========================================================
PHISHING_PATTERNS = [
    (r"\b(won|winner|congratulations|lucky draw|lottery)\b", "Prize/lottery bait language", 25),
    (r"\b(click here|click now|verify now|act now|update immediately)\b", "Urgency / call-to-action pressure", 20),
    (r"\b(suspend|blocked|expire[sd]?|deactivat)\w*\b", "Threat of account suspension", 20),
    (r"(https?://|bit\.ly|tinyurl|[a-z0-9-]+\.(xyz|top|club|info|tk))", "Suspicious/shortened link", 20),
    (r"\b(otp|pin|cvv|password)\b.{0,15}\b(share|send|enter|confirm)\b", "Requests OTP/PIN/CVV — legitimate services never ask this", 30),
    (r"\b(refund|cashback|reward)\b.{0,20}\b(claim|collect)\b", "Unexpected refund/cashback claim", 15),
    (r"₹\s?[5-9][0-9]{4,}|₹\s?[0-9]{6,}", "Unusually large amount mentioned", 10),
    (r"\bdear (customer|user|sir/madam)\b", "Generic greeting (not personalized)", 10),
    (r"\b(share (this )?screenshot|send screenshot)\b", "Requests to share/send a screenshot", 15),
    (r"\b(customer care|helpline)\b.{0,20}\b(\d{10}|call)\b", "Fake customer-care/helpline number pattern", 15),
]

TRUSTED_SENDER_HINTS = ["phonepe", "paytm", "gpay", "google pay", "bhim", "upi"]


def scan_text_for_fraud_patterns(text):
    text_lower = text.lower()
    score = 0
    matched = []
    for pattern, label, weight in PHISHING_PATTERNS:
        if re.search(pattern, text_lower):
            score += weight
            matched.append((label, weight))
    score = min(score, 100)
    return score, matched


def risk_band(score):
    if score >= 50:
        return "🔴 High Risk — Possible phishing/scam", "#ff4b4b"
    elif score >= 20:
        return "🟠 Medium Risk — Some suspicious signals", "#ffa500"
    else:
        return "🟢 Low Risk — No strong red flags", "#2ecc71"


def safety_recommendation(score):
    if score >= 50:
        return "Do not click links or share OTP, PIN, CVV, or passwords. Verify the transaction directly through the official payment app."
    elif score >= 20:
        return "Some suspicious signals were detected. Verify the sender and transaction directly through the official app."
    else:
        return "No strong phishing patterns were detected. This does not guarantee that the transaction is legitimate."


def render_pattern_scan_result(score, matched, text_for_brand_check):
    risk_word, color = risk_band(score)
    st.markdown(
        f"""
        <div style="background-color:{color}22; padding:16px; border-radius:10px; border-left:6px solid {color};">
            <h3 style="margin-top:0; color:{color};">{risk_word}</h3>
            <p style="font-size:15px; margin-bottom:0;">Risk Score: <b>{score}/100</b></p>
        </div>
        """, unsafe_allow_html=True
    )

    if matched:
        st.markdown("**Detected:**")
        for label, weight in matched:
            st.markdown(f"- ⚠️ {label} *(+{weight})*")
    else:
        st.markdown("No known phishing patterns detected in this text.")

    st.markdown(f"**Safety Recommendation:** {safety_recommendation(score)}")

    if any(s in text_for_brand_check.lower() for s in TRUSTED_SENDER_HINTS) and score >= 20:
        st.warning(
            "⚠️ This text mentions a payment app name (PhonePe/Paytm/GPay/UPI) alongside suspicious "
            "patterns — scammers often impersonate these brands. Never share your PIN or OTP, "
            "and verify directly in the official app, not via a link in the message."
        )

# ==========================================================
# Helper functions — Receipt Screenshot OCR
# ==========================================================
def extract_text_from_image(pil_image):
    try:
        max_dim = 2200
        if max(pil_image.size) > max_dim:
            ratio = max_dim / max(pil_image.size)
            new_size = tuple(int(dim * ratio) for dim in pil_image.size)
            pil_image = pil_image.resize(new_size)
        text = pytesseract.image_to_string(pil_image)
        return text.strip(), None
    except pytesseract.TesseractNotFoundError:
        return "", "tesseract_not_found"
    except Exception as e:
        return "", f"ocr_error:{e}"


def extract_receipt_fields(text):
    """Robust field extraction from OCR text with typo-tolerance."""
    fields = {}

    # 1. Amount: handles 'Amount 12.880.00', 'Amount: 2890.00', or '₹2,890.00'
    amt_match = re.search(
        r"(?:Amount|Amt|Paid|Total)\s*[:\-'\"]*\s*(?:₹|Rs\.?|INR)?\s*([\d,\.]+(?:\.\d{1,2})?)",
        text,
        re.IGNORECASE
    )

    if not amt_match:
        amt_match = re.search(
            r"(?:₹|Rs\.?|INR)\s*([\d,\.]+)",
            text,
            re.IGNORECASE
        )

    if not amt_match:
        amt_match = re.search(
            r"\b(\d{1,7}\.\d{1,2})\b",
            text
        )

    fields['amount'] = (
        f"₹{amt_match.group(1)}"
        if amt_match else "Not detected"
    )

    # 2. Date
    date_match = re.search(
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        text
    )

    if not date_match:
        date_match = re.search(
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
            text,
            re.IGNORECASE
        )

    fields['date'] = (
        date_match.group(1)
        if date_match else "Not detected"
    )

    # 3. Time
    time_match = re.search(
        r"(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)",
        text
    )

    fields['time'] = (
        time_match.group(1)
        if time_match else "Not detected"
    )

    # 4. Payment App
    apps_found = []
    for keyword, label in [
        ("phonepe", "PhonePe"),
        ("paytm", "Paytm"),
        ("google pay", "Google Pay"),
        ("gpay", "Google Pay"),
        ("upi", "UPI")
    ]:
        if keyword in text.lower() and label not in apps_found:
            apps_found.append(label)

    fields['payment_app'] = (
        ", ".join(apps_found)
        if apps_found else "Not detected"
    )

    # 5. Transaction / Reference ID (handles 'xn ID', 'ef No', 'UTR', or raw long numbers)
    txn_match = re.search(
        r"(?:UTR|[T]?xn(?:\s?ID)?|Transaction\s?ID|[R]?ef(?:\s?No\.?|Number))"
        r"[:\s\.'\"]*([A-Za-z0-9]{10,25})",
        text,
        re.IGNORECASE
    )

    if not txn_match:
        txn_match = re.search(r"\b(T\d{10,22})\b", text, re.IGNORECASE)

    if not txn_match:
        txn_match = re.search(r"\b([A-Za-z0-9]{12,22})\b", text)

    fields['txn_id'] = (
        txn_match.group(1)
        if txn_match else "Not detected"
    )

    # 6. Merchant / Receiver (handles 'Paid to', 'Paid ta', 'Paid 2')
    merchant_match = re.search(
        r"(?:Paid\s?[tT][oa2]|Received\s+by|Payee)"
        r"[:\s\n]+([A-Za-z0-9 .&@]{2,40})",
        text,
        re.IGNORECASE
    )

    merchant_val = merchant_match.group(1).strip() if merchant_match else "Not detected"
    merchant_val = re.sub(r"^[^A-Za-z0-9]+", "", merchant_val)
    fields['merchant'] = merchant_val if merchant_val else "Not detected"

    return fields

# ==========================================================
# Header
# ==========================================================
st.title("🛡️ FraudSight AI")
st.caption("Explainable real-time fraud detection using XGBoost + SHAP")

st.sidebar.header("Model Info")
st.sidebar.metric("Precision", "81%")
st.sidebar.metric("Recall", "79%")
st.sidebar.metric("PR-AUC", "0.817")
st.sidebar.caption("In plain terms: about 4 out of 5 flagged transactions are real fraud, and the model catches about 4 out of 5 actual fraud cases.")
st.sidebar.write(f"Decision threshold: {threshold:.4f}")
st.sidebar.info("Trained on 283,726 transactions (0.17% fraud rate) using SMOTE-resampled XGBoost.")

with st.sidebar.expander("📋 Required CSV columns"):
    st.code(", ".join(EXPECTED_COLUMNS))

st.divider()
mode = st.radio(
    "Choose mode:",
    [
        "Single Transaction (sample data)",
        "Upload CSV (batch analysis)",
        "SMS / Receipt Check (Beta)",
        "📷 Receipt Screenshot Check (Beta)",
    ],
    horizontal=True
)

# ==========================================================
# MODE 1: Single transaction
# ==========================================================
if mode == "Single Transaction (sample data)":
    st.subheader("Select a transaction to analyze")
    row_idx = st.selectbox("Choose a sample transaction (index)", sample_df.index)
    transaction = sample_df.loc[[row_idx]]
    st.dataframe(transaction)

    if st.button("Analyze Transaction", type="primary"):
        proba = model.predict_proba(transaction)[:, 1][0]
        prediction = "🚨 Flagged (High Risk)" if proba >= threshold else "✅ Legitimate"
        is_flagged = proba >= threshold

        gcol, rcol = st.columns([1, 1.4])
        with gcol:
            fig = draw_gauge(proba, threshold)
            st.pyplot(fig, use_container_width=True)
        with rcol:
            st.markdown(f"### {prediction}")
            st.markdown(f"**{plain_language_result(proba, is_flagged)}**")
            st.caption(f"Risk level: {get_risk_level(proba)}")

        show_single_alert(proba)

        st.subheader("Why did the model make this decision?")
        shap_values = explainer.shap_values(transaction)
        explanation_text = generate_explanation(shap_values[0], transaction.columns.tolist(), is_flagged)
        st.info(f"**Reason for Prediction:**\n\n{explanation_text}")

        shap_explanation = shap.Explanation(
            values=shap_values[0], base_values=explainer.expected_value,
            data=transaction.iloc[0].values, feature_names=transaction.columns.tolist()
        )
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation, show=False)
        st.pyplot(fig2)
        show_feature_note()

# ==========================================================
# MODE 2: CSV upload (batch analysis)
# ==========================================================
elif mode == "Upload CSV (batch analysis)":
    st.subheader("Upload a transaction CSV")
    st.caption("File must contain the same columns as the training data (Time, V1–V28, Amount).")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"⚠️ Couldn't read this file as a CSV. Error: {e}")
            st.stop()

        missing, extra = validate_columns(batch_df)
        if missing:
            st.error(
                f"⚠️ This CSV is missing {len(missing)} required column(s) and cannot be processed.\n\n"
                f"**Missing columns:** {', '.join(sorted(missing))}"
            )
            st.info(
                "This app was trained on the Kaggle Credit Card Fraud dataset, which requires exactly these "
                "columns: `Time`, `V1`–`V28`, `Amount` (and optionally `Class`)."
            )
            st.stop()
        if extra:
            st.warning(f"ℹ️ Ignoring {len(extra)} extra column(s): {', '.join(sorted(extra))}")

        features_only = batch_df[EXPECTED_COLUMNS]
        probas = model.predict_proba(features_only)[:, 1]
        predictions = (probas >= threshold).astype(int)
        risk_labels = [get_risk_level(p) for p in probas]

        results_df = batch_df.copy()
        results_df.insert(0, 'Transaction_ID', results_df.index)
        results_df['Fraud_Probability'] = probas.round(4)
        results_df['Risk_Level'] = risk_labels
        results_df['Prediction'] = np.where(predictions == 1, '🚨 Flagged', '✅ Legitimate')

        st.divider()
        st.subheader("📊 Fraud Summary — In Plain Terms")
        total = len(results_df)
        fraud_count = int(predictions.sum())
        legit_count = total - fraud_count
        fraud_rate = (fraud_count / total) * 100
        highest_risk = results_df['Fraud_Probability'].max()

        st.markdown(
            f"##### Out of **{total:,}** transactions checked, **{fraud_count}** look suspicious "
            f"({fraud_rate:.2f}% of the batch) and **{legit_count:,}** look normal."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total:,}")
        col2.metric("🚨 Flagged", f"{fraud_count:,}")
        col3.metric("✅ Legitimate", f"{legit_count:,}")
        col4.metric("⚠️ Fraud Rate", f"{fraud_rate:.2f}%")

        show_batch_alert(fraud_count, total, highest_risk)

        st.divider()
        st.subheader("📈 Fraud Trends")
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.hist(results_df['Fraud_Probability'], bins=30, color='crimson', alpha=0.7)
            ax1.set_title('Fraud Probability Distribution')
            ax1.set_xlabel('Fraud Probability')
            ax1.set_ylabel('Count')
            st.pyplot(fig1)
        with col2:
            risk_counts = results_df['Risk_Level'].value_counts()
            color_map = {'🟢 Low Risk': '#2ecc71', '🟠 Medium Risk': '#ffa500', '🔴 High Risk': '#ff4b4b'}
            pie_colors = [color_map.get(label, '#999999') for label in risk_counts.index]
            fig2, ax2 = plt.subplots()
            ax2.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=pie_colors)
            ax2.set_title('Risk Level Breakdown')
            st.pyplot(fig2)

        st.subheader("Transaction Amount vs Fraud Risk")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        colors = results_df['Prediction'].map({'🚨 Flagged': 'red', '✅ Legitimate': 'steelblue'})
        ax3.scatter(results_df['Amount'], results_df['Fraud_Probability'], c=colors, alpha=0.5, s=15)
        ax3.set_xlabel('Transaction Amount')
        ax3.set_ylabel('Fraud Probability')
        ax3.set_title('Amount vs Fraud Risk')
        st.pyplot(fig3)

        st.divider()
        st.subheader("Transaction Results")
        show_only_fraud = st.checkbox("Show only flagged transactions")
        display_df = results_df[results_df['Prediction'] == '🚨 Flagged'] if show_only_fraud else results_df
        display_cols = ['Transaction_ID', 'Amount', 'Fraud_Probability', 'Risk_Level', 'Prediction']

        MAX_STYLED_ROWS = 5000
        if len(display_df) <= MAX_STYLED_ROWS:
            st.dataframe(
                display_df[display_cols].style.apply(
                    lambda row: ['background-color: #ffe6e6' if row['Prediction'] == '🚨 Flagged' else '' for _ in row], axis=1
                ), use_container_width=True
            )
        else:
            st.info(f"Showing {len(display_df):,} rows — highlighting disabled for large tables (performance).")
            st.dataframe(display_df[display_cols], use_container_width=True)

        if fraud_count > 0:
            st.divider()
            st.subheader("🔍 Explain a Flagged Transaction")
            fraud_ids = results_df[results_df['Prediction'] == '🚨 Flagged']['Transaction_ID'].tolist()
            selected_id = st.selectbox("Choose a flagged transaction to explain", fraud_ids)
            selected_row = features_only.loc[[selected_id]]
            shap_values_single = explainer.shap_values(selected_row)
            explanation_text = generate_explanation(shap_values_single[0], selected_row.columns.tolist(), True)
            st.info(f"**Reason for Prediction:**\n\n{explanation_text}")

            shap_explanation = shap.Explanation(
                values=shap_values_single[0], base_values=explainer.expected_value,
                data=selected_row.iloc[0].values, feature_names=selected_row.columns.tolist()
            )
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation, show=False)
            st.pyplot(fig4)
            show_feature_note()

        st.download_button(
            "⬇️ Download Prediction Report (CSV)",
            data=display_df[display_cols].to_csv(index=False),
            file_name="fraud_prediction_report.csv", mime="text/csv"
        )

# ==========================================================
# MODE 3: SMS / Receipt Text Check (Beta)
# ==========================================================
elif mode == "SMS / Receipt Check (Beta)":
    st.subheader("📩 SMS / Payment Receipt Fraud Check")
    st.markdown(
        "<span style='background-color:#FFF3CD; padding:4px 10px; border-radius:6px; font-size:13px;'>"
        "🧪 BETA — Rule-based pattern scanner, not the trained ML model above</span>",
        unsafe_allow_html=True
    )
    st.caption(
        "This checks SMS or payment-receipt text for **common phishing/scam patterns** — suspicious links, "
        "urgency language, OTP requests, etc. It is a simple keyword/pattern heuristic, "
        "not a trained machine-learning classifier like the transaction model above."
    )

    example = st.selectbox(
        "Try an example, or paste your own text below:",
        [
            "— type your own —",
            "Dear Customer, your PhonePe account will be BLOCKED in 24 hrs. Click here to verify: http://bit.ly/verify-now and enter your UPI PIN.",
            "You have WON ₹50,000 cashback! Claim now: paytm-reward.xyz/claim before it expires.",
            "₹500 debited from your account for grocery purchase at BigBasket via PhonePe UPI. Avl bal: ₹4,200.",
        ]
    )
    default_text = "" if example == "— type your own —" else example
    sms_text = st.text_area("SMS / receipt text:", value=default_text, height=110)

    if st.button("Scan Text", type="primary"):
        if not sms_text.strip():
            st.warning("Please paste or select some text first.")
        else:
            score, matched = scan_text_for_fraud_patterns(sms_text)
            render_pattern_scan_result(score, matched, sms_text)

    st.divider()
    st.caption(
        "How this differs from the main model: the transaction classifier above is a trained XGBoost model "
        "evaluated on real labeled data (81% precision, 79% recall). This SMS scanner is a rule-based "
        "keyword/pattern matcher — a lightweight proof-of-concept, not a benchmarked ML classifier."
    )

# ==========================================================
# MODE 4: Receipt Screenshot Check (Beta)
# ==========================================================
else:
    st.subheader("📷 Payment Receipt Fraud Check")
    st.caption("Upload a PhonePe, Paytm, Google Pay or UPI receipt screenshot and check for suspicious patterns.")

    st.markdown(
        "<span style='background-color:#FFF3CD; padding:4px 10px; border-radius:6px; font-size:13px;'>"
        "🧪 BETA — Rule-based OCR scanner, separate from the trained XGBoost model</span>",
        unsafe_allow_html=True
    )
    st.info(
        "This is a beta rule-based OCR scanner for educational/project purposes. "
        "It is **not** a production banking fraud detection system, and it does **not** connect to "
        "PhonePe, Paytm, Google Pay, or any real banking system in any way."
    )
    st.caption(
        "This scanner is a separate pipeline from the trained XGBoost transaction classifier used in "
        "Modes 1 and 2 — it does not use machine learning and does not feed image/OCR data into that model. "
        "OCR and heuristic results may contain false positives or false negatives."
    )

    if not OCR_AVAILABLE:
        st.error(
            "⚠️ OCR is not available in this environment — the `pytesseract` / `Pillow` packages could not be "
            "imported. See the deployment notes for this app (requirements.txt and packages.txt) to enable this feature."
        )
    else:
        uploaded_image = st.file_uploader("Upload a receipt screenshot (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

        if uploaded_image is not None:
            try:
                pil_img = Image.open(uploaded_image).convert("RGB")
            except Exception:
                st.error("⚠️ This file couldn't be opened as an image. Please upload a valid PNG or JPG screenshot.")
                st.stop()

            st.image(pil_img, caption="Uploaded Receipt", use_container_width=True)

            with st.spinner("Extracting text from the image..."):
                ocr_text, ocr_error = extract_text_from_image(pil_img)

            if ocr_error == "tesseract_not_found":
                st.error(
                    "⚠️ The Tesseract OCR engine was not found on this server. "
                    "This usually means the deployment is missing the `tesseract-ocr` system package "
                    "(see deployment notes — `packages.txt`)."
                )
            elif ocr_error is not None:
                st.error(f"⚠️ OCR failed on this image: {ocr_error.replace('ocr_error:', '')}")
            elif not ocr_text:
                st.warning(
                    "No readable text was detected in this image. Try a clearer, higher-resolution "
                    "screenshot with visible text."
                )
            else:
                with st.expander("📄 Extracted OCR Text"):
                    st.text(ocr_text)

                st.markdown("**Extracted Transaction Details**")
                st.caption("Receipt information is extracted from the uploaded image using OCR and may contain recognition errors.")

                fields = extract_receipt_fields(ocr_text)
                fcol1, fcol2, fcol3 = st.columns(3)
                fcol1.metric("Amount", fields['amount'])
                fcol2.metric("Date", fields['date'])
                fcol3.metric("Time", fields['time'])

                fcol4, fcol5 = st.columns(2)
                fcol4.write(f"**Payment App Mentioned:** {fields['payment_app']}")
                fcol5.write(f"**Transaction / Ref ID:** {fields['txn_id']}")
                st.write(f"**Merchant / Receiver:** {fields['merchant']}")

                st.divider()
                if st.button("🔍 Scan Receipt for Fraud Patterns", type="primary"):
                    score, matched = scan_text_for_fraud_patterns(ocr_text)
                    render_pattern_scan_result(score, matched, ocr_text)

    st.divider()
    st.caption(
        "Disclaimer: This tool does not claim bank-level fraud detection, production-grade security, "
        "guaranteed fraud detection, or actual integration with PhonePe/Paytm/Google Pay/any banking system. "
        "It is a rule-based educational prototype only."
    )

# ==========================================================
# Model comparison table (always visible)
# ==========================================================
st.divider()
st.subheader("Model Performance Summary")
perf_df = pd.DataFrame({
    'Model': ['Logistic Regression (baseline)', 'LogReg + SMOTE', 'Random Forest + SMOTE', 'XGBoost + SMOTE (final)'],
    'Precision': [0.85, 0.05, 0.55, 0.81],
    'Recall': [0.59, 0.87, 0.81, 0.79],
    'PR-AUC': [0.692, 0.715, 0.780, 0.817]
})
st.dataframe(perf_df, use_container_width=True)
