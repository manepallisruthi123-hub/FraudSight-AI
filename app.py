import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

st.set_page_config(page_title="FraudSight AI", layout="wide")

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
# Helper functions
# ==========================================================
def get_risk_level(prob):
    if prob >= threshold:
        return "🔴 High Risk"
    elif prob >= threshold * 0.4:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"


def validate_columns(df):
    df_cols = set(df.columns) - {'Class'}
    expected_cols = set(EXPECTED_COLUMNS)
    missing = expected_cols - df_cols
    extra = df_cols - expected_cols
    return missing, extra


def generate_explanation(shap_values_row, feature_names, is_flagged):
    """Convert SHAP values into a plain-English explanation."""
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
    """Consistent anonymization disclaimer, used wherever SHAP explanations are shown."""
    st.caption(
        "Note: The dataset anonymizes transaction attributes for privacy. "
        "Features V1–V28 are transformed variables rather than real-world "
        "labels such as merchant, device, or location."
    )


def show_single_alert(proba):
    """Real-time style alert banner + toast notification for single transaction mode."""
    if proba >= threshold:
        st.error(f"🚨 **FLAGGED — HIGH RISK** — Transaction flagged at {proba:.1%} risk. Immediate review recommended.")
        st.toast("🚨 High-risk transaction flagged!", icon="🚨")
    elif proba >= threshold * 0.4:
        st.warning(f"⚠️ Medium risk transaction ({proba:.1%}) — monitor for confirmation.")
        st.toast("⚠️ Medium-risk transaction flagged", icon="⚠️")
    else:
        st.toast("✅ Transaction looks legitimate", icon="✅")


def show_batch_alert(fraud_count, highest_risk):
    """Prominent alert banner for batch CSV analysis."""
    if fraud_count > 0:
        st.markdown(
            f"""
            <div style="background-color:#ffe6e6; padding:20px; border-radius:10px; border-left:6px solid #ff4b4b;">
                <h2 style="color:#cc0000; margin-top:0;">🚨 HIGH RISK ALERT</h2>
                <p style="font-size:18px; margin-bottom:5px;">
                    <b>{fraud_count}</b> suspicious transaction(s) flagged.
                </p>
                <p style="font-size:18px; margin-bottom:5px;">
                    Highest Risk Score: <b>{highest_risk:.0%}</b>
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
            """
            <div style="background-color:#e6ffed; padding:20px; border-radius:10px; border-left:6px solid #2ecc71;">
                <h2 style="color:#1e8449; margin-top:0;">✅ No Suspicious Transactions Detected</h2>
                <p style="font-size:16px; color:#555;">
                    All uploaded transactions appear to be low risk.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.toast("✅ No suspicious transactions flagged in this batch", icon="✅")

# ==========================================================
# Header
# ==========================================================
st.title("🛡️ FraudSight AI")
st.caption("Explainable real-time fraud detection using XGBoost + SHAP")

st.sidebar.header("Model Info")
st.sidebar.metric("Precision", "81%")
st.sidebar.metric("Recall", "79%")
st.sidebar.metric("PR-AUC", "0.817")
st.sidebar.write(f"Decision threshold: {threshold:.4f}")
st.sidebar.info("Trained on 283,726 transactions (0.17% fraud rate) using SMOTE-resampled XGBoost.")

with st.sidebar.expander("📋 Required CSV columns"):
    st.code(", ".join(EXPECTED_COLUMNS))

st.divider()
mode = st.radio("Choose mode:", ["Single Transaction (sample data)", "Upload CSV (batch analysis)"], horizontal=True)

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
        risk_label = get_risk_level(proba)
        prediction = "🚨 Flagged (High Risk)" if proba >= threshold else "✅ Legitimate"
        is_flagged = proba >= threshold

        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{proba:.4f}")
        col2.metric("Prediction", prediction)
        col3.markdown(f"### {risk_label}")

        st.progress(min(float(proba), 1.0))

        # --- Real-time alert ---
        show_single_alert(proba)

        # --- SHAP explanation ---
        st.subheader("Why did the model make this decision?")
        shap_values = explainer.shap_values(transaction)

        explanation_text = generate_explanation(shap_values[0], transaction.columns.tolist(), is_flagged)
        st.info(f"**Reason for Prediction:**\n\n{explanation_text}")

        shap_explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=transaction.iloc[0].values,
            feature_names=transaction.columns.tolist()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation, show=False)
        st.pyplot(fig)

        show_feature_note()

# ==========================================================
# MODE 2: CSV upload (batch analysis)
# ==========================================================
else:
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

        # --- Fraud Summary metrics ---
        st.divider()
        st.subheader("📊 Fraud Summary")
        total = len(results_df)
        fraud_count = int(predictions.sum())
        legit_count = total - fraud_count
        fraud_rate = (fraud_count / total) * 100
        highest_risk = results_df['Fraud_Probability'].max()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total:,}")
        col2.metric("🚨 Transactions Flagged", f"{fraud_count:,}")
        col3.metric("✅ Legitimate", f"{legit_count:,}")
        col4.metric("⚠️ Fraud Rate", f"{fraud_rate:.2f}%")

        # --- Prominent alert banner ---
        show_batch_alert(fraud_count, highest_risk)

        # --- Fraud Trends Dashboard ---
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

        # --- Results table ---
        st.divider()
        st.subheader("Transaction Results")
        show_only_fraud = st.checkbox("Show only flagged transactions")
        display_df = results_df[results_df['Prediction'] == '🚨 Flagged'] if show_only_fraud else results_df

        display_cols = ['Transaction_ID', 'Amount', 'Fraud_Probability', 'Risk_Level', 'Prediction']

        MAX_STYLED_ROWS = 5000
        if len(display_df) <= MAX_STYLED_ROWS:
            st.dataframe(
                display_df[display_cols].style.apply(
                    lambda row: ['background-color: #ffe6e6' if row['Prediction'] == '🚨 Flagged' else '' for _ in row],
                    axis=1
                ),
                use_container_width=True
            )
        else:
            st.info(f"Showing {len(display_df):,} rows — highlighting disabled for large tables (performance).")
            st.dataframe(display_df[display_cols], use_container_width=True)

        # --- Explain a specific flagged transaction ---
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
                values=shap_values_single[0],
                base_values=explainer.expected_value,
                data=selected_row.iloc[0].values,
                feature_names=selected_row.columns.tolist()
            )
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation, show=False)
            st.pyplot(fig4)

            show_feature_note()

        # --- Download report ---
        st.download_button(
            "⬇️ Download Prediction Report (CSV)",
            data=display_df[display_cols].to_csv(index=False),
            file_name="fraud_prediction_report.csv",
            mime="text/csv"
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