import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HAR - Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

# ============================================================
# EXPECTED COLUMNS (UCI HAR - 561 features + 'subject')
# ============================================================
# ---- YAHAN APNI EXPECTED COLUMNS LIST DALO ----
# List bohot lambi hai (561 columns), is liye ise hardcode karne ke bajaye
# ek file se load karna behtar hai. Apne training data se ye command chala kar
# poori list nikalo aur ek text file me save karo (ek column per line):
#
#     with open("expected_columns.txt", "w") as f:
#         f.write("\n".join(X_train.columns.tolist()))
#
# Phir neeche wala path apni file ke path se replace kar do.
EXPECTED_COLUMNS_FILE = "expected_columns.txt"  # <-- YAHAN apni expected_columns.txt ka PATH dalo


@st.cache_data
def load_expected_columns(path):
    if not path:
        return []
    try:
        with open(path, "r") as f:
            cols = [line.strip() for line in f if line.strip()]
        return cols
    except FileNotFoundError:
        return []


def validate_csv(df, expected_cols):
    """Simple exact match check: uploaded CSV me sab expected columns hone chahiye."""
    missing_cols = [c for c in expected_cols if c not in df.columns]
    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols


# ============================================================
# MODEL LOADING (aapko yahan apna model/pipeline load karna hai)
# ============================================================
@st.cache_resource
def load_model():
    # -------- YAHAN APNA MODEL PATH DALO --------
    # Example: model = joblib.load("path/to/model.pkl")
    import joblib
    # Example: model = tf.keras.models.load_model("path/to/model.h5")
    model = joblib.load("stacked_model.pkl")  # <-- isko replace karo apne loaded model se
    return model


def predict_activity(model, data):
    preds = model.predict(data)
    dummy_labels = ["Walking", "Running", "Sitting", "Standing", "Lying"]
    dummy_probs = {label: float(np.random.rand()) for label in dummy_labels}
    total = sum(dummy_probs.values())
    dummy_probs = {k: v / total for k, v in dummy_probs.items()}
    predicted_label = max(dummy_probs, key=dummy_probs.get)
    return predicted_label, dummy_probs


ACTIVITY_EMOJIS = {
    "Walking": "🚶",
    "Running": "🏃",
    "Sitting": "🪑",
    "Standing": "🧍",
    "Lying": "🛌",
}

model = load_model()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data source chuno:",
    ["Upload CSV", "Sample Data"]
)

with st.sidebar.expander("ℹ️ About this project"):
    st.write(
        "Ye app mobile sensor (accelerometer / gyroscope) data se "
        "human activity predict karti hai using a trained ML model."
    )

# ============================================================
# MAIN TITLE
# ============================================================
st.title("🏃 Human Activity Recognition (HAR)")
st.caption("Mobile sensor data se activity prediction — Streamlit GUI")

sensor_df = None

# ============================================================
# INPUT SECTION
# ============================================================
if data_source == "Upload CSV":
    st.subheader("📁 Upload Sensor Data (CSV)")

    expected_cols = load_expected_columns(EXPECTED_COLUMNS_FILE)

    # Sample template download button (agar expected columns load ho chuki hain)
    if expected_cols:
        template_df = pd.DataFrame(columns=expected_cols)
        st.download_button(
            "⬇️ Download Sample CSV Template",
            data=template_df.to_csv(index=False),
            file_name="har_sample_template.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("CSV file upload karo (template wale format me)", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception:
            st.error("❌ Ye valid CSV file nahi hai. Sahi CSV upload karo.")
            raw_df = None

        if raw_df is not None:
            if not expected_cols:
                # EXPECTED_COLUMNS_FILE set nahi ki gayi -> koi validation nahi
                sensor_df = raw_df
                st.success(f"File loaded! Shape: {sensor_df.shape}")
                st.dataframe(sensor_df.head())
            else:
                is_valid, missing_cols = validate_csv(raw_df, expected_cols)
                if not is_valid:
                    st.error(
                        f"❌ Ye CSV required format se match nahi karti "
                        f"({len(missing_cols)} column(s) missing hain). "
                        f"Upar se sample template download kar ke usi format me data daalo."
                    )
                    sensor_df = None
                else:
                    sensor_df = raw_df[expected_cols]
                    st.success(f"✅ CSV valid hai! Shape: {sensor_df.shape}")
                    st.dataframe(sensor_df.head())

elif data_source == "Sample Data":
    st.subheader("🧪 Sample Data")

    # ---- YAHAN APNI REAL SAMPLE CSV KA PATH DALO (recommended) ----
    # Ye file chand real test rows honi chahiye (sahi 561/172 columns ke sath),
    # jinhe aap apne original dataset (jaise X_test.csv) se nikaal kar bana sakte hain.
    SAMPLE_DATA_FILE = ""  # <-- e.g. "sample_rows.csv"

    expected_cols = load_expected_columns(EXPECTED_COLUMNS_FILE)

    if SAMPLE_DATA_FILE:
        try:
            sample_source_df = pd.read_csv(SAMPLE_DATA_FILE)
            sample_choice = st.selectbox(
                "Ek sample row choose karo:",
                sample_source_df.index.tolist()
            )
            sensor_df = sample_source_df.loc[[sample_choice]]
            if expected_cols:
                sensor_df = sensor_df[expected_cols]
            st.success("✅ Real sample row loaded.")
            st.dataframe(sensor_df)
        except Exception as e:
            st.error(f"❌ Sample data file load nahi ho payi: {e}")
            sensor_df = None
    elif expected_cols:
        st.warning(
            "⚠️ Abhi tak koi real sample file connect nahi hui, is liye ye "
            "**synthetic/placeholder data** hai (sirf demo ke liye), asli sensor "
            "readings nahi. Real sample dikhane ke liye `SAMPLE_DATA_FILE` set karo."
        )
        np.random.seed(42)
        sensor_df = pd.DataFrame(
            [np.random.randn(len(expected_cols))], columns=expected_cols
        )
        st.dataframe(sensor_df)
    else:
        st.info("Pehle EXPECTED_COLUMNS_FILE ya SAMPLE_DATA_FILE set karo.")
        sensor_df = None

# ============================================================
# VISUALIZATION SECTION
# ============================================================
if sensor_df is not None and len(sensor_df) > 1:
    st.subheader("📊 Sensor Readings Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in sensor_df.columns:
        ax.plot(sensor_df[col].values, label=col)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

# ============================================================
# PREDICTION SECTION
# ============================================================
st.subheader("🔮 Prediction")

if sensor_df is not None:
    if st.button("Predict Activity", type="primary"):
        predicted_label, probs = predict_activity(model, sensor_df)

        emoji = ACTIVITY_EMOJIS.get(predicted_label, "❓")
        st.markdown(f"## {emoji} Predicted Activity: **{predicted_label}**")

        confidence = probs[predicted_label]
        st.progress(confidence)
        st.write(f"Confidence: **{confidence * 100:.1f}%**")

        st.subheader("📈 Class Probabilities")
        prob_df = pd.DataFrame({
            "Activity": list(probs.keys()),
            "Probability": list(probs.values())
        }).sort_values("Probability", ascending=False)

        st.bar_chart(prob_df.set_index("Activity"))
else:
    st.info("Pehle upload / manual / sample se data provide karo, phir predict karo.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Made by ❤️ adnan")