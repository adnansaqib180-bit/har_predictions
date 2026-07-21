import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HAR - Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
)

EXPECTED_COLUMNS_FILE = "expected_columns.txt"


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
    missing_cols = [c for c in expected_cols if c not in df.columns]
    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_resource
def load_model():
    model = joblib.load("stacked_model.pkl")
    return model


def predict_activity(model, data):
    preds = model.predict(data)
    
    if hasattr(model, "predict_proba"):
        probs_matrix = model.predict_proba(data)
        classes = model.classes_
    else:
        probs_matrix = None
        classes = None
        
    return preds, probs_matrix, classes


ACTIVITY_EMOJIS = {
    "Walking": "🚶",
    "Running": "🏃",
    "Sitting": "🪑",
    "Standing": "🧍",
    "Lying": "🛌",
    "Walking_Upstairs": "🧗",
    "Walking_Downstairs": "📉",
}

model = load_model()

st.sidebar.title("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data source chuno:", ["Upload CSV", "Sample Data"]
)

with st.sidebar.expander("ℹ️ About this project"):
    st.write(
        "Ye app mobile sensor (accelerometer / gyroscope) data se "
        "human activity predict karti hai using a trained ML model."
    )

st.title("🏃 Human Activity Recognition (HAR)")
st.caption("Mobile sensor data se activity prediction — Streamlit GUI")

sensor_df = None

if data_source == "Upload CSV":
    st.subheader("📁 Upload Sensor Data (CSV)")

    expected_cols = load_expected_columns(EXPECTED_COLUMNS_FILE)

    if expected_cols:
        template_df = pd.DataFrame(columns=expected_cols)
        st.download_button(
            "⬇️ Download Sample CSV Template",
            data=template_df.to_csv(index=False),
            file_name="har_sample_template.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader(
        "CSV file upload karo (template wale format me)", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception:
            st.error("❌ Ye valid CSV file nahi hai. Sahi CSV upload karo.")
            raw_df = None

        if raw_df is not None:
            if not expected_cols:
                sensor_df = raw_df
                st.success(f"File loaded! Shape: {sensor_df.shape}")
                st.dataframe(sensor_df.head())
            else:
                is_valid, missing_cols = validate_csv(raw_df, expected_cols)
                if not is_valid:
                    st.error(
                        f"❌ Ye CSV required format se match nahi karti "
                        f"({len(missing_cols)} column(s) missing hain)."
                    )
                    sensor_df = None
                else:
                    sensor_df = raw_df[expected_cols]
                    st.success(f"✅ CSV valid hai! Shape: {sensor_df.shape}")
                    st.dataframe(sensor_df.head())

elif data_source == "Sample Data":
    st.subheader("🧪 Sample Data")

    SAMPLE_DATA_FILE = ""
    expected_cols = load_expected_columns(EXPECTED_COLUMNS_FILE)

    if SAMPLE_DATA_FILE:
        try:
            sample_source_df = pd.read_csv(SAMPLE_DATA_FILE)
            sample_choice = st.selectbox(
                "Ek sample row choose karo:", sample_source_df.index.tolist()
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
            "⚠️ Synthetic placeholder data showing for demo."
        )
        np.random.seed(42)
        sensor_df = pd.DataFrame(
            [np.random.randn(len(expected_cols))], columns=expected_cols
        )
        st.dataframe(sensor_df)
    else:
        st.info("Pehle EXPECTED_COLUMNS_FILE ya SAMPLE_DATA_FILE set karo.")
        sensor_df = None

st.subheader("🔮 Prediction")

if sensor_df is not None:
    if st.button("Predict Activity", type="primary"):
        preds, probs_matrix, classes = predict_activity(model, sensor_df)

        results_df = sensor_df.copy()
        results_df["Predicted_Activity"] = preds

        if probs_matrix is not None:
            confidences = np.max(probs_matrix, axis=1)
            results_df["Confidence_Score"] = confidences
        else:
            confidences = None

        first_pred = str(preds[0])
        emoji = ACTIVITY_EMOJIS.get(first_pred, "❓")

        if len(preds) == 1:
            st.markdown(f"## {emoji} Predicted Activity: **{first_pred}**")
            if confidences is not None:
                conf = confidences[0]
                st.progress(float(conf))
                st.write(f"Confidence: **{conf * 100:.1f}%**")

                st.subheader("📈 Class Probabilities")
                prob_dict = dict(zip(classes, probs_matrix[0]))
                prob_df = pd.DataFrame(
                    {"Activity": list(prob_dict.keys()), "Probability": list(prob_dict.values())}
                ).sort_values("Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Activity"))
        else:
            st.success(f"✅ Total {len(preds)} rows predict ho gayi hain!")
            st.dataframe(results_df[["Predicted_Activity"] + (["Confidence_Score"] if confidences is not None else [])].head(10))

        st.markdown("---")
        st.subheader("📥 Download Predictions")

        csv_data = convert_df_to_csv(results_df)

        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv_data,
            file_name="har_predictions_output.csv",
            mime="text/csv",
            type="secondary",
        )
else:
    st.info(
        "Pehle upload / manual / sample se data provide karo, phir predict karo."
    )

st.markdown("---")
st.caption("Made by ❤️ adnan")