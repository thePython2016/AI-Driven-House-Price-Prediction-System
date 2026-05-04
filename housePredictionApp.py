import pandas as pd
import pickle as pkl
import streamlit as st
import numpy as np

# Model and Transformer Import
model = pkl.load(open("model.pkl", "rb"))
Transformer = pkl.load(open("Transformer.pkl", "rb"))
columns = pkl.load(open("columns.pkl", "rb"))

# --- CUSTOM CSS FOR FONT SIZES & DARK MODE ---
st.markdown("""
    <style>
    /* Adjust Header Font Sizes */
    h1 {
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    h2 {
        font-size: 24px !important;
        font-weight: 600 !important;
    }
    h3 {
        font-size: 20px !important;
    }

    /* Adaptive Success Badge (Dark/Light Mode) */
    :root {
        --success-bg: #d4edda;
        --success-text: #155724;
        --success-border: #c3e6cb;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --success-bg: #1e4620;
            --success-text: #9fdf9f;
            --success-border: #2e7d32;
        }
    }

    .success-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: var(--success-bg);
        color: var(--success-text);
        border: 1px solid var(--success-border);
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 14px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("House Prediction App")
st.subheader("Upload a data File to Predict House Price")

fileUpload = st.file_uploader("Upload File", type="csv", key="house_file_uploader")
button = st.button("Predict Score", key="house_predict_button")

# Initialize session state
if "show_success" not in st.session_state:
    st.session_state.show_success = False
if "predicted_file" not in st.session_state:
    st.session_state.predicted_file = None

if button:
    if not fileUpload:
        st.error("Select File to Upload")
    else:
        file = pd.read_csv(fileUpload)
        try:
            # Transformation logic
            Transform = Transformer.transform(file)
            if not isinstance(Transform, pd.DataFrame):
                try:
                    feature_names = Transformer.get_feature_names_out()
                except AttributeError:
                    feature_names = columns
                Transform = pd.DataFrame(Transform, columns=feature_names)

            missing_cols = set(columns) - set(Transform.columns)
            if missing_cols:
                st.error(f"Transformation produced unexpected output. Missing: {missing_cols}")
            else:
                # Predicting
                Prediction = np.expm1(model.predict(Transform[columns]))
                # If price was log-transformed in training, use: Prediction = np.expm1(Prediction)
                file["Predicted Price"] = Prediction
                st.session_state.show_success = True
                st.session_state.predicted_file = file

        except Exception as e:
            st.error(f"Error during processing: {e}")

# Result Display Area
if st.session_state.predicted_file is not None:
    if st.session_state.show_success:
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.markdown('<div class="success-badge">✅ Predictions done!</div>', unsafe_allow_html=True)
        with col2:
            if st.button("✖", key="dismiss_success", help="Dismiss"):
                st.session_state.show_success = False
                st.rerun()
    
    st.dataframe(st.session_state.predicted_file, use_container_width=True)