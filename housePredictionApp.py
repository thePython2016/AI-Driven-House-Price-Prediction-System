import pandas as pd
import pickle as pkl
import streamlit as st
import numpy as np


# Model and Transformer Import
model = pkl.load(open("model.pkl", "rb"))
Transformer = pkl.load(open("Transformer.pkl", "rb"))
columns = pkl.load(open("columns.pkl", "rb"))

st.title("House Prediction App")

st.subheader("House Prediction Model")
fileUpload = st.file_uploader("Upload File", type="csv", key="house_file_uploader")
button = st.button("Predict Score", key="house_predict_button")

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
                Prediction = model.predict(Transform[columns])
                file["Predicted Price"] = Prediction
                st.session_state.show_success = True
                st.session_state.predicted_file = file

        except Exception as e:
            st.error(f"Error during processing: {e}")

if st.session_state.predicted_file is not None:
    if st.session_state.show_success:
        st.markdown("""
            <style>
            .success-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background-color: #d4edda;
                color: #155724;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 14px;
                margin-bottom: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.markdown('<div class="success-badge">Predictions done!</div>', unsafe_allow_html=True)
        with col2:
            if st.button("✖", key="dismiss_success", help="Dismiss"):
                st.session_state.show_success = False
                st.rerun()
    st.dataframe(st.session_state.predicted_file)