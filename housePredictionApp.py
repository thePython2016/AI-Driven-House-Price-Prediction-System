import pandas as pd
import pickle as pkl
import streamlit as st
import numpy as np

# --- 1. LOAD EXTERNAL CSS ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("style.css")

# --- 2. LOAD MODELS ---
model = pkl.load(open("model.pkl", "rb"))
Transformer = pkl.load(open("Transformer.pkl", "rb"))
columns = pkl.load(open("columns.pkl", "rb"))

# --- 3. UI INITIALIZATION ---
st.title("House Prediction App")
st.subheader("Upload a data file to predict house price")

if "predicted_file" not in st.session_state: st.session_state.predicted_file = None
if "show_success"   not in st.session_state: st.session_state.show_success   = False

# ── Dynamic Card Content ──────────────────────────────────────────────────────
if "house_uploader" in st.session_state and st.session_state.house_uploader is not None:
    u_file = st.session_state.house_uploader
    kb = round(u_file.size / 1024, 1)
    size_info = f"{kb} KB" if kb < 1024 else f"{round(kb/1024,2)} MB"
    
    card_html = f"""
        <div class="upload-card" id="ucard">
            <div class="upload-icon">📄</div>
            <div class="upload-title">{u_file.name}</div>
            <div class="upload-sub">{size_info} • Ready to Predict</div>
        </div>
    """
else:
    card_html = """
        <div class="upload-card" id="ucard">
            <div class="upload-icon">☁️</div>
            <div class="upload-title">Choose a file or drag &amp; drop it here</div>
            <div class="upload-sub">CSV format · up to 50 MB</div>
        </div>
    """

st.markdown(card_html, unsafe_allow_html=True)

fileUpload = st.file_uploader(
    "Upload CSV",
    type="csv",
    key="house_uploader",
    label_visibility="collapsed",
)

st.markdown("""
    <script>
    (function() {
        function init() {
            const card = document.getElementById('ucard');
            const zone = document.querySelector('[data-testid="stFileUploadDropzone"]');
            if (!card || !zone) { setTimeout(init, 150); return; }
            zone.addEventListener('dragenter', () => card.classList.add('active'));
            zone.addEventListener('dragover',  (e) => { e.preventDefault(); card.classList.add('active'); });
            zone.addEventListener('dragleave', () => card.classList.remove('active'));
            zone.addEventListener('drop',      () => setTimeout(() => card.classList.remove('active'), 250));
        }
        init();
    })();
    </script>
    <div style="margin-bottom: 16px;"></div>
""", unsafe_allow_html=True)

# --- 4. PREDICTION LOGIC & UI ACTIONS ---
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict Score", key="predict_btn", use_container_width=True):
        if fileUpload is None:
            st.error("Please upload a CSV file first.")
        else:
            try:
                file = pd.read_csv(fileUpload)
                Transform = Transformer.transform(file)
                
                if not isinstance(Transform, pd.DataFrame):
                    try:    f_names = Transformer.get_feature_names_out()
                    except: f_names = columns
                    Transform = pd.DataFrame(Transform, columns=f_names)
                
                if not set(columns).issubset(Transform.columns):
                    st.error("Uploaded file is missing required features.")
                else:
                    preds = np.expm1(model.predict(Transform[columns]))
                    file["Predicted Price"] = preds
                    st.session_state.predicted_file = file
                    st.session_state.show_success = True
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {e}")

with col2:
    if st.button("Clear All", key="clear_all", use_container_width=True):
        st.session_state.predicted_file = None
        st.session_state.show_success = False
        if "house_uploader" in st.session_state:
            del st.session_state["house_uploader"]
        st.rerun()

# --- 5. RESULTS ---
if st.session_state.predicted_file is not None:
    st.divider()
    with st.expander("📊 Prediction Table", expanded=True):
        if st.session_state.show_success:
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown('<div class="success-badge">✅ Predictions completed successfully!</div>', unsafe_allow_html=True)
            with c2:
                if st.button("✖", key="hide_table"):
                    st.session_state.show_success = False
                    st.rerun()
        
        st.dataframe(st.session_state.predicted_file, use_container_width=True)