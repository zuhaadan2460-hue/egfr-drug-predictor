import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EGFR Drug Predictor",
    page_icon="🧬",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #F5F8FF; }
    .stButton>button {
        background-color: #1565C0;
        color: white;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 16px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #0D47A1; }
    .result-active {
        background-color: #E8F5E9;
        border-left: 5px solid #2E7D32;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .result-inactive {
        background-color: #FFEBEE;
        border-left: 5px solid #C62828;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background: linear-gradient(135deg, #0A1628, #1565C0);
            padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 25px;'>
    <h1 style='color: white; margin: 0; font-size: 28px;'>🧬 EGFR Drug Predictor</h1>
    <p style='color: #90CAF9; margin: 8px 0 0 0; font-size: 15px;'>
        Machine Learning-Based Prediction of EGFR Inhibitors for Lung Cancer Treatment
    </p>
    <p style='color: #64B5F6; margin: 4px 0 0 0; font-size: 12px;'>
        University of Agriculture Faisalabad — Department of Computer Science
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('drug_target_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

try:
    model = load_model()
    generator = load_generator()
    model_loaded = True
except:
    model_loaded = False
    st.error("Model file not found! Make sure drug_target_model.pkl is in the same folder.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Model Accuracy", "80.25%")
with col2:
    st.metric("AUC Score", "0.8898")

st.markdown("---")

st.subheader("Test Your Drug")

drug_name = st.text_input(
    "Drug Name",
    placeholder="Example: Gefitinib"
)

smiles = st.text_area(
    "Drug SMILES String",
    placeholder="Paste SMILES here... Example: COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    height=100
)

st.markdown("---")
st.subheader("Quick Test Examples")

examples = {
    "Gefitinib (Active — FDA approved)":    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Erlotinib (Active — FDA approved)":    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "Aspirin (Inactive — pain killer)":     "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen (Inactive — pain killer)":   "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
}

selected = st.selectbox("Or select a known drug to test:", ["-- Select --"] + list(examples.keys()))

if selected != "-- Select --":
    smiles = examples[selected]
    drug_name = selected.split(" (")[0]
    st.info(f"SMILES loaded for {drug_name}")

if st.button("Predict EGFR Activity"):
    if not smiles.strip():
        st.warning("Please enter a SMILES string first!")
    elif not model_loaded:
        st.error("Model not loaded!")
    else:
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                st.error("Invalid SMILES string! Please check and try again.")
            else:
                fp = generator.GetFingerprintAsNumPy(mol)
                fp = fp.reshape(1, -1)
                prediction = model.predict(fp)[0]
                probability = model.predict_proba(fp)[0]
                active_prob  = round(probability[1] * 100, 2)
                inactive_prob = round(probability[0] * 100, 2)
                name = drug_name if drug_name else "Unknown Drug"

                if prediction == 1:
                    st.markdown(f"""
                    <div class='result-active'>
                        <h2 style='color:#2E7D32; margin:0'>✅ ACTIVE</h2>
                        <p style='font-size:16px; margin:8px 0;'><b>{name}</b> is predicted to <b>WORK</b> against EGFR!</p>
                        <p style='color:#555; margin:0'>This drug is likely to bind to the EGFR protein and inhibit lung cancer cell growth.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-inactive'>
                        <h2 style='color:#C62828; margin:0'>❌ INACTIVE</h2>
                        <p style='font-size:16px; margin:8px 0;'><b>{name}</b> is predicted to <b>NOT work</b> against EGFR!</p>
                        <p style='color:#555; margin:0'>This drug is unlikely to bind to the EGFR protein.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### Confidence Scores")
                st.progress(int(active_prob))
                st.write(f"Active Probability: **{active_prob}%**")
                st.progress(int(inactive_prob))
                st.write(f"Inactive Probability: **{inactive_prob}%**")

                st.markdown("---")
                st.caption(f"Model: Random Forest | Fingerprint: Morgan (radius=2, 2048 bits) | Accuracy: 80.25% | AUC: 0.8898 | Data: ChEMBL EGFR (CHEMBL203)")

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#90A4AE; font-size:12px;'>
    FYP Project — Machine Learning-Based Prediction of EGFR Inhibitors for Lung Cancer Treatment<br>
    University of Agriculture Faisalabad | Department of Computer Science | 2025
</div>
""", unsafe_allow_html=True)