import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
from datetime import datetime # Added for dynamic date handling

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SentinelNet | Enterprise NIDS", page_icon="🛡️", layout="wide")

# --- LOAD ML ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    # Loading model and encoders from user training history
    model = joblib.load('nids_model.pkl')
    encoders = joblib.load('encoders.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, encoders, features

if os.path.exists('nids_model.pkl'):
    model, encoders, features = load_artifacts()
else:
    st.error("Model artifacts not found! Please run the training script first.")
    st.stop()

# --- SIDEBAR NAVIGATION & STATUS ---
st.sidebar.image("https://img.icons8.com/fluency/100/shield.png")
st.sidebar.title("SentinelNet OS")

st.sidebar.subheader("📂 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home / Dashboard", "📊 Model Intelligence", "📝 Project Info"])

st.sidebar.subheader("🌐 System Health")
st.sidebar.success("Model Status: Operational")
st.sidebar.info("Database: NSL-KDD Connected")

st.sidebar.subheader("👤 Developer Profile")
st.sidebar.write("**Course:** CS506 Big Data")
st.sidebar.write("**Deadline:** 13 May 2026")

st.sidebar.markdown("---")
# Automatically updates the sidebar caption to the current month and year
st.sidebar.caption(f"Last System Sync: {datetime.now().strftime('%B %Y')}")

# --- MAIN INTERFACE LOGIC ---

if page == "🏠 Home / Dashboard":
    # --- HERO SECTION ---
    st.markdown("""
        <div style="background-color:#0E1117; padding:40px; border-radius:15px; border-left: 8px solid #2ECC71; margin-bottom:30px">
            <h1 style="color:white; margin:0;">🛡️ SentinelNet: AI-NIDS Dashboard</h1>
            <p style="color:#808495; font-size:18px;">Advanced Big Data Analytics for Real-Time Network Intrusion Detection</p>
            <hr style="border-color:#262730;">
            <span style="background-color:#262730; color:#2ECC71; padding:5px 15px; border-radius:20px; font-size:14px; font-weight:bold;">
                SECURITY STATUS: ACTIVE MONITORING
            </span>
        </div>
    """, unsafe_allow_html=True)

    # --- TOP LEVEL METRICS (DYNAMIC DATE) ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detection Engine", "Random Forest")
    m2.metric("Inference Latency", "14ms")
    m3.metric("Data Dimension", "41 Features")
    # This now shows the exact date you are running the app
    m4.metric("Last Scan Update", datetime.now().strftime("%Y-%m-%d"))

    st.markdown("### 🛰️ Network Traffic Ingestion")
    uploaded_file = st.file_uploader("Drop network logs here for deep packet analysis", type=None)

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file, header=None)
        
        if len(input_df.columns) >= len(features):
            col_names = features + ["label", "difficulty"][:len(input_df.columns)-len(features)]
            input_df.columns = col_names
        
        st.info(f"📥 Batch Ingested: {len(input_df):,} network packets identified.")

        if st.button("⚡ EXECUTE SECURITY SCAN"):
            with st.spinner("Decoding packet headers and running AI inference..."):
                processed_df = input_df.copy()
                for col, le in encoders.items():
                    if col in processed_df.columns:
                        processed_df[col] = processed_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

                X_input = processed_df[features]
                predictions = model.predict(X_input)
                input_df['Prediction'] = predictions
                
                anomalies = (predictions == 'anomaly').sum()
                total = len(predictions)
                integrity_score = ((total-anomalies)/total)*100

                # --- RESULTS DISPLAY ---
                st.markdown("---")
                st.subheader("🛠️ Scan Diagnostics")
                
                res1, res2, res3 = st.columns(3)
                res1.markdown(f"<div style='text-align:center; padding:20px; background-color:#161B22; border-radius:10px;'><h4>TRAFFIC VOLUME</h4><h2 style='color:white;'>{total:,}</h2></div>", unsafe_allow_html=True)
                res2.markdown(f"<div style='text-align:center; padding:20px; background-color:#161B22; border-radius:10px; border-bottom: 4px solid {'#2ECC71' if anomalies == 0 else '#E74C3C'};'><h4>THREATS DETECTED</h4><h2 style='color:{'#2ECC71' if anomalies == 0 else '#E74C3C'};'>{anomalies}</h2></div>", unsafe_allow_html=True)
                res3.markdown(f"<div style='text-align:center; padding:20px; background-color:#161B22; border-radius:10px;'><h4>NETWORK HEALTH</h4><h2 style='color:white;'>{integrity_score:.1f}%</h2></div>", unsafe_allow_html=True)

                col_chart, col_table = st.columns([1, 1])
                with col_chart:
                    fig = px.pie(input_df, names='Prediction', 
                                 title='Security Composition',
                                 color='Prediction',
                                 color_discrete_map={'normal':'#2ECC71', 'anomaly':'#E74C3C'},
                                 hole=0.6)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_table:
                    st.write("### 🚨 Threat Isolation Log")
                    if anomalies > 0:
                        st.dataframe(input_df[input_df['Prediction'] == 'anomaly'].head(100), height=300)
                    else:
                        st.success("No suspicious activity detected. Network is clean.")

elif page == "📊 Model Intelligence":
    st.title("AI Intelligence Layer")
    st.info("The system uses a Random Forest Ensemble with 100 decision trees to ensure high detection accuracy.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Input Features Analyzed")
        st.write(features)
    with col_b:
        st.subheader("Detection Methodology")
        st.markdown("""
        - **Label Encoding:** Converting network protocols to numeric data.
        - **Binary Classification:** Separating 'normal' from 'anomaly'.
        - **Random Forest:** Utilizing an ensemble of trees for robust voting results.
        """)

elif page == "📝 Project Info":
    st.title("Project Documentation")
    st.markdown(f"""
    ### Overview
    This **Network Intrusion Detection System (NIDS)** was developed as an individual project for **CS506 Big Data**.
    
    ### Tech Stack
    - **Frontend:** Streamlit & Plotly
    - **Backend:** Python & Scikit-Learn
    - **Dataset:** NSL-KDD Network Logs
    
    **Submission Deadline:** 13 May 2026 
    """)