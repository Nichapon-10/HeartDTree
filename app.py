"""
app.py
-------------------------------------------
Streamlit Web Application
สำหรับทำนายความเสี่ยงโรคหัวใจ
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Load Model
# ==========================================
@st.cache_resource
def load_model():
    """โหลดโมเดลและ metadata"""
    model = joblib.load("models/heart_model.pkl")
    metadata = joblib.load("models/metadata.pkl")
    return model, metadata

try:
    model, metadata = load_model()
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์โมเดล! กรุณารัน training.py ก่อน")
    st.stop()

# ==========================================
# Header
# ==========================================
st.markdown('<h1 class="main-header">🫀 Heart Disease Prediction System</h1>', 
            unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 1.1rem; color: #666;'>
    ระบบทำนายความเสี่ยงโรคหัวใจด้วย <strong>Decision Tree Classifier</strong><br>
    พัฒนาโดย: AI Research Team
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# Sidebar - Input Form
# ==========================================
st.sidebar.header("📝 Input Patient Data")
st.sidebar.markdown("กรุณากรอกข้อมูลผู้ป่วยด้านล่าง")

# ดึงช่วงค่าจาก metadata
feature_ranges = metadata["feature_ranges"]
categorical_values = metadata["categorical_values"]

with st.sidebar.form("input_form"):
    st.subheader("ข้อมูลพื้นฐาน")
    
    # Age
    age = st.number_input(
        "🎂 Age (อายุ)", 
        min_value=feature_ranges["Age"][0], 
        max_value=feature_ranges["Age"][1], 
        value=55,
        help="อายุของผู้ป่วย (ปี)"
    )
    
    # Sex
    sex = st.selectbox(
        "⚧ Sex (เพศ)", 
        categorical_values["Sex"],
        format_func=lambda x: "Male (ชาย)" if x == 1 else "Female (หญิง)",
        help="1 = ชาย, 0 = หญิง"
    )
    
    st.subheader("ข้อมูลทางคลินิก")
    
    # Chest Pain Type
    cp = st.selectbox(
        "💔 Chest Pain Type (ชนิดอาการเจ็บหน้าอก)", 
        categorical_values["ChestPainType"],
        format_func=lambda x: {
            1: "Typical Angina",
            2: "Atypical Angina", 
            3: "Non-Anginal Pain",
            4: "Asymptomatic"
        }[x],
        help="ชนิดของอาการเจ็บหน้าอก"
    )
    
    # Resting BP
    trestbps = st.number_input(
        "🩺 Resting BP (ความดันโลหิตขณะพัก)", 
        min_value=feature_ranges["RestingBP"][0], 
        max_value=feature_ranges["RestingBP"][1], 
        value=130,
        help="ความดันโลหิตขณะพัก (mm Hg)"
    )
    
    # Cholesterol
    chol = st.number_input(
        "🧪 Cholesterol (คอเลสเตอรอล)", 
        min_value=feature_ranges["Cholesterol"][0], 
        max_value=feature_ranges["Cholesterol"][1], 
        value=200,
        help="ระดับคอเลสเตอรอลใน血清 (mg/dl)"
    )
    
    # Fasting Blood Sugar
    fbs = st.selectbox(
        "🍬 Fasting BS > 120 (น้ำตาลในเลือดขณะอดอาหาร)", 
        categorical_values["FastingBS"],
        format_func=lambda x: "Yes (> 120 mg/dl)" if x == 1 else "No (≤ 120 mg/dl)",
        help="ระดับน้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl"
    )
    
    # Resting ECG
    restecg = st.selectbox(
        "📈 Resting ECG", 
        categorical_values["RestingECG"],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy",
            3: "Probable or definite left ventricular hypertrophy"
        }[x],
        help="ผลการตรวจคลื่นไฟฟ้าหัวใจขณะพัก"
    )
    
    st.subheader("ข้อมูลการออกกำลังกาย")
    
    # Max Heart Rate
    thalach = st.number_input(
        "❤️ Max Heart Rate (อัตราการเต้นหัวใจสูงสุด)", 
        min_value=feature_ranges["MaxHR"][0], 
        max_value=feature_ranges["MaxHR"][1], 
        value=150,
        help="อัตราการเต้นหัวใจสูงสุดที่บรรลุได้ (bpm)"
    )
    
    # Exercise Angina
    exang = st.selectbox(
        "🏃 Exercise Induced Angina", 
        categorical_values["ExerciseAngina"],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="มีอาการเจ็บหน้าอกจากการออกกำลังกายหรือไม่"
    )
    
    # Oldpeak
    oldpeak = st.number_input(
        "📉 ST Depression (Oldpeak)", 
        min_value=feature_ranges["Oldpeak"][0], 
        max_value=feature_ranges["Oldpeak"][1], 
        value=1.0, 
        step=0.1,
        help="ST depression induced by exercise relative to rest"
    )
    
    # ST Slope
    slope = st.selectbox(
        "📊 Slope of ST Segment", 
        categorical_values["ST_Slope"],
        format_func=lambda x: {
            1: "Upsloping",
            2: "Flat",
            3: "Downsloping"
        }[x],
        help="ความชันของ ST segment"
    )
    
    submitted = st.form_submit_button("🔍 Predict Heart Disease", use_container_width=True)

# ==========================================
# Main Content
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    if submitted:
        # สร้าง DataFrame จาก input
        input_data = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": cp,
            "RestingBP": trestbps,
            "Cholesterol": chol,
            "FastingBS": fbs,
            "RestingECG": restecg,
            "MaxHR": thalach,
            "ExerciseAngina": exang,
            "Oldpeak": oldpeak,
            "ST_Slope": slope
        }])
        
        # ทำนาย
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]
        
        # แสดงผล
        st.subheader("📊 ผลการทำนาย")
        
        # Result Card
        if prediction[0] == 1:
            st.error("⚠️ **มีความเสี่ยงสูง** ที่จะเกิดโรคหัวใจ")
            st.markdown(f"""
            <div style='background-color: #ffe6e6; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #ff4b4b;'>
                <h3 style='color: #ff4b4b; margin-top: 0;'>คำเตือน</h3>
                <p>ผู้ป่วยรายนี้มีความเสี่ยงสูงที่จะเป็นโรคหัวใจ 
                ควรปรึกษาแพทย์และตรวจสุขภาพเพิ่มเติม</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ **ความเสี่ยงต่ำ** ไม่พบสัญญาณของโรคหัวใจ")
            st.markdown(f"""
            <div style='background-color: #e6ffe6; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #00c853;'>
                <h3 style='color: #00c853; margin-top: 0;'>ผลการตรวจ</h3>
                <p>ผู้ป่วยรายนี้มีความเสี่ยงต่ำที่จะเป็นโรคหัวใจ 
                แต่ควรดูแลสุขภาพและตรวจสุขภาพเป็นประจำ</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability Metrics
        st.subheader("📈 ความน่าจะเป็น")
        
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            st.metric(
                "ความเสี่ยงโรคหัวใจ", 
                f"{probability[1]*100:.2f}%",
                delta=None
            )
        
        with col_prob2:
            st.metric(
                "ไม่มีความเสี่ยง", 
                f"{probability[0]*100:.2f}%",
                delta=None
            )
        
        # Probability Bar
        st.progress(float(probability[1]))
        
        # Risk Level
        if probability[1] < 0.3:
            risk_level = "🟢 ต่ำ"
            risk_color = "green"
        elif probability[1] < 0.7:
            risk_level = "🟡 ปานกลาง"
            risk_color = "orange"
        else:
            risk_level = "🔴 สูง"
            risk_color = "red"
        
        st.markdown(f"### ระดับความเสี่ยง: <span style='color:{risk_color}'>{risk_level}</span>", 
                   unsafe_allow_html=True)
        
        # Input Data Summary
        with st.expander("📋 ดูข้อมูลที่คุณกรอก"):
            st.dataframe(input_data.T.rename(columns={0: "Value"}))

with col2:
    st.subheader("ℹ️ เกี่ยวกับระบบ")
    st.info("""
    **โมเดลที่ใช้:**
    - Decision Tree Classifier
    - Hyperparameter Tuning ด้วย GridSearchCV
    - Cross-Validation 5-fold
    
    **ฟีเจอร์ที่สำคัญ:**
    - Age (อายุ)
    - ChestPainType (ชนิดอาการเจ็บหน้าอก)
    - MaxHR (อัตราการเต้นหัวใจสูงสุด)
    - Oldpeak (ST Depression)
    """)
    
    st.warning("""
    **⚠️ คำเตือน:**
    ระบบนี้เป็นเครื่องมือช่วยในการตัดสินใจเท่านั้น 
    ไม่ใช่การวินิจฉัยทางการแพทย์ 
    กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญเสมอ
    """)

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>© 2026 Heart Disease Prediction System | Powered by Machine Learning</p>
    <p>Developed with ❤️ using Python, Scikit-learn, and Streamlit</p>
</div>
""", unsafe_allow_html=True)