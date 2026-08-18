import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnIQ • Customer Churn Intelligence Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Tailwind CSS, Fonts & Enterprise Glassmorphism Styling System
# ---------------------------------------------------------
st.markdown("""
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>

    <style>
    /* CSS Variables & Custom Theme System */
    :root {
        --bg-main: #090d16;
        --bg-card: rgba(17, 24, 39, 0.65);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(99, 102, 241, 0.4);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --accent-indigo: #6366f1;
        --accent-violet: #8b5cf6;
        --accent-emerald: #10b981;
        --accent-rose: #ef4444;
        --accent-amber: #f59e0b;
    }

    /* Global Body & Streamlit Wrapper Reset */
    .stApp {
        background-color: #070a12 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.10) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
        background-attachment: fixed;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #f8fafc;
    }

    /* Hide default Streamlit header bar and footer */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    footer {visibility: hidden;}

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(11, 17, 29, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px);
    }
    
    /* Hide Radio Button Circles / Dots in Sidebar */
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
    section[data-testid="stSidebar"] div[role="radiogroup"] label input,
    section[data-testid="stSidebar"] div[data-baseweb="radio"] > div:first-child {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 8px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        display: flex;
        align-items: center;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.88rem;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(99, 102, 241, 0.10);
        border-color: rgba(99, 102, 241, 0.3);
        color: #f8fafc;
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] .stRadio div[aria-checked="true"] label,
    section[data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.25) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.5) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
    }

    /* Input Fields Styling Override */
    div[data-baseweb="select"] > div,
    input[type="number"], input[type="text"] {
        background-color: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover,
    input[type="number"]:hover, input[type="text"]:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
    }
    div[data-baseweb="select"]:focus-within > div,
    input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
    }

    /* Slider Styling Override */
    .stSlider [data-baseweb="slider"] {
        padding-top: 10px;
    }
    
    /* Primary CTA Button Custom Styling */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        border-radius: 12px !important;
        padding: 0.75rem 2.2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4), 0 8px 10px -6px rgba(124, 58, 237, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.005) !important;
        box-shadow: 0 15px 30px -5px rgba(79, 70, 229, 0.5), 0 10px 12px -5px rgba(124, 58, 237, 0.4) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    .stButton>button:active {
        transform: translateY(0) scale(0.99) !important;
    }

    /* Glass Cards */
    .glass-card-container {
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .glass-card-container:hover {
        border-color: rgba(99, 102, 241, 0.25);
    }

    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
        animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SVG Icon Helper Library (Lucide Style)
# ---------------------------------------------------------
def get_svg_icon(name, color="#6366f1", size=20):
    icons = {
        "user": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>',
        "wifi": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.55a11 11 0 0 1 14.08 0"></path><path d="M1.42 9a16 16 0 0 1 21.16 0"></path><path d="M8.53 16.11a6 6 0 0 1 6.95 0"></path><line x1="12" y1="20" x2="12.01" y2="20"></line></svg>',
        "credit-card": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="4" width="22" height="16" rx="2" ry="2"></rect><line x1="1" y1="10" x2="23" y2="10"></line></svg>',
        "activity": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
        "shield-alert": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
        "shield-check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>',
        "sliders": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>',
        "zap": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>',
        "bar-chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>'
    }
    return icons.get(name, "")


# ---------------------------------------------------------
# Model Loading & Inference Helper
# ---------------------------------------------------------
@st.cache_resource
def load_model_pipeline():
    model_path = Path('models/churn_model.pkl')
    if model_path.exists():
        return joblib.load(model_path)
    return None

pipeline = load_model_pipeline()

def predict_customer(data_dict):
    """Encapsulates dataframe creation and prediction logic."""
    df_in = pd.DataFrame([data_dict])
    pred = pipeline.predict(df_in)[0]
    prob = pipeline.predict_proba(df_in)[0, 1]
    return pred, prob


# ---------------------------------------------------------
# Sidebar Navigation & Branding
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="flex items-center gap-3 px-2 py-4 mb-4 border-b border-slate-800/80">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                </svg>
            </div>
            <div>
                <div class="flex items-center gap-2">
                    <h1 class="text-lg font-extrabold text-white tracking-tight leading-none">ChurnIQ</h1>
                    <span class="px-2 py-0.5 text-[10px] font-bold bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 rounded-md uppercase tracking-wider">v2.4</span>
                </div>
                <p class="text-[11px] text-slate-400 font-medium mt-1">Predictive Intelligence Platform</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p class='text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-3 px-2'>Navigation</p>", unsafe_allow_html=True)
    
    nav_options = [
        "Single Customer Risk Analyzer",
        "What-If Retention Simulator"
    ]
    
    selected_page = st.radio(
        label="Select Page",
        options=nav_options,
        label_visibility="collapsed"
    )
    
    st.markdown("""
        <div class="mt-16 p-4 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <div class="flex items-center gap-2 text-xs font-semibold text-emerald-400 mb-1">
                <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                <span>Model Active</span>
            </div>
            <p class="text-xs text-slate-300 font-medium">Random Forest Classifier</p>
            <p class="text-[11px] text-slate-500 mt-1">Accuracy: 80.7% | ROC-AUC: 0.837</p>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session state initialization for customer profile
# ---------------------------------------------------------
if 'customer_profile' not in st.session_state:
    st.session_state.customer_profile = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 89.85,
        'TotalCharges': 1078.20
    }

if pipeline is None:
    st.error("Trained model pipeline not found at `models/churn_model.pkl`. Please execute `python src/train.py`.")
    st.stop()


# =========================================================
# PAGE 1: Single Customer Risk Analyzer
# =========================================================
if "Single Customer Risk Analyzer" in selected_page:
    # Hero Section
    st.markdown("""
        <div class="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900/90 to-indigo-950/40 border border-slate-800 p-6 md:p-8 mb-8 shadow-xl">
            <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative z-10">
                <div>
                    <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-semibold uppercase tracking-wider mb-3">
                        Real-Time Risk Calculator
                    </div>
                    <h1 class="text-2xl md:text-3xl font-extrabold text-slate-50 tracking-tight">Customer Demographic & Service Parameters</h1>
                    <p class="text-sm text-slate-400 mt-1 max-w-2xl leading-relaxed">Input customer attributes to calculate real-time churn probability and inspect key risk factors.</p>
                </div>
                <div class="hidden lg:flex items-center gap-3 bg-slate-950/60 px-4 py-3 rounded-xl border border-slate-800">
                    <div class="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
                    <div class="text-xs font-semibold text-slate-300">Live Scoring Ready</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3-Column Inputs Grid
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="flex items-center gap-2 mb-3 px-1">
                {get_svg_icon("user", "#6366f1", 18)}
                <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider">Demographics & Tenure</h3>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="glass-card-container">', unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if st.session_state.customer_profile['gender']=='Female' else 1)
            senior_citizen_str = st.selectbox("Senior Citizen Status", ["No", "Yes"], index=st.session_state.customer_profile['SeniorCitizen'])
            partner = st.selectbox("Partner Status", ["Yes", "No"], index=0 if st.session_state.customer_profile['Partner']=='Yes' else 1)
            dependents = st.selectbox("Dependents Status", ["No", "Yes"], index=0 if st.session_state.customer_profile['Dependents']=='No' else 1)
            tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=int(st.session_state.customer_profile['tenure']))
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="flex items-center gap-2 mb-3 px-1">
                {get_svg_icon("wifi", "#8b5cf6", 18)}
                <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider">Subscribed Services</h3>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="glass-card-container">', unsafe_allow_html=True)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if st.session_state.customer_profile['PhoneService']=='Yes' else 1)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"], index=0 if st.session_state.customer_profile['InternetService']=='Fiber optic' else (1 if st.session_state.customer_profile['InternetService']=='DSL' else 2))
            online_security = st.selectbox("Online Security Service", ["No", "Yes", "No internet service"], index=0 if st.session_state.customer_profile['OnlineSecurity']=='No' else 1)
            online_backup = st.selectbox("Online Backup Service", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support Subscription", ["No", "Yes", "No internet service"], index=0 if st.session_state.customer_profile['TechSupport']=='No' else 1)
            streaming_tv = st.selectbox("Streaming TV Service", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies Service", ["Yes", "No", "No internet service"])
            st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="flex items-center gap-2 mb-3 px-1">
                {get_svg_icon("credit-card", "#10b981", 18)}
                <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider">Account & Billing Terms</h3>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="glass-card-container">', unsafe_allow_html=True)
            contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"], index=0 if st.session_state.customer_profile['Contract']=='Month-to-month' else 1)
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], index=0 if st.session_state.customer_profile['PaymentMethod']=='Electronic check' else 2)
            monthly_charges = st.number_input("Monthly Charges (₹)", min_value=18.0, max_value=10000.0, value=float(st.session_state.customer_profile['MonthlyCharges']), step=50.0)
            calc_total = float(np.round(monthly_charges * tenure, 2))
            total_charges = st.number_input("Total Charges (₹)", min_value=18.0, max_value=500000.0, value=calc_total, step=500.0)
            st.markdown('</div>', unsafe_allow_html=True)

    # Store inputs in session state
    current_customer = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen_str == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    st.session_state.customer_profile = current_customer

    # Action Button Section
    st.markdown("<div class='my-8 flex justify-center'>", unsafe_allow_html=True)
    analyze_btn = st.button("ANALYZE CHURN RISK")
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Calculation
    pred_val, prob_val = predict_customer(current_customer)
    risk_percentage = prob_val * 100

    # Results Section
    st.markdown("<div class='mt-10 mb-6'><h2 class='text-lg font-bold text-white tracking-tight'>Diagnostic Results & Risk Analysis</h2></div>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1.2, 1.8])

    with res_col1:
        if risk_percentage > 50:
            badge_bg = "bg-rose-500/10 border-rose-500/30 text-rose-400"
            badge_text = "HIGH CHURN RISK"
            gauge_color = "#ef4444"
            desc_text = "Customer exhibits critical risk indicators. Immediate proactive retention offer recommended."
        elif risk_percentage > 30:
            badge_bg = "bg-amber-500/10 border-amber-500/30 text-amber-400"
            badge_text = "MEDIUM CHURN RISK"
            gauge_color = "#f59e0b"
            desc_text = "Account displays moderate risk factors. Recommend value-add service engagement."
        else:
            badge_bg = "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
            badge_text = "LOW CHURN RISK"
            gauge_color = "#10b981"
            desc_text = "Account displays strong loyalty stability profile with minimal cancellation risk."

        st.markdown(f"""
            <div class="glass-card-container text-center relative overflow-hidden animate-fade-in">
                <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full border {badge_bg} text-xs font-bold tracking-wider uppercase mb-4">
                    {badge_text}
                </div>
                <div class="flex justify-center items-center my-3">
                    <div style="position: relative; width: 190px; height: 190px; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
                        <svg style="width: 100%; height: 100%; transform: rotate(-90deg);" viewBox="0 0 36 36">
                            <path stroke="#1e293b" stroke-width="3" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                            <path stroke="{gauge_color}" stroke-dasharray="{risk_percentage}, 100" stroke-width="3.2" stroke-linecap="round" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                        </svg>
                        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; pointer-events: none;">
                            <span style="font-size: 2.1rem; font-weight: 800; color: #ffffff; line-height: 1; letter-spacing: -0.5px;">{risk_percentage:.1f}%</span>
                            <span style="font-size: 0.65rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-top: 6px; letter-spacing: 0.5px;">Churn Risk</span>
                        </div>
                    </div>
                </div>
                <p class="text-xs text-slate-300 mt-2 px-4 leading-relaxed text-center">{desc_text}</p>
            </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="glass-card-container animate-fade-in">', unsafe_allow_html=True)
        st.markdown("""
            <div class="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
                <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider">Top Identified Risk Factors</h3>
                <span class="text-xs text-slate-500 font-medium">Impact Severity</span>
            </div>
        """, unsafe_allow_html=True)

        risk_drivers = []
        if contract == "Month-to-month":
            risk_drivers.append(("Contract Commitment", "Month-to-month terms elevate flight risk", "+32% High Impact", "rose"))
        if internet_service == "Fiber optic" and tech_support == "No":
            risk_drivers.append(("Service Package", "High-speed Fiber Optic without active Tech Support", "+18% Medium Impact", "amber"))
        if tech_support == "No":
            risk_drivers.append(("Technical Assistance", "Absence of Tech Support subscription", "+15% Medium Impact", "amber"))
        if payment_method == "Electronic check":
            risk_drivers.append(("Payment Channel", "Manual Electronic Check payment method", "+12% Low Impact", "indigo"))
        if monthly_charges > 70:
            risk_drivers.append(("Billing Baseline", f"₹{monthly_charges}/mo billing exceeds tier average", "+10% Low Impact", "indigo"))
        if tenure < 12:
            risk_drivers.append(("Customer Lifecycle", f"Tenure of {tenure} months within fragile 1st year", "+25% High Impact", "rose"))

        if risk_drivers:
            for title, desc, impact, color in risk_drivers:
                if color == "rose":
                    badge_cls = "bg-rose-500/10 text-rose-400 border-rose-500/20"
                    dot_cls = "bg-rose-500"
                elif color == "amber":
                    badge_cls = "bg-amber-500/10 text-amber-400 border-amber-500/20"
                    dot_cls = "bg-amber-500"
                else:
                    badge_cls = "bg-indigo-500/10 text-indigo-400 border-indigo-500/20"
                    dot_cls = "bg-indigo-500"

                st.markdown(f"""
                    <div class="flex items-center justify-between p-3 mb-2 rounded-xl bg-slate-900/50 border border-slate-800/80 hover:border-slate-700 transition-all">
                        <div class="flex items-center gap-3">
                            <span class="w-2 h-2 rounded-full {dot_cls}"></span>
                            <div>
                                <h4 class="text-xs font-bold text-slate-200">{title}</h4>
                                <p class="text-[11px] text-slate-400">{desc}</p>
                            </div>
                        </div>
                        <span class="px-2.5 py-1 text-[10px] font-semibold rounded-md border {badge_cls}">{impact}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
                    ✓ No high-severity churn drivers detected for this customer profile.
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 2: What-If Retention Simulator
# =========================================================
elif "What-If Retention Simulator" in selected_page:
    st.markdown("""
        <div class="rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900/90 to-purple-950/30 border border-slate-800 p-6 mb-8">
            <h1 class="text-2xl font-extrabold text-slate-50 tracking-tight">What-If Retention Simulator & CLV ROI Calculator</h1>
            <p class="text-sm text-slate-400 mt-1">Adjust contract terms and incentive packages to simulate counterfactual risk reduction and financial ROI.</p>
        </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        st.markdown('<div class="glass-card-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-4">Intervention Strategy Controls</h3>', unsafe_allow_html=True)
        
        baseline_cust = st.session_state.customer_profile.copy()

        sim_contract = st.selectbox(
            "Target Contract Term",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(baseline_cust['Contract'])
        )
        sim_tech_support = st.selectbox(
            "Add Tech Support Package",
            ["No", "Yes"],
            index=0 if baseline_cust['TechSupport'] != "Yes" else 1
        )
        sim_security = st.selectbox(
            "Add Online Security Package",
            ["No", "Yes"],
            index=0 if baseline_cust['OnlineSecurity'] != "Yes" else 1
        )
        sim_payment = st.selectbox(
            "Target Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(baseline_cust['PaymentMethod'])
        )
        monthly_discount = st.slider(
            "Monthly Promotional Discount (₹)",
            min_value=0.0, max_value=500.0, value=100.0, step=25.0
        )
        st.markdown('</div>', unsafe_allow_html=True)

        sim_cust = baseline_cust.copy()
        sim_cust['Contract'] = sim_contract
        sim_cust['TechSupport'] = sim_tech_support
        sim_cust['OnlineSecurity'] = sim_security
        sim_cust['PaymentMethod'] = sim_payment

        added_service_cost = (100.0 if sim_tech_support == "Yes" and baseline_cust['TechSupport'] != "Yes" else 0.0) + \
                             (100.0 if sim_security == "Yes" and baseline_cust['OnlineSecurity'] != "Yes" else 0.0)
        sim_cust['MonthlyCharges'] = max(18.0, baseline_cust['MonthlyCharges'] + added_service_cost - monthly_discount)

        base_pred, base_prob = predict_customer(baseline_cust)
        sim_pred, sim_prob = predict_customer(sim_cust)
        risk_drop = (base_prob - sim_prob) * 100

    with sim_col2:
        st.markdown('<div class="grid grid-cols-3 gap-3 mb-6">', unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
                <div class="glass-card-container text-center p-3">
                    <div class="text-[10px] font-bold text-slate-400 uppercase">Baseline Risk</div>
                    <div class="text-xl font-extrabold mt-1 {'text-rose-400' if base_prob > 0.5 else 'text-emerald-400'}">{base_prob*100:.1f}%</div>
                    <div class="text-[10px] text-slate-500">Original Profile</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
                <div class="glass-card-container text-center p-3">
                    <div class="text-[10px] font-bold text-slate-400 uppercase">Simulated Risk</div>
                    <div class="text-xl font-extrabold text-emerald-400 mt-1">{sim_prob*100:.1f}%</div>
                    <div class="text-[10px] text-slate-500">Target Intervention</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
                <div class="glass-card-container text-center p-3">
                    <div class="text-[10px] font-bold text-slate-400 uppercase">Risk Drop</div>
                    <div class="text-xl font-extrabold text-indigo-400 mt-1">-{risk_drop:.1f}%</div>
                    <div class="text-[10px] text-slate-500">Delta Reduction</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            y=['Baseline Profile', 'Simulated Offer'],
            x=[base_prob*100, sim_prob*100],
            orientation='h',
            marker=dict(
                color=['rgba(239, 68, 68, 0.85)', 'rgba(16, 185, 129, 0.85)'],
                line=dict(color=['#ef4444', '#10b981'], width=1)
            ),
            text=[f"{base_prob*100:.1f}%", f"{sim_prob*100:.1f}%"],
            textposition='auto'
        ))
        fig_sim.update_layout(
            title=dict(text="Risk Comparison (%)", font=dict(size=13, color="#94a3b8")),
            xaxis=dict(range=[0, 100], title="Probability (%)"),
            height=200,
            margin=dict(l=10, r=10, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.4)',
            font=dict(color='#f8fafc')
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        annual_revenue = baseline_cust['MonthlyCharges'] * 12
        saved_annual_revenue = (base_prob - sim_prob) * annual_revenue
        annual_incentive_cost = (monthly_discount * 12)
        net_financial_value = saved_annual_revenue - annual_incentive_cost
        roi_percentage = (net_financial_value / annual_incentive_cost * 100) if annual_incentive_cost > 0 else 100.0

        st.markdown('<div class="glass-card-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">Financial ROI & CLV Impact</h3>', unsafe_allow_html=True)
        roi_c1, roi_c2, roi_c3 = st.columns(3)
        roi_c1.metric("Retained Annual Revenue", f"₹{saved_annual_revenue:,.2f}/yr")
        roi_c2.metric("Incentive Cost", f"₹{annual_incentive_cost:,.2f}/yr")
        roi_c3.metric("Net Financial Gain", f"₹{net_financial_value:,.2f}/yr", delta=f"{roi_percentage:.0f}% ROI")
        st.markdown('</div>', unsafe_allow_html=True)
