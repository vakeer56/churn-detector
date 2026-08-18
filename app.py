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
    page_title="Customer Churn Intelligence & Retention Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Professional Minimalist CSS & Subtle Glassmorphism Tokens
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Minimalist Dark Palette */
    :root {
        --bg-main: #0b0f19;
        --bg-card: rgba(18, 26, 43, 0.6);
        --border-subtle: rgba(255, 255, 255, 0.07);
        --border-active: rgba(99, 102, 241, 0.4);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-indigo: #6366f1;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
    }
    
    /* Background & Global Typography */
    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.05) 0px, transparent 50%),
            radial-gradient(at 90% 90%, rgba(16, 185, 129, 0.03) 0px, transparent 50%);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Minimal Header Container */
    .header-container {
        background: rgba(18, 26, 43, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 2rem 2.2rem;
        border-radius: 12px;
        border: 1px solid var(--border-subtle);
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }
    .header-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
    }
    .header-badge {
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
        margin-top: 0.6rem;
    }

    /* Subtle Glass Card Container */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 10px;
        border: 1px solid var(--border-subtle);
        padding: 1.4rem;
        margin-bottom: 1.2rem;
    }
    
    /* Section Headings inside Cards */
    .card-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-secondary);
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.5rem;
    }

    /* Minimal Metric Cards */
    .kpi-card {
        background: rgba(18, 26, 43, 0.7);
        border-radius: 8px;
        border: 1px solid var(--border-subtle);
        padding: 1rem;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.2rem 0;
        color: var(--text-primary);
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: var(--text-secondary);
    }

    /* Risk Status Callout Panels */
    .risk-panel-high {
        background: rgba(244, 63, 94, 0.08);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-panel-low {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Crisp Minimal Primary Button */
    .stButton>button {
        background: #4f46e5 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.8rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stButton>button:hover {
        background: #4338ca !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Clean Text Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: rgba(11, 15, 25, 0.8);
        padding: 4px;
        border-radius: 8px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 6px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        color: #a5b4fc !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        font-weight: 600 !important;
    }

    /* Clean Diagnostics Text Item */
    .diag-item {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #64748b;
        margin-bottom: 0.6rem;
    }
    .diag-item-high {
        border-left-color: var(--accent-rose);
        background: rgba(244, 63, 94, 0.05);
    }
    .diag-item-medium {
        border-left-color: var(--accent-amber);
        background: rgba(245, 158, 11, 0.05);
    }
    .diag-item-low {
        border-left-color: var(--accent-indigo);
        background: rgba(99, 102, 241, 0.05);
    }

    h3 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Loading Helper
# ---------------------------------------------------------
@st.cache_resource
def load_model_pipeline():
    model_path = Path('models/churn_model.pkl')
    if model_path.exists():
        return joblib.load(model_path)
    return None

pipeline = load_model_pipeline()

# ---------------------------------------------------------
# Minimalist Header Section
# ---------------------------------------------------------
st.markdown("""
    <div class="header-container">
        <div class="header-title">Customer Churn Intelligence Platform</div>
        <div class="header-subtitle">Predictive risk modeling, scenario simulation, and customer retention strategy analysis</div>
        <div class="header-badge">Model v2.4 • Tuned Random Forest Classifier</div>
    </div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("Trained model pipeline not found at `models/churn_model.pkl`. Please execute `python src/train.py` to train and save the model pipeline.")
    st.stop()

# ---------------------------------------------------------
# Global Helper Functions
# ---------------------------------------------------------
def create_plotly_gauge(prob_val, title="Predicted Churn Risk"):
    """Generates a clean, minimalist Plotly Gauge Chart."""
    val_percent = prob_val * 100
    
    if val_percent < 30:
        bar_color = "#10b981"
    elif val_percent < 60:
        bar_color = "#f59e0b"
    else:
        bar_color = "#f43f5e"
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val_percent,
        number = {'suffix': "%", 'font': {'size': 40, 'color': bar_color, 'family': 'Inter, sans-serif'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 14, 'color': '#94a3b8', 'family': 'Inter, sans-serif'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': "rgba(15, 23, 42, 0.4)",
            'bordercolor': "rgba(255, 255, 255, 0.08)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.08)'},
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.08)'},
                {'range': [60, 100], 'color': 'rgba(244, 63, 94, 0.08)'}
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 3},
                'thickness': 0.75,
                'value': val_percent
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f8fafc'}
    )
    return fig

def predict_customer(data_dict):
    """Encapsulates dataframe creation and prediction logic."""
    df_in = pd.DataFrame([data_dict])
    pred = pipeline.predict(df_in)[0]
    prob = pipeline.predict_proba(df_in)[0, 1]
    return pred, prob

# ---------------------------------------------------------
# Navigation Tabs (Clean Text Only)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Single Customer Risk Analyzer",
    "What-If Retention Simulator",
    "AI Retention Campaign Studio",
    "Batch Churn Risk Auditor",
    "Model Insights & Analytics"
])

# Session state initialization
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

# =========================================================
# TAB 1: Single Customer Risk Analyzer
# =========================================================
with tab1:
    st.markdown("### Customer Demographic & Service Parameters")
    st.caption("Input customer attributes to calculate real-time churn probability and inspect key risk factors.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='glass-card'><div class='card-title'>Demographics & Tenure</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"], index=0 if st.session_state.customer_profile['gender']=='Female' else 1)
        senior_citizen_str = st.selectbox("Senior Citizen Status", ["No", "Yes"], index=st.session_state.customer_profile['SeniorCitizen'])
        partner = st.selectbox("Partner Status", ["Yes", "No"], index=0 if st.session_state.customer_profile['Partner']=='Yes' else 1)
        dependents = st.selectbox("Dependents Status", ["No", "Yes"], index=0 if st.session_state.customer_profile['Dependents']=='No' else 1)
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=int(st.session_state.customer_profile['tenure']))
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'><div class='card-title'>Subscribed Services</div>", unsafe_allow_html=True)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if st.session_state.customer_profile['PhoneService']=='Yes' else 1)
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"], index=0 if st.session_state.customer_profile['InternetService']=='Fiber optic' else (1 if st.session_state.customer_profile['InternetService']=='DSL' else 2))
        online_security = st.selectbox("Online Security Service", ["No", "Yes", "No internet service"], index=0 if st.session_state.customer_profile['OnlineSecurity']=='No' else 1)
        online_backup = st.selectbox("Online Backup Service", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support Subscription", ["No", "Yes", "No internet service"], index=0 if st.session_state.customer_profile['TechSupport']=='No' else 1)
        streaming_tv = st.selectbox("Streaming TV Service", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies Service", ["Yes", "No", "No internet service"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='glass-card'><div class='card-title'>Account & Billing Terms</div>", unsafe_allow_html=True)
        contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"], index=0 if st.session_state.customer_profile['Contract']=='Month-to-month' else 1)
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], index=0 if st.session_state.customer_profile['PaymentMethod']=='Electronic check' else 2)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=float(st.session_state.customer_profile['MonthlyCharges']), step=1.0)
        calc_total = float(np.round(monthly_charges * tenure, 2))
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=10000.0, value=calc_total, step=25.0)
        st.markdown("</div>", unsafe_allow_html=True)

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

    analyze_btn = st.button("ANALYZE CHURN RISK")
    
    pred_val, prob_val = predict_customer(current_customer)
    
    st.markdown("---")
    st.markdown("### Risk Analysis Results")
    
    res_col1, res_col2 = st.columns([1.2, 1.8])
    
    with res_col1:
        st.plotly_chart(create_plotly_gauge(prob_val, "Model Predicted Risk"), use_container_width=True)
        
        if pred_val == 1:
            st.markdown(f"""
                <div class="risk-panel-high">
                    <div style="color: #f43f5e; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">HIGH CHURN RISK</div>
                    <h2 style="color: #f43f5e; margin: 0.3rem 0; font-size: 2.2rem; font-weight: 700;">{prob_val*100:.1f}% Probability</h2>
                    <p style="color: #fda4af; margin: 0; font-size: 0.85rem;">Customer exhibits elevated churn indicators.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="risk-panel-low">
                    <div style="color: #10b981; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">LOW CHURN RISK</div>
                    <h2 style="color: #10b981; margin: 0.3rem 0; font-size: 2.2rem; font-weight: 700;">{prob_val*100:.1f}% Probability</h2>
                    <p style="color: #a7f3d0; margin: 0; font-size: 0.85rem;">Account displays strong stability profile.</p>
                </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("<div class='glass-card'><div class='card-title'>Risk Factor Diagnostics</div>", unsafe_allow_html=True)
        
        risk_drivers = []
        if contract == "Month-to-month":
            risk_drivers.append(("Contract Terms", "Month-to-month commitment (+32% estimated risk impact)", "high"))
        if internet_service == "Fiber optic":
            risk_drivers.append(("Service Line", "Fiber Optic without technical support package (+18% risk impact)", "medium"))
        if tech_support == "No":
            risk_drivers.append(("Support Package", "Absence of active Tech Support (+15% risk impact)", "medium"))
        if payment_method == "Electronic check":
            risk_drivers.append(("Payment Method", "Manual Electronic Check payment (+12% risk impact)", "low"))
        if monthly_charges > 70:
            risk_drivers.append(("Billing Rate", f"${monthly_charges}/month billing rate (+10% risk impact)", "low"))
        if tenure < 12:
            risk_drivers.append(("Account Age", f"Customer tenure ({tenure} months) within initial 12-month lifecycle", "high"))
            
        if risk_drivers:
            for title, desc, severity in risk_drivers:
                css_class = f"diag-item-{severity}"
                st.markdown(f"""
                    <div class="diag-item {css_class}">
                        <strong style="color: var(--text-primary);">{title}:</strong> 
                        <span style="color: var(--text-secondary); font-size: 0.88rem;">{desc}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='diag-item diag-item-low'>No high-weight risk drivers identified for this profile.</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2: What-If Retention Simulator
# =========================================================
with tab2:
    st.markdown("### What-If Retention Simulator & CLV ROI Calculator")
    st.caption("Adjust contract parameters and service add-ons to evaluate counterfactual risk reduction and financial return.")
    
    sim_col1, sim_col2 = st.columns([1, 1.2])
    
    with sim_col1:
        st.markdown("<div class='glass-card'><div class='card-title'>Counterfactual Strategy Controls</div>", unsafe_allow_html=True)
        
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
            "Monthly Promotional Discount ($)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=2.5
        )
        st.markdown("</div>", unsafe_allow_html=True)

        sim_cust = baseline_cust.copy()
        sim_cust['Contract'] = sim_contract
        sim_cust['TechSupport'] = sim_tech_support
        sim_cust['OnlineSecurity'] = sim_security
        sim_cust['PaymentMethod'] = sim_payment
        
        added_service_cost = (5.0 if sim_tech_support == "Yes" and baseline_cust['TechSupport'] != "Yes" else 0.0) + \
                             (5.0 if sim_security == "Yes" and baseline_cust['OnlineSecurity'] != "Yes" else 0.0)
        sim_cust['MonthlyCharges'] = max(18.0, baseline_cust['MonthlyCharges'] + added_service_cost - monthly_discount)
        
        base_pred, base_prob = predict_customer(baseline_cust)
        sim_pred, sim_prob = predict_customer(sim_cust)
        
        risk_drop = (base_prob - sim_prob) * 100
        
    with sim_col2:
        st.markdown("### Strategy Impact Analysis")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Baseline Risk</div>
                    <div class="kpi-value" style="color: {'#f43f5e' if base_prob > 0.5 else '#10b981'};">{base_prob*100:.1f}%</div>
                    <div class="kpi-sub">Original Profile</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Simulated Risk</div>
                    <div class="kpi-value" style="color: {'#10b981' if sim_prob < 0.3 else '#f59e0b'};">{sim_prob*100:.1f}%</div>
                    <div class="kpi-sub">Target Intervention</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Risk Reduction</div>
                    <div class="kpi-value" style="color: #6366f1;">-{risk_drop:.1f}%</div>
                    <div class="kpi-sub">Delta Risk Drop</div>
                </div>
            """, unsafe_allow_html=True)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            y=['Baseline Profile', 'Simulated Offer'],
            x=[base_prob*100, sim_prob*100],
            orientation='h',
            marker=dict(
                color=['rgba(244, 63, 94, 0.75)', 'rgba(16, 185, 129, 0.75)'],
                line=dict(color=['#f43f5e', '#10b981'], width=1)
            ),
            text=[f"{base_prob*100:.1f}%", f"{sim_prob*100:.1f}%"],
            textposition='auto'
        ))
        fig_sim.update_layout(
            title=dict(text="Risk Comparison (%)", font=dict(size=14, color="#94a3b8")),
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

        st.markdown("<div class='glass-card'><div class='card-title'>Financial ROI & CLV Impact</div>", unsafe_allow_html=True)
        
        roi_c1, roi_c2, roi_c3 = st.columns(3)
        roi_c1.metric("Retained Revenue Value", f"${saved_annual_revenue:.2f}/yr")
        roi_c2.metric("Incentive Cost", f"${annual_incentive_cost:.2f}/yr")
        roi_c3.metric("Net Financial Gain", f"${net_financial_value:.2f}/yr", delta=f"{roi_percentage:.0f}% ROI")
        
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3: AI Retention Campaign Studio
# =========================================================
with tab3:
    st.markdown("### AI Retention Campaign Studio")
    st.caption("Generate customer retention communication drafts and sales call talking points.")
    
    cam_col1, cam_col2 = st.columns([1, 1.5])
    
    with cam_col1:
        st.markdown("<div class='glass-card'><div class='card-title'>Campaign Parameters</div>", unsafe_allow_html=True)
        
        cust_name = st.text_input("Customer Name", value="Alex Morgan")
        rep_name = st.text_input("Account Executive Name", value="Sarah Jenkins")
        campaign_tone = st.selectbox("Communication Tone", ["Professional & Executive", "Empathetic & Service-Oriented", "Direct Promotional Offer"])
        offered_perk = st.selectbox("Target Incentive", [
            "$10/mo Discount on 1-Year Contract",
            "Free Tech Support & Online Security Upgrade",
            "Complimentary Service Upgrade + $5 Auto-Pay Credit"
        ])
        st.markdown("</div>", unsafe_allow_html=True)

    with cam_col2:
        st.markdown("### Customer Outreach Template")
        
        email_body = f"""Subject: Dedicated Account Review and Promotional Offer for {cust_name}

Dear {cust_name},

Thank you for your continued account tenure of {st.session_state.customer_profile['tenure']} months with our communications network. 

During a recent review of your plan ({st.session_state.customer_profile['Contract']} contract with {st.session_state.customer_profile['InternetService']} Internet), we identified an eligible preferred rate adjustment for your account:

Account Review Offer:
• Executive Incentive: {offered_perk}
• Priority 24/7 Technical Support
• Guaranteed Monthly Rate Lock

To apply this adjustment to your account or speak with a representative, please reply directly to this communication or contact customer success at 1-800-555-0199.

Sincerely,

{rep_name}
Customer Success Operations
"""
        st.text_area("Retention Email Template", value=email_body, height=260)
        
        st.download_button(
            label="Download Email Template (.txt)",
            data=email_body,
            file_name=f"retention_email_{cust_name.lower().replace(' ', '_')}.txt",
            mime="text/plain"
        )
        
        st.markdown("---")
        st.markdown("### Service Representative Talking Points")
        script_text = f"""1. Account Verification: Confirm tenure of {st.session_state.customer_profile['tenure']} months.
2. Value Assessment: Address high monthly billing baseline (${st.session_state.customer_profile['MonthlyCharges']}/mo).
3. Offer Presentation: Present {offered_perk} in exchange for 12-month contract commitment.
4. Resolution: Confirm immediate monthly credit on current billing cycle.
"""
        st.code(script_text, language="text")

# =========================================================
# TAB 4: Batch Churn Risk Auditor
# =========================================================
with tab4:
    st.markdown("### Batch Churn Risk Auditor & Cohort Analysis")
    st.caption("Process multi-customer datasets to evaluate cohort risk distribution and export targeted retention reports.")
    
    batch_mode = st.radio("Select Data Source", ["Generate 50-Customer Sample Cohort", "Upload CSV Dataset"], horizontal=True)
    
    batch_df = None
    
    if batch_mode == "Generate 50-Customer Sample Cohort":
        if st.button("Generate Cohort & Run Batch Audit"):
            np.random.seed(42)
            n_samples = 50
            genders = np.random.choice(["Female", "Male"], n_samples)
            seniors = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
            partners = np.random.choice(["Yes", "No"], n_samples)
            dependents = np.random.choice(["Yes", "No"], n_samples)
            tenures = np.random.randint(1, 72, n_samples)
            contracts = np.random.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20])
            internets = np.random.choice(["Fiber optic", "DSL", "No"], n_samples, p=[0.45, 0.40, 0.15])
            tech_supports = np.random.choice(["No", "Yes", "No internet service"], n_samples)
            payments = np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n_samples)
            monthly = np.round(np.random.uniform(20.0, 115.0, n_samples), 2)
            total = np.round(monthly * tenures, 2)
            
            batch_df = pd.DataFrame({
                'gender': genders,
                'SeniorCitizen': seniors,
                'Partner': partners,
                'Dependents': dependents,
                'tenure': tenures,
                'PhoneService': ['Yes']*n_samples,
                'MultipleLines': np.random.choice(["No", "Yes"], n_samples),
                'InternetService': internets,
                'OnlineSecurity': np.random.choice(["No", "Yes"], n_samples),
                'OnlineBackup': np.random.choice(["No", "Yes"], n_samples),
                'DeviceProtection': np.random.choice(["No", "Yes"], n_samples),
                'TechSupport': tech_supports,
                'StreamingTV': np.random.choice(["No", "Yes"], n_samples),
                'StreamingMovies': np.random.choice(["No", "Yes"], n_samples),
                'Contract': contracts,
                'PaperlessBilling': np.random.choice(["Yes", "No"], n_samples),
                'PaymentMethod': payments,
                'MonthlyCharges': monthly,
                'TotalCharges': total
            })
    else:
        uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            
    if batch_df is not None:
        probs = pipeline.predict_proba(batch_df)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        batch_df['Churn_Probability'] = np.round(probs * 100, 1)
        batch_df['Churn_Prediction'] = np.where(preds == 1, 'High Risk', 'Low Risk')
        
        st.markdown("### Cohort Analytics Overview")
        
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Total Cohort Size", f"{len(batch_df)} Customers")
        high_risk_cnt = sum(preds)
        col_b2.metric("High Risk Customers", f"{high_risk_cnt}", delta=f"{high_risk_cnt/len(batch_df)*100:.1f}%")
        avg_risk = np.mean(probs)*100
        col_b3.metric("Average Churn Risk", f"{avg_risk:.1f}%")
        revenue_at_risk = batch_df[batch_df['Churn_Prediction']=='High Risk']['MonthlyCharges'].sum() * 12
        col_b4.metric("Annual Revenue at Risk", f"${revenue_at_risk:,.2f}")

        fig_batch = px.histogram(
            batch_df,
            x="Churn_Probability",
            color="Contract",
            nbins=20,
            title="Churn Risk Distribution by Contract Term",
            color_discrete_sequence=['#f43f5e', '#f59e0b', '#10b981'],
            template="plotly_dark"
        )
        fig_batch.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.4)',
            title=dict(font=dict(size=14, color="#94a3b8"))
        )
        st.plotly_chart(fig_batch, use_container_width=True)

        st.markdown("### Flagged Retention Targets")
        sorted_df = batch_df.sort_values(by="Churn_Probability", ascending=False)
        st.dataframe(sorted_df[['tenure', 'Contract', 'InternetService', 'MonthlyCharges', 'TotalCharges', 'Churn_Probability', 'Churn_Prediction']], use_container_width=True)
        
        csv_data = sorted_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Audit Report (CSV)",
            data=csv_data,
            file_name="churn_risk_cohort_audit.csv",
            mime="text/csv"
        )

# =========================================================
# TAB 5: Model Insights & Analytics
# =========================================================
with tab5:
    st.markdown("### Model Architecture & Feature Importance")
    st.caption("Technical analytics regarding model feature importances and pipeline performance metrics.")
    
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(15)
        
        feat_df['CleanFeature'] = feat_df['Feature'].str.replace('cat__', '').str.replace('num__', '')
        
        fig_feat = px.bar(
            feat_df,
            x='Importance',
            y='CleanFeature',
            orientation='h',
            title='Top 15 Feature Importances (Random Forest Pipeline)',
            color='Importance',
            color_continuous_scale='Purples',
            template='plotly_dark'
        )
        fig_feat.update_layout(
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.4)',
            title=dict(font=dict(size=14, color="#94a3b8"))
        )
        st.plotly_chart(fig_feat, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance extraction warning: {e}")
        
    st.markdown("<div class='glass-card'><div class='card-title'>Pipeline Architecture & Validation Metrics</div>", unsafe_allow_html=True)
    st.markdown("""
    - **Classifier**: Random Forest Classifier with `GridSearchCV` hyperparameter optimization.
    - **Preprocessing**: `ColumnTransformer` applying `StandardScaler` to continuous numeric variables (`tenure`, `MonthlyCharges`, `TotalCharges`) and `OneHotEncoder` to nominal categoricals.
    - **Validation Metrics**: Held-out test accuracy of **80.7%**, ROC-AUC of **0.837**, and F1-Score of **0.595**.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
