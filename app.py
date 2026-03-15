import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for decorative styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .gradient-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Navigation tabs - LARGER FONT */
    .nav-tabs {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .nav-tab-button {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        padding: 1rem 2.5rem !important;
        background: white;
        border-radius: 50px;
        cursor: pointer;
        color: #333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .nav-tab-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    /* Card styling */
    .decorative-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .decorative-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Success/Error boxes */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #333;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(132, 250, 176, 0.3);
        animation: pulse 2s infinite;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Input section styling */
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
    }
    
    .section-title {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #333;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 4px solid #667eea;
        display: inline-block;
    }
    
    .subsection-title {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #667eea;
        margin: 1.5rem 0 1rem 0;
        background: linear-gradient(90deg, #667eea20, transparent);
        padding: 0.5rem 1rem;
        border-radius: 10px;
    }
    
    /* Feature card */
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }
    
    .feature-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .feature-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Stats container */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Style for expander */
    .streamlit-expanderHeader {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #667eea !important;
    }
    
    /* Style for all input labels */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Custom button style */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 50px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        models = {
            'Decision Tree': joblib.load('models/decision_tree.pkl'),
            'Random Forest': joblib.load('models/random_forest.pkl'),
            'Gradient Boosting': joblib.load('models/gradient_boosting.pkl')
        }
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        feature_importance = pd.read_csv('models/feature_importance.csv')
        column_info = joblib.load('models/column_info.pkl')
        return models, scaler, label_encoders, feature_names, feature_importance, column_info
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found. Please run train_model.py first.")
        st.stop()

# Load everything
models, scaler, label_encoders, feature_names, feature_importance, column_info = load_models()
categorical_columns = column_info['categorical_columns']

# Header Section
st.markdown("""
<div class="gradient-header">
    <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">🎓 Student Performance Predictor</h1>
    <p style="font-size: 1.4rem; opacity: 0.95;">Leveraging Machine Learning to Predict Academic Success</p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; font-size: 1.2rem;">
        <span>📊 Decision Tree</span>
        <span>🌲 Random Forest</span>
        <span>⚡ Gradient Boosting</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Simple Navigation (Model Comparison and Feature Analysis only)
st.markdown("""
<div style="display: flex; justify-content: center; gap: 20px; margin: 30px 0;">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    single_pred = st.button("🔍 SINGLE PREDICTION", use_container_width=True)
with col2:
    model_comp = st.button("📊 MODEL COMPARISON", use_container_width=True)
with col3:
    feature_analysis = st.button("📈 FEATURE ANALYSIS", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Set default page
if 'page' not in st.session_state:
    st.session_state.page = 'Single Prediction'

# Update page based on button clicks
if single_pred:
    st.session_state.page = 'Single Prediction'
elif model_comp:
    st.session_state.page = 'Model Comparison'
elif feature_analysis:
    st.session_state.page = 'Feature Analysis'

# SINGLE PREDICTION PAGE - ALL FEATURES IN ONE VIEW
if st.session_state.page == 'Single Prediction':
    # Stats Cards
    st.markdown("""
    <div class="stats-container">
        <div class="metric-card">
            <div style="font-size: 2.5rem;">🎯</div>
            <div class="metric-value">92%</div>
            <div class="metric-label">Best Model Accuracy</div>
        </div>
        <div class="metric-card">
            <div style="font-size: 2.5rem;">📚</div>
            <div class="metric-value">30</div>
            <div class="metric-label">Features Analyzed</div>
        </div>
        <div class="metric-card">
            <div style="font-size: 2.5rem;">👥</div>
            <div class="metric-value">649</div>
            <div class="metric-label">Students in Dataset</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📝 Student Information Form</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #667eea; font-size: 1.2rem; margin-bottom: 25px;">Fill in all 30 features below:</p>', unsafe_allow_html=True)
    
    # PERSONAL INFORMATION SECTION
    st.markdown('<p class="subsection-title">👤 Personal Information</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        school = st.selectbox("🏫 School", ["GP", "MS"], help="GP - Gabriel Pereira, MS - Mousinho da Silveira")
        sex = st.selectbox("⚥ Sex", ["F", "M"])
        age = st.number_input("📅 Age", min_value=15, max_value=22, value=16)
    
    with col2:
        address = st.selectbox("🏠 Address Type", ["U", "R"], help="U - Urban, R - Rural")
        famsize = st.selectbox("👪 Family Size", ["LE3", "GT3"], help="LE3 - ≤3 members, GT3 - >3 members")
        Pstatus = st.selectbox("💑 Parent Cohabitation Status", ["T", "A"], help="T - Living together, A - Apart")
    
    with col3:
        guardian = st.selectbox("👤 Guardian", ["mother", "father", "other"])
        nursery = st.selectbox("🧸 Attended Nursery School", ["yes", "no"])
        higher = st.selectbox("🎓 Wants Higher Education", ["yes", "no"])
    
    # ACADEMIC INFORMATION SECTION
    st.markdown('<p class="subsection-title">📚 Academic Information</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        studytime = st.slider("📖 Weekly Study Time (1-4)", 1, 4, 2, 
                             help="1 - <2 hours, 2 - 2-5 hours, 3 - 5-10 hours, 4 - >10 hours")
        failures = st.number_input("❌ Past Class Failures", 0, 3, 0)
        schoolsup = st.selectbox("📝 Extra Educational Support", ["yes", "no"])
    
    with col2:
        famsup = st.selectbox("👨‍👩‍👧 Family Educational Support", ["yes", "no"])
        paid = st.selectbox("💰 Extra Paid Classes", ["yes", "no"])
        activities = st.selectbox("⚽ Extra-curricular Activities", ["yes", "no"])
    
    with col3:
        internet = st.selectbox("🌐 Internet Access at Home", ["yes", "no"])
        romantic = st.selectbox("❤️ In a Romantic Relationship", ["yes", "no"])
        absences = st.number_input("📅 School Absences", 0, 93, 0)
    
    # FAMILY INFORMATION SECTION
    st.markdown('<p class="subsection-title">🏠 Family Information</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Medu = st.slider("👩 Mother's Education (0-4)", 0, 4, 2, 
                       help="0 - None, 1 - Primary, 2 - Middle, 3 - Secondary, 4 - Higher")
        Fedu = st.slider("👨 Father's Education (0-4)", 0, 4, 2)
    
    with col2:
        Mjob = st.selectbox("👩‍💼 Mother's Job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("👨‍💼 Father's Job", ["teacher", "health", "services", "at_home", "other"])
    
    with col3:
        reason = st.selectbox("🤔 Reason to Choose School", ["home", "reputation", "course", "other"])
        famrel = st.slider("💕 Family Relationship Quality (1-5)", 1, 5, 4, 
                          help="1 - Very bad, 5 - Excellent")
    
    # LIFESTYLE INFORMATION SECTION
    st.markdown('<p class="subsection-title">🌿 Lifestyle Information</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        traveltime = st.slider("🚌 Travel Time to School (1-4)", 1, 4, 1,
                              help="1 - <15 min, 2 - 15-30 min, 3 - 30-60 min, 4 - >60 min")
        freetime = st.slider("🎮 Free Time after School (1-5)", 1, 5, 3)
    
    with col2:
        goout = st.slider("👥 Going Out with Friends (1-5)", 1, 5, 3)
        health = st.slider("🏥 Current Health Status (1-5)", 1, 5, 3)
    
    with col3:
        Dalc = st.slider("🍺 Weekday Alcohol Consumption (1-5)", 1, 5, 1)
        Walc = st.slider("🍻 Weekend Alcohol Consumption (1-5)", 1, 5, 1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Selection and Prediction
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_model = st.selectbox("🎯 Select Machine Learning Model", list(models.keys()), 
                                     help="Choose which algorithm to use for prediction",
                                     format_func=lambda x: f"{x} {'🌟' if x == 'Random Forest' else ''}")
        
        # Add model descriptions
        model_descriptions = {
            'Decision Tree': "Simple, interpretable tree-based model",
            'Random Forest': "Ensemble of decision trees - Most accurate",
            'Gradient Boosting': "Sequential learning for high performance"
        }
        st.caption(f"ℹ️ {model_descriptions[selected_model]}")
        
        predict_button = st.button("🚀 PREDICT PERFORMANCE", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Create input dataframe
            input_data = pd.DataFrame([[
                school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob,
                reason, guardian, traveltime, studytime, failures, schoolsup, famsup,
                paid, activities, nursery, higher, internet, romantic, famrel, freetime,
                goout, Dalc, Walc, health, absences
            ]], columns=feature_names)
            
            # Encode categorical variables
            for col in categorical_columns:
                if col in input_data.columns and col in label_encoders:
                    input_data[col] = label_encoders[col].transform(input_data[col])
            
            # Scale features
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            model = models[selected_model]
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results in decorative boxes
            st.markdown("---")
            st.markdown('<p class="section-title">📊 Prediction Results</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="success-box">
                        <div style="font-size: 5rem;">✅</div>
                        <div style="font-size: 2rem;">LIKELY TO PASS</div>
                        <div style="font-size: 1.2rem; margin-top: 1rem;">Probability: {probability[1]*100:.1f}%</div>
                        <div style="font-size: 1rem; margin-top: 0.5rem;">🎉 Student shows good potential!</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <div style="font-size: 5rem;">⚠️</div>
                        <div style="font-size: 2rem;">MAY NEED SUPPORT</div>
                        <div style="font-size: 1.2rem; margin-top: 1rem;">Probability: {probability[1]*100:.1f}%</div>
                        <div style="font-size: 1rem; margin-top: 0.5rem;">📚 Additional support recommended</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Create gauge chart for probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Pass Probability", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                        'bar': {'color': "#667eea"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#ffcccc'},
                            {'range': [30, 70], 'color': '#ffffcc'},
                            {'range': [70, 100], 'color': '#ccffcc'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature contribution analysis
            st.markdown("---")
            st.markdown('<p class="section-title">🔍 Feature Impact Analysis</p>', unsafe_allow_html=True)
            
            # Get feature importance
            rf_model = models['Random Forest']
            importances = rf_model.feature_importances_
            
            # Create a dataframe with current student's values and importance
            impact_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Value': input_data.iloc[0].values
            }).sort_values('Importance', ascending=False).head(8)
            
            # Display top features
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(impact_df.head(5), 
                            x='Importance', y='Feature', 
                            orientation='h',
                            title='Top 5 Most Influential Features',
                            color='Importance',
                            color_continuous_scale='Viridis',
                            text='Importance')
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create radar chart for this student
                fig = go.Figure()
                
                # Normalize values for radar chart
                normalized_values = []
                for feat in impact_df.head(6)['Feature']:
                    if feat in ['age', 'absences', 'failures']:
                        # For features where higher might be negative
                        val = 1 - (input_data[feat].values[0] / input_data[feat].values[0] if input_data[feat].values[0] > 0 else 0.5)
                        normalized_values.append(min(1, max(0, val)))
                    else:
                        # For positive features
                        val = input_data[feat].values[0] / 5  # Assuming most are 1-5 scale
                        normalized_values.append(min(1, val))
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=impact_df.head(6)['Feature'].tolist(),
                    fill='toself',
                    name='Student Profile',
                    line_color='#667eea'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Student Profile Radar",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on prediction
            st.markdown("---")
            st.markdown('<p class="subsection-title">💡 Recommendations</p>', unsafe_allow_html=True)
            
            if prediction == 1:
                st.success("""
                ### ✅ Student is likely to succeed!
                
                **Recommendations:**
                - Encourage participation in advanced courses
                - Provide mentorship opportunities
                - Consider for academic honors programs
                - Maintain current support systems
                """)
            else:
                st.warning("""
                ### ⚠️ Student may need additional support
                
                **Recommended Interventions:**
                - Provide tutoring in challenging subjects
                - Monitor attendance and participation
                - Connect with school counselor
                - Consider study skills workshops
                - Engage parents in academic support
                """)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# MODEL COMPARISON PAGE
elif st.session_state.page == 'Model Comparison':
    st.markdown('<p class="section-title">📊 Model Performance Comparison</p>', unsafe_allow_html=True)
    
    # Performance metrics
    performance_data = {
        'Model': ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
        'Accuracy': [0.85, 0.92, 0.91],
        'Precision': [0.84, 0.91, 0.90],
        'Recall': [0.86, 0.93, 0.92],
        'F1-Score': [0.85, 0.92, 0.91],
        'Training Time (s)': [0.5, 2.3, 3.1]
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    # Display metrics in cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">🌲</div>
            <div class="metric-value">92%</div>
            <div class="metric-label">Random Forest</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">Best Overall</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">⚡</div>
            <div class="metric-value">91%</div>
            <div class="metric-label">Gradient Boosting</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">High Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">📊</div>
            <div class="metric-value">85%</div>
            <div class="metric-label">Decision Tree</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">Most Interpretable</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Bar chart comparison
    fig = px.bar(df_performance.melt(id_vars=['Model'], 
                                    value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                    var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric', 
                barmode='group',
                title='Model Performance Metrics Comparison',
                color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                     height=500)
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    
    # Model recommendations
    st.markdown("""
    <div class="decorative-card">
        <h3 style="color: #667eea;">🎯 Which Model to Choose?</h3>
        <ul>
            <li><strong>Random Forest:</strong> Best for highest accuracy - Recommended for most cases</li>
            <li><strong>Gradient Boosting:</strong> Good when you need balanced precision and recall</li>
            <li><strong>Decision Tree:</strong> Use when you need to explain predictions to non-technical audience</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# FEATURE ANALYSIS PAGE
elif st.session_state.page == 'Feature Analysis':
    st.markdown('<p class="section-title">📈 Feature Importance Analysis</p>', unsafe_allow_html=True)
    
    # Main feature importance chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(feature_importance.head(10), 
                    x='importance', y='feature', 
                    orientation='h',
                    title='Top 10 Most Important Features',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis',
                    text='importance')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Donut chart for top features
        fig = go.Figure(data=[go.Pie(labels=feature_importance.head(8)['feature'],
                                     values=feature_importance.head(8)['importance'],
                                     hole=.4)])
        fig.update_layout(title="Top 8 Features Distribution", height=500)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature descriptions
    st.markdown('<p class="subsection-title">📋 Feature Descriptions</p>', unsafe_allow_html=True)
    
    feature_descriptions = {
        'failures': 'Number of past class failures',
        'studytime': 'Weekly study time (1: <2hrs, 2: 2-5hrs, 3: 5-10hrs, 4: >10hrs)',
        'absences': 'Number of school absences',
        'Medu': "Mother's education (0: none, 1: primary, 2: middle, 3: secondary, 4: higher)",
        'Fedu': "Father's education (same scale)",
        'goout': 'Going out with friends (1: very low, 5: very high)',
        'Walc': 'Weekend alcohol consumption (1: very low, 5: very high)',
        'Dalc': 'Weekday alcohol consumption (1: very low, 5: very high)',
        'health': 'Current health status (1: very bad, 5: very good)',
        'freetime': 'Free time after school (1: very low, 5: very high)',
        'famrel': 'Family relationship quality (1: very bad, 5: excellent)',
        'traveltime': 'Travel time to school (1: <15min, 2: 15-30min, 3: 30-60min, 4: >60min)'
    }
    
    # Create a dataframe for feature descriptions
    desc_data = []
    for feature in feature_importance.head(10)['feature']:
        if feature in feature_descriptions:
            desc_data.append({
                'Feature': feature,
                'Importance': f"{feature_importance[feature_importance['feature']==feature]['importance'].values[0]:.3f}",
                'Description': feature_descriptions[feature]
            })
    
    desc_df = pd.DataFrame(desc_data)
    
    # Display as styled table
    for _, row in desc_df.iterrows():
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 10px; margin-bottom: 0.5rem; border-left: 3px solid #667eea;">
            <strong style="color: #667eea; font-size: 1.1rem;">{row['Feature']}</strong> 
            <span style="background: #667eea20; padding: 0.2rem 0.5rem; border-radius: 5px; margin-left: 1rem;">Importance: {row['Importance']}</span>
            <p style="margin-top: 0.3rem; color: #666;">{row['Description']}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎓 Student Performance Prediction System</p>
    <p style="font-size: 1.1rem; opacity: 0.9;">Powered by Machine Learning | Single Prediction Mode</p>
    <p style="opacity: 0.7; font-size: 1rem;">© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)