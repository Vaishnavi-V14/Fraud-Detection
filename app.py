import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, classification_report, precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ML imports with SHAP
try:
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    import shap
    ML_AVAILABLE = True
    SHAP_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    SHAP_AVAILABLE = False

# ========================================
# UI CONFIGURATION & THEME
# ========================================
st.set_page_config(page_title="Ecommerce Fraud Detection", page_icon="🛡️", layout="wide")

# High-Visibility Custom CSS
st.markdown("""
<style>
    /* Main App Background - Deep Midnight */
    .stApp { 
        background: radial-gradient(circle at center, #0f172a, #020617); 
        color: #FFFFFF !important; 
    }
    
    /* Glassmorphism Cards */
    .glass-card { 
        background: rgba(30, 41, 59, 0.7); 
        backdrop-filter: blur(12px); 
        border-radius: 15px; 
        border: 1px solid rgba(0, 212, 255, 0.3); 
        padding: 25px; 
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }
    
    /* Headers - Neon Cyan */
    h1, h2, h3 { 
        color: #00d4ff !important; 
        text-shadow: 0px 0px 10px rgba(0, 212, 255, 0.4);
        font-weight: 800 !important;
    }
    
    /* Clear Text Labels */
    label, p, .stMarkdown { 
        color: #FFFFFF !important; 
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }

    /* Primary Action Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        color: white !important;
        border: none;
        padding: 12px;
        border-radius: 10px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: 0.4s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(2, 6, 23, 0.9);
        border-right: 2px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)


# ========================================
# 🌍 BANGALORE LOCATION
# ========================================
def get_bangalore_info():
    return "📍 LIVE: Bangalore, Karnataka", "🔒 HQ: Bangalore | 12.97°N, 77.59°E"

live_loc, hq_loc = get_bangalore_info()

# ========================================
# ✅ AUTHENTICATION SYSTEM
# ========================================
if 'users' not in st.session_state:
    st.session_state.users = {'admin': '123'}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'role' not in st.session_state:
    st.session_state.role = ""
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'model_result' not in st.session_state:
    st.session_state.model_result = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'XGBClassifier'

def register_user(username, password):
    """Register new user"""
    username = username.strip().lower()
    password = password.strip()
    
    if len(username) < 3:
        return False, "❌ Username must be 3+ characters"
    if len(password) < 3:
        return False, "❌ Password must be 3+ characters"
    if username in st.session_state.users:
        return False, f"❌ User '{username}' already exists!"
    
    st.session_state.users[username] = password
    return True, f"✅ Registered '{username}' successfully!"

def login_user(username, password):
    """Login user"""
    username = username.strip().lower()
    password = password.strip()
    
    if username in st.session_state.users and st.session_state.users[username] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = "ADMIN" if username == "admin" else "USER"
        return True, "✅ Login successful!"
    return False, "❌ Invalid username or password!"

def logout_user():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""

# ========================================
# ✅ DATASET GENERATOR 
# ========================================
@st.cache_data
def generate_production_dataset(n=8000):
    np.random.seed(42)
    data = {}
    
    # Transaction ID
    data['transaction_id'] = [f"TXN{str(i).zfill(8)}" for i in range(1, n+1)]
    
    # Customer ID
    data['customer_id'] = [f"CUST{np.random.randint(10000, 99999)}" for _ in range(n)]
    
    # Transaction Date
    start_date = pd.Timestamp('2024-01-01')
    data['transaction_date'] = [start_date + pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(n)]
    
    # Transaction Amount
    data['transaction_amount'] = np.random.lognormal(7.9, 1.3, n).clip(25, 25000)
    
    # Payment Method
    payment_methods = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Wallet']
    data['payment_method'] = np.random.choice(payment_methods, n, p=[0.35, 0.3, 0.2, 0.1, 0.05])
    
    # Product Category
    product_categories = ['Electronics', 'Clothing', 'Grocery', 'Furniture', 'Beauty']
    data['product_category'] = np.random.choice(product_categories, n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Quantity
    data['quantity'] = np.random.poisson(2, n).clip(1, 25)
    
    # Customer Age
    data['customer_age'] = np.random.normal(36, 14, n).clip(18, 75).astype(int)
    
    # Device Used
    devices = ['Mobile', 'Desktop', 'Tablet', 'Other']
    data['device_used'] = np.random.choice(devices, n, p=[0.4, 0.35, 0.2, 0.05])
    
    # IP Address
    data['ip_address'] = [f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,255)}" for _ in range(n)]
    
    # Shipping Address
    cities = ['Bangalore', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Kolkata', 'Pune', 'Ahmedabad']
    data['shipping_address'] = np.random.choice(cities, n)
    
    # Billing Address
    data['billing_address'] = np.random.choice(cities, n)
    
    # Address Match (1 if shipping == billing, 0 otherwise)
    data['is_address_match'] = (np.array(data['shipping_address']) == np.array(data['billing_address'])).astype(int)
    
    # Account Age Days
    data['account_age_days'] = np.random.exponential(600, n).clip(1, 3650).astype(int)
    
    # Transaction Hour
    hour_probs = np.array([0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.09, 0.08, 0.07, 0.06] + 
                         [0.09, 0.09, 0.09, 0.09, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])
    data['transaction_hour'] = np.random.choice(range(24), n, p=hour_probs/hour_probs.sum())
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Encode categorical for fraud calculation
    payment_encoded = df['payment_method'].map({v: k for k, v in enumerate(payment_methods)})
    device_encoded = df['device_used'].map({v: k for k, v in enumerate(devices)})
    
    # Calculate fraud probability
    fraud_prob = (
        0.4 * (df['transaction_amount'] > 8000) + 
        0.3 * (df['quantity'] > 10) + 
        0.25 * (df['customer_age'] < 22) +
        0.3 * (device_encoded == 3) + 
        0.4 * (df['is_address_match'] == 0) + 
        0.25 * (df['transaction_hour'] < 6) + 
        0.2 * (df['account_age_days'] < 45) +
        np.random.uniform(0, 0.2, n)
    )
    df['is_fraudulent'] = (fraud_prob > 0.45).astype(int)
    
    return df

def safe_extract_scalar(value):
    try:
        if hasattr(value, '__len__') and len(value) == 1: 
            return float(value.item())
        elif isinstance(value, (int, float)): 
            return float(value)
        elif isinstance(value, np.ndarray) and value.size == 1: 
            return float(value.item())
        return 0.0
    except: 
        return 0.0

# ========================================
# 🚀 ADVANCED 3-MODEL SYSTEM
# ========================================
def train_advanced_model(df, model_type):
    if not ML_AVAILABLE or df is None: 
        return None
    
    # Prepare features
    df_encoded = df.copy()
    
    # Encode categorical variables
    payment_map = {'Credit Card': 0, 'Debit Card': 1, 'UPI': 2, 'Net Banking': 3, 'Wallet': 4}
    product_map = {'Electronics': 0, 'Clothing': 1, 'Grocery': 2, 'Furniture': 3, 'Beauty': 4}
    device_map = {'Mobile': 0, 'Desktop': 1, 'Tablet': 2, 'Other': 3}
    
    df_encoded['payment_method_encoded'] = df_encoded['payment_method'].map(payment_map)
    df_encoded['product_category_encoded'] = df_encoded['product_category'].map(product_map)
    df_encoded['device_used_encoded'] = df_encoded['device_used'].map(device_map)
    
    features = ['transaction_amount', 'payment_method_encoded', 'product_category_encoded', 
                'quantity', 'customer_age', 'device_used_encoded', 'account_age_days', 
                'transaction_hour', 'is_address_match']
    
    X = df_encoded[features].astype(np.float64)
    y = df_encoded['is_fraudulent'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'XGBClassifier': XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=6, 
                                     scale_pos_weight=3.5, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', 
                                             random_state=42, n_jobs=-1),
        'Stacking': StackingClassifier(
            estimators=[('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')),
                       ('rf', RandomForestClassifier(n_estimators=100, random_state=42))],
            final_estimator=LogisticRegression(class_weight='balanced'),
            cv=5, n_jobs=-1
        )
    }
    
    model = models[model_type]
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    explainer = None
    shap_values = None
    if SHAP_AVAILABLE:
        try: 
            if model_type != 'Stacking':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_scaled)
        except: 
            pass
    
    return {
        'model': model, 'scaler': scaler, 'features': features, 
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train.values, 'y_test': y_test.values, 
        'y_proba': y_proba, 'auc': auc, 'y_pred': y_pred,
        'model_type': model_type, 'explainer': explainer, 'shap_values': shap_values,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'feature_names': features
    }

def predict_with_shap(input_data, model_result):
    features = model_result['features']
    model, scaler, explainer = model_result['model'], model_result['scaler'], model_result.get('explainer')
    
    row = [float(input_data.get(f, 0.0)) for f in features]
    X = np.array([row], dtype=np.float64)
    X_scaled = scaler.transform(X)
    
    proba = model.predict_proba(X_scaled)[0]
    prediction = 1 if proba[1] > 0.45 else 0
    
    shap_values = None
    if explainer:
        try:
            shap_result = explainer.shap_values(X_scaled)
            shap_values = shap_result[1] if isinstance(shap_result, list) else shap_result
        except: 
            pass
    
    feature_importance = {feat: safe_extract_scalar(shap_values[0][i]) if shap_values is not None else 0.0 
                         for i, feat in enumerate(features)}
    
    return {
        'prediction': prediction, 'confidence': float(proba[prediction]),
        'probability_fraud': float(proba[1]), 'feature_importance': feature_importance,
        'shap_values': shap_values, 'X_scaled': X_scaled
    }

# ========================================
# 📊 SHAP EXPLAINABILITY CHARTS
# ========================================
def create_shap_explainability_charts(model_result, prediction_result=None):
    """Create comprehensive SHAP explainability charts"""
    charts = []
    features = model_result['feature_names']
    
    if prediction_result and prediction_result.get('feature_importance'):
        # 1. SHAP Waterfall Chart (Bar format)
        feature_importance = prediction_result['feature_importance']
        numeric_imp = {k: safe_extract_scalar(v) for k, v in feature_importance.items()}
        
        sorted_features = sorted(numeric_imp.items(), key=lambda x: abs(x[1]), reverse=True)
        feature_names = [f[0] for f in sorted_features]
        shap_vals = [f[1] for f in sorted_features]
        colors = ['#ef4444' if v > 0 else '#3b82f6' for v in shap_vals]
        
        fig1 = go.Figure(go.Bar(
            x=shap_vals,
            y=feature_names,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.3f}' for v in shap_vals],
            textposition='outside'
        ))
        fig1.update_layout(
            title="🔍 SHAP Feature Impact (Waterfall Style)",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Features",
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(15,23,42,0.9)',
            plot_bgcolor='rgba(15,23,42,0.9)'
        )
        charts.append(('waterfall', fig1))
        
        # 2. SHAP Force Plot Style (Contribution Chart)
        positive_contrib = {k: v for k, v in numeric_imp.items() if v > 0}
        negative_contrib = {k: v for k, v in numeric_imp.items() if v <= 0}
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='🚨 Fraud Indicators',
            x=list(positive_contrib.keys()),
            y=list(positive_contrib.values()),
            marker_color='#ef4444'
        ))
        fig2.add_trace(go.Bar(
            name='🛡️ Legit Indicators',
            x=list(negative_contrib.keys()),
            y=list(negative_contrib.values()),
            marker_color='#10b981'
        ))
        fig2.update_layout(
            title="⚖️ Fraud vs Legitimate Contribution",
            barmode='group',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(15,23,42,0.9)',
            plot_bgcolor='rgba(15,23,42,0.9)'
        )
        charts.append(('contribution', fig2))
        
        # 3. SHAP Absolute Impact Pie Chart
        abs_imp = {k: abs(v) for k, v in numeric_imp.items()}
        fig3 = px.pie(
            values=list(abs_imp.values()),
            names=list(abs_imp.keys()),
            title="🎯 Feature Importance Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig3.update_layout(
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(15,23,42,0.9)',
            plot_bgcolor='rgba(15,23,42,0.9)'
        )
        charts.append(('pie', fig3))
        
        # 4. SHAP Decision Plot Style
        cumulative = np.cumsum([shap_vals[i] for i in range(len(shap_vals))])
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=list(range(len(feature_names))),
            y=cumulative,
            mode='lines+markers',
            marker=dict(size=10, color=colors),
            line=dict(width=2, color='#00d4ff'),
            text=feature_names,
            hovertemplate='Feature: %{text}<br>Cumulative SHAP: %{y:.3f}<extra></extra>'
        ))
        fig4.update_layout(
            title="📈 SHAP Decision Path",
            xaxis_title="Feature Order",
            yaxis_title="Cumulative SHAP Value",
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(15,23,42,0.9)',
            plot_bgcolor='rgba(15,23,42,0.9)'
        )
        charts.append(('decision', fig4))
    
    # 5. Global SHAP Summary (if model has been trained with SHAP values)
    if model_result.get('shap_values') is not None:
        shap_vals = model_result['shap_values']
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        fig5 = go.Figure(go.Bar(
            x=mean_abs_shap,
            y=features,
            orientation='h',
            marker=dict(
                color=mean_abs_shap,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'{v:.4f}' for v in mean_abs_shap],
            textposition='outside'
        ))
        fig5.update_layout(
            title="🌐 Global Feature Importance (Mean |SHAP|)",
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title="Features",
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(15,23,42,0.9)',
            plot_bgcolor='rgba(15,23,42,0.9)'
        )
        charts.append(('global_importance', fig5))
        
        # 6. SHAP Beeswarm Style (Scatter)
        fig6 = go.Figure()
        for i, feat in enumerate(features):
            feat_shap = shap_vals[:, i]
            feat_values = model_result['X_test'][:, i]
            fig6.add_trace(go.Scatter(
                x=feat_shap[:200],
                y=[feat] * min(200, len(feat_shap)),
                mode='markers',
                marker=dict(
                    size=5,
                    color=feat_values[:200],
                    colorscale='RdBu',
                    opacity=0.6
                ),
                name=feat,
                showlegend=False
            ))
        fig6.update_layout(
            title="🐝 SHAP Beeswarm Plot (Feature Value Impact)",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            height=500,
            template='plotly_dark',
            paper_bgcolor='rgba(15,23,42,0.9)',
            plot_bgcolor='rgba(15,23,42,0.9)'
        )
        charts.append(('beeswarm', fig6))
    
    return charts

# ========================================
# 📊 ANALYTICS CHARTS
# ========================================
def create_analytics_charts(df, model_result=None):
    """Create 7+ analytics charts"""
    charts = []
    
    # 1. Fraud Distribution by Amount
    fig1 = px.histogram(
        df, x='transaction_amount', color='is_fraudulent', 
        nbins=50, marginal='box',
        title="💰 Transaction Amount Distribution by Fraud Status",
        color_discrete_map={0: '#3b82f6', 1: '#ef4444'},
        labels={'is_fraudulent': 'Is Fraud', 'transaction_amount': 'Amount (₹)'}
    )
    fig1.update_layout(template='plotly_dark', paper_bgcolor='rgba(15,23,42,0.9)', plot_bgcolor='rgba(15,23,42,0.9)')
    charts.append(('amount_dist', fig1))
    
    # 2. Fraud by Payment Method
    payment_fraud = df.groupby(['payment_method', 'is_fraudulent']).size().unstack(fill_value=0)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name='Legitimate', x=payment_fraud.index, y=payment_fraud[0], marker_color='#3b82f6'))
    fig2.add_trace(go.Bar(name='Fraud', x=payment_fraud.index, y=payment_fraud[1], marker_color='#ef4444'))
    fig2.update_layout(
        title="💳 Fraud by Payment Method",
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)'
    )
    charts.append(('payment_method', fig2))
    
    # 3. Fraud by Product Category
    category_fraud = df.groupby('product_category')['is_fraudulent'].mean().sort_values(ascending=True)
    fig3 = go.Figure(go.Bar(
        x=category_fraud.values * 100,
        y=category_fraud.index,
        orientation='h',
        marker=dict(color=category_fraud.values, colorscale='Reds'),
        text=[f'{v:.1f}%' for v in category_fraud.values * 100],
        textposition='outside'
    ))
    fig3.update_layout(
        title="🛒 Fraud Rate by Product Category",
        xaxis_title="Fraud Rate (%)",
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)'
    )
    charts.append(('category_fraud', fig3))
    
    # 4. Fraud by Hour (Heatmap style)
    hour_fraud = df.groupby('transaction_hour')['is_fraudulent'].agg(['sum', 'count', 'mean'])
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=hour_fraud.index,
        y=hour_fraud['mean'] * 100,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8, color='#ef4444')
    ))
    fig4.update_layout(
        title="🕐 Fraud Rate by Transaction Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Fraud Rate (%)",
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)'
    )
    charts.append(('hour_fraud', fig4))
    
    # 5. Age vs Amount Scatter
    sample_df = df.sample(min(1000, len(df)))
    fig5 = px.scatter(
        sample_df, x='customer_age', y='transaction_amount',
        color='is_fraudulent',
        size='quantity',
        title="👤 Customer Age vs Transaction Amount",
        color_discrete_map={0: '#3b82f6', 1: '#ef4444'},
        labels={'is_fraudulent': 'Is Fraud'}
    )
    fig5.update_layout(template='plotly_dark', paper_bgcolor='rgba(15,23,42,0.9)', plot_bgcolor='rgba(15,23,42,0.9)')
    charts.append(('age_amount', fig5))
    
    # 6. Device Usage Distribution
    device_fraud = df.groupby(['device_used', 'is_fraudulent']).size().unstack(fill_value=0)
    fig6 = px.sunburst(
        df, path=['device_used', 'is_fraudulent'], 
        title="📱 Device Usage & Fraud Distribution",
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    fig6.update_layout(template='plotly_dark', paper_bgcolor='rgba(15,23,42,0.9)', plot_bgcolor='rgba(15,23,42,0.9)')
    charts.append(('device_sunburst', fig6))
    
    # 7. Account Age Analysis
    df['account_age_group'] = pd.cut(df['account_age_days'], bins=[0, 30, 90, 180, 365, 1000, 4000], 
                                     labels=['0-30', '31-90', '91-180', '181-365', '366-1000', '1000+'])
    account_fraud = df.groupby('account_age_group')['is_fraudulent'].mean() * 100
    fig7 = go.Figure(go.Bar(
        x=account_fraud.index.astype(str),
        y=account_fraud.values,
        marker=dict(color=account_fraud.values, colorscale='RdYlGn_r'),
        text=[f'{v:.1f}%' for v in account_fraud.values],
        textposition='outside'
    ))
    fig7.update_layout(
        title="📅 Fraud Rate by Account Age (Days)",
        xaxis_title="Account Age Group",
        yaxis_title="Fraud Rate (%)",
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)'
    )
    charts.append(('account_age', fig7))
    
    # 8. Address Match Impact
    address_data = df.groupby('is_address_match')['is_fraudulent'].agg(['sum', 'count', 'mean'])
    address_data['label'] = ['Mismatch', 'Match']
    fig8 = go.Figure(data=[
        go.Pie(
            labels=address_data['label'],
            values=address_data['sum'],
            hole=0.5,
            marker_colors=['#ef4444', '#10b981'],
            textinfo='label+percent'
        )
    ])
    fig8.update_layout(
        title="📍 Fraud Distribution by Address Match",
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)'
    )
    charts.append(('address_match', fig8))
    
    # 9. Correlation Heatmap
    numeric_cols = ['transaction_amount', 'quantity', 'customer_age', 'account_age_days', 
                   'transaction_hour', 'is_address_match', 'is_fraudulent']
    corr_matrix = df[numeric_cols].corr()
    fig9 = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title="🔗 Feature Correlation Matrix"
    )
    fig9.update_layout(template='plotly_dark', paper_bgcolor='rgba(15,23,42,0.9)', plot_bgcolor='rgba(15,23,42,0.9)')
    charts.append(('correlation', fig9))
    
    return charts

def create_model_performance_charts(model_result):
    """Create model performance charts including confusion matrix"""
    charts = []
    
    y_test = model_result['y_test']
    y_pred = model_result['y_pred']
    y_proba = model_result['y_proba']
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig1 = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Legitimate', 'Fraud'],
        y=['Legitimate', 'Fraud'],
        color_continuous_scale='Blues',
        title="🎯 Confusion Matrix"
    )
    fig1.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)',
        height=400
    )
    charts.append(('confusion_matrix', fig1))
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={model_result["auc"]:.3f})', 
                              line=dict(color='#00d4ff', width=3)))
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', 
                              line=dict(dash='dash', color='gray')))
    fig2.update_layout(
        title="📈 ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)',
        height=400
    )
    charts.append(('roc_curve', fig2))
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=recall, y=precision, mode='lines', 
                              line=dict(color='#7c3aed', width=3)))
    fig3.update_layout(
        title="📊 Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)',
        height=400
    )
    charts.append(('pr_curve', fig3))
    
    # 4. Prediction Distribution
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=y_proba[y_test == 0], name='Legitimate', 
                                marker_color='#3b82f6', opacity=0.7, nbinsx=50))
    fig4.add_trace(go.Histogram(x=y_proba[y_test == 1], name='Fraud', 
                                marker_color='#ef4444', opacity=0.7, nbinsx=50))
    fig4.update_layout(
        title="📉 Prediction Probability Distribution",
        xaxis_title="Fraud Probability",
        yaxis_title="Count",
        barmode='overlay',
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)',
        height=400
    )
    charts.append(('prob_dist', fig4))
    
    # 5. Metrics Bar Chart
    metrics = {
        'Accuracy': model_result['accuracy'],
        'Precision': model_result['precision'],
        'Recall': model_result['recall'],
        'F1 Score': model_result['f1'],
        'AUC-ROC': model_result['auc']
    }
    fig5 = go.Figure(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker=dict(
            color=list(metrics.values()),
            colorscale='Viridis'
        ),
        text=[f'{v:.3f}' for v in metrics.values()],
        textposition='outside'
    ))
    fig5.update_layout(
        title="📊 Model Performance Metrics",
        yaxis_range=[0, 1.1],
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)',
        height=400
    )
    charts.append(('metrics_bar', fig5))
    
    # 6. Threshold Analysis
    thresholds_analysis = np.arange(0.1, 1.0, 0.05)
    precisions, recalls, f1s = [], [], []
    for thresh in thresholds_analysis:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_thresh, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_thresh, zero_division=0))
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=thresholds_analysis, y=precisions, name='Precision', line=dict(color='#10b981')))
    fig6.add_trace(go.Scatter(x=thresholds_analysis, y=recalls, name='Recall', line=dict(color='#3b82f6')))
    fig6.add_trace(go.Scatter(x=thresholds_analysis, y=f1s, name='F1 Score', line=dict(color='#ef4444')))
    fig6.add_vline(x=0.45, line_dash="dash", line_color="yellow", annotation_text="Current Threshold (0.45)")
    fig6.update_layout(
        title="🎚️ Threshold Analysis",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        template='plotly_dark',
        paper_bgcolor='rgba(15,23,42,0.9)',
        plot_bgcolor='rgba(15,23,42,0.9)',
        height=400
    )
    charts.append(('threshold_analysis', fig6))
    
    return charts

# ========================================
# 🚀 MAIN PRODUCTION APP
# ========================================
st.title("🛡️ **Ecommerce Fraud Detection **")
st.markdown("### 🌍 **Bangalore** ")

# LOGIN / REGISTER
if not st.session_state.logged_in:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 🔐 **Access Control**")
    
    col1, col2 = st.columns([1, 1])
    with col1: 
        st.markdown(f"**{hq_loc}**")
    with col2: 
        st.markdown(f"**{live_loc}**")
    
    tab1, tab2 = st.tabs(["🔓 **LOGIN**", "➕ **REGISTER**"])
    
    with tab1:
        st.markdown('<div class="glass-card" style="padding: 2rem;">', unsafe_allow_html=True)
        st.markdown("### 👤 **Login to Continue**")
        
        col1, col2 = st.columns([1.5, 1])
        with col1:
            login_username = st.text_input("🆔 **Username**", placeholder="admin", key="login_user")
        with col2:
            login_password = st.text_input("🔑 **Password**", type="password", placeholder="123", key="login_pass")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**💡 Default:** `admin` / `123`")
        with col2:
            if st.button("🚀 **LOGIN**", type="primary", use_container_width=True, key="login_btn"):
                success, message = login_user(login_username, login_password)
                st.info(message)
                if success:
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="glass-card" style="padding: 2rem;">', unsafe_allow_html=True)
        st.markdown("### ➕ **Create New Account**")
        
        col1, col2 = st.columns(2)
        with col1:
            reg_username = st.text_input("🆔 **New Username**", placeholder="Enter username (3+ chars)", key="reg_user")
        with col2:
            reg_password = st.text_input("🔑 **New Password**", type="password", placeholder="Enter password (3+ chars)", key="reg_pass")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**📝 Min 3 chars each**")
        with col2:
            if st.button("✅ **REGISTER**", type="primary", use_container_width=True, key="register_btn"):
                success, message = register_user(reg_username, reg_password)
                st.info(message)
        
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# DASHBOARD - USER LOGGED IN
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
with col1: 
    st.metric("👑", st.session_state.role)
with col2: 
    st.markdown(f"### 🚀 **Welcome {st.session_state.username.upper()}!**")
with col3: 
    st.metric("🏢", hq_loc)
with col4: 
    st.metric("📍", live_loc)
    if st.button("🔒 **LOGOUT**", use_container_width=True):
        logout_user()
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Generate dataset if needed
if st.session_state.dataset is None:
    with st.spinner("🎯 Generating 8K production transactions..."):
        st.session_state.dataset = generate_production_dataset()

# Sidebar
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🎯 **PRODUCTION MODE**")
    st.markdown("**v18.0 - UPDATED**")
    page = st.radio("📋 **Navigate**", ["🤖 Train", "🔮 Predict", "📊 Analytics", "📈 Model Performance"])
    st.markdown('</div>', unsafe_allow_html=True)

# PAGES
if page == "🤖 Train":
    st.header("🤖 ** Model Training**")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    df = st.session_state.dataset
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 **Total Transactions**", f"{len(df):,}")
    col2.metric("🚨 **Fraud Cases**", int(df['is_fraudulent'].sum()))
    col3.metric("📈 **Fraud Rate**", f"{df['is_fraudulent'].mean():.2%}")
    col4.metric("💰 **Avg Amount**", f"₹{df['transaction_amount'].mean():,.0f}")
    
    # Show dataset sample
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        model_type = st.selectbox("🎯 **Production Model**", 
                                ["XGBClassifier", "RandomForest", "Stacking"])
    with col2:
        st.info("**XGBClassifier**: Fast & Accurate | **RandomForest**: Robust | **Stacking**: XGBoost + RF Ensemble")
    
    if st.button("🚀 **TRAIN MODEL**", type="primary", use_container_width=True):
        with st.spinner(f"Training {model_type}..."):
            result = train_advanced_model(df, model_type)
            if result:
                st.session_state.model_result = result
                st.session_state.model_trained = True
                st.session_state.model_type = model_type
                st.success(f"✅ **{model_type} TRAINED SUCCESSFULLY!**")
                
                # Show metrics immediately
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("📈 **AUC**", f"{result['auc']:.3f}")
                col2.metric("🎯 **Accuracy**", f"{result['accuracy']:.3f}")
                col3.metric("📊 **Precision**", f"{result['precision']:.3f}")
                col4.metric("🔍 **Recall**", f"{result['recall']:.3f}")
                col5.metric("⚖️ **F1 Score**", f"{result['f1']:.3f}")
    
    if st.session_state.model_trained:
        result = st.session_state.model_result
        st.markdown("---")
        st.markdown("### ✅ **Model Ready for Predictions**")
        col1, col2, col3 = st.columns(3)
        col1.metric("🤖 **Model Type**", result['model_type'])
        col2.metric("📈 **AUC-ROC**", f"{result['auc']:.3f}")
        col3.metric("✅ **Status**", "LIVE ✓")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🔮 Predict":
    st.header("🔮 **Real-Time Fraud Detection**")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ **Please train a model first!**")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    model_result = st.session_state.model_result
    
    st.markdown("### 📝 Enter Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        transaction_amount = st.number_input("💰 **Transaction Amount (₹)**", 10.0, 50000.0, 2800.0)
        payment_method = st.selectbox("💳 **Payment Method**", 
                                      ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"])
        product_category = st.selectbox("🛒 **Product Category**", 
                                        ["Electronics", "Clothing", "Grocery", "Furniture", "Beauty"])
    
    with col2:
        quantity = st.number_input("📦 **Quantity**", 1, 50, 2)
        customer_age = st.slider("👤 **Customer Age**", 18, 75, 34)
        device_used = st.selectbox("📱 **Device Used**", ["Mobile", "Desktop", "Tablet", "Other"])
    
    with col3:
        account_age_days = st.slider("📅 **Account Age (Days)**", 1, 2000, 180)
        transaction_hour = st.slider("🕐 **Transaction Hour (0-23)**", 0, 23, 15)
        is_address_match = st.selectbox("📍 **Billing = Shipping Address?**", ["Yes", "No"])
    
    if st.button("🔍 **DETECT FRAUD**", type="primary", use_container_width=True):
        # Encode inputs
        payment_map = {'Credit Card': 0, 'Debit Card': 1, 'UPI': 2, 'Net Banking': 3, 'Wallet': 4}
        product_map = {'Electronics': 0, 'Clothing': 1, 'Grocery': 2, 'Furniture': 3, 'Beauty': 4}
        device_map = {'Mobile': 0, 'Desktop': 1, 'Tablet': 2, 'Other': 3}
        
        input_data = {
            'transaction_amount': float(transaction_amount),
            'payment_method_encoded': float(payment_map.get(payment_method, 0)),
            'product_category_encoded': float(product_map.get(product_category, 0)),
            'quantity': float(quantity),
            'customer_age': float(customer_age),
            'device_used_encoded': float(device_map.get(device_used, 0)),
            'account_age_days': float(account_age_days),
            'transaction_hour': float(transaction_hour),
            'is_address_match': 1.0 if is_address_match == "Yes" else 0.0
        }
        
        result = predict_with_shap(input_data, model_result)
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 **Prediction Results**")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 **Amount**", f"₹{transaction_amount:,.0f}")
        col2.metric("🚨 **Fraud Probability**", f"{result['probability_fraud']:.1%}")
        col3.metric("📊 **Prediction**", "🚨 FRAUD" if result['prediction'] else "🛡️ LEGIT")
        col4.metric("🎯 **Confidence**", f"{result['confidence']:.1%}")
        
        # Alert box
        color = "#ef4444" if result['prediction'] else "#10b981"
        message = "🚨 **FRAUD DETECTED** - BLOCK TRANSACTION!" if result['prediction'] else "✅ **LEGITIMATE** - APPROVE TRANSACTION!"
        st.markdown(f"""
        <div style='padding: 2.5rem; background: {color}; border-radius: 20px; text-align: center; 
                   color: white; font-size: 1.6rem; font-weight: bold; margin: 1rem 0;'>
            {message}
        </div>
        """, unsafe_allow_html=True)
        
        

elif page == "📊 Analytics":
    st.header("📊 **Fraud Analytics Dashboard**")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    df = st.session_state.dataset
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 **Total Transactions**", f"{len(df):,}")
    col2.metric("🚨 **Fraud Cases**", int(df['is_fraudulent'].sum()))
    col3.metric("💸 **Fraud Amount**", f"₹{df.loc[df['is_fraudulent']==1, 'transaction_amount'].sum():,.0f}")
    col4.metric("📈 **Fraud Rate**", f"{df['is_fraudulent'].mean():.2%}")
    
    st.markdown("---")
    
    # Generate all analytics charts
    analytics_charts = create_analytics_charts(df, st.session_state.model_result)
    
    # Display charts in grid
    st.markdown("### 📈 **Transaction Analytics**")
    
    for i in range(0, len(analytics_charts), 2):
        col1, col2 = st.columns(2)
        with col1:
            if i < len(analytics_charts):
                st.plotly_chart(analytics_charts[i][1], use_container_width=True)
        with col2:
            if i + 1 < len(analytics_charts):
                st.plotly_chart(analytics_charts[i + 1][1], use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Model Performance":
    st.header("📈 **Model Performance Dashboard**")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ **Please train a model first to view performance metrics!**")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    model_result = st.session_state.model_result
    
    # Model info
    st.markdown(f"### 🤖 **Model: {model_result['model_type']}**")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📈 **AUC-ROC**", f"{model_result['auc']:.3f}")
    col2.metric("🎯 **Accuracy**", f"{model_result['accuracy']:.3f}")
    col3.metric("📊 **Precision**", f"{model_result['precision']:.3f}")
    col4.metric("🔍 **Recall**", f"{model_result['recall']:.3f}")
    col5.metric("⚖️ **F1 Score**", f"{model_result['f1']:.3f}")
    
    st.markdown("---")
    
    # Classification Report
    st.markdown("### 📋 **Classification Report**")
    report = classification_report(model_result['y_test'], model_result['y_pred'], 
                                  target_names=['Legitimate', 'Fraud'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Performance charts
    performance_charts = create_model_performance_charts(model_result)
    
    st.markdown("### 📊 **Performance Visualizations**")
    
    for i in range(0, len(performance_charts), 2):
        col1, col2 = st.columns(2)
        with col1:
            if i < len(performance_charts):
                st.plotly_chart(performance_charts[i][1], use_container_width=True)
        with col2:
            if i + 1 < len(performance_charts):
                st.plotly_chart(performance_charts[i + 1][1], use_container_width=True)
    

# Footer
st.markdown("---")
st.markdown(f"""
<div class="glass-card" style="text-align: center; padding: 2rem; font-size: 1.1rem;">
    <strong>🛡️ Ecommerce Fraud Detection v18.0</strong> | {live_loc} | 
    <strong>XGBoost + RandomForest + Stacking | Analytics</strong>
</div>
""", unsafe_allow_html=True)
