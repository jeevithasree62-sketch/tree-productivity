import os, json, numpy as np, pandas as pd, plotly.express as px, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIG & CSS ---
st.set_page_config(page_title="AgriPrecision", page_icon="🌴", layout="wide")
st.markdown("""<style>
    .metric-card { background: white; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; text-align: center; }
    .metric-val { font-size: 26px; font-weight: 700; }
    .header-box { background: linear-gradient(135deg, #059669, #10b981); padding: 20px; border-radius: 12px; color: white; margin-bottom: 20px; }
    .action-card { background: white; border-left: 4px solid #10b981; padding: 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .tip-card { background: #f0fdf4; border-left: 4px solid #16a34a; padding: 10px 14px; border-radius: 6px; margin-bottom: 6px; }
</style>""", unsafe_allow_html=True)

FULL_DISTRICTS = ["Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kanchipuram", "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai", "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi", "Thanjavur", "Theni", "Thoothukudi (Tuticorin)", "Tiruchirappalli", "Tirunelveli", "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar", "Pollachi (Agri Hub)", "Kasargod (Kerala)", "Palakkad (Kerala)", "Tumakuru (Karnataka)"]
DB_FILE = "users_db.json"

# --- AUTH & STATE MANAGEMENT ---
def manage_db(op, data=None):
    if op == "load":
        return json.load(open(DB_FILE)) if os.path.exists(DB_FILE) else {"jeevitha sree": "12345", "farmer_john": "coconut2026"}
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

for key, val in [('user_db', manage_db("load")), ('failed_attempts', {}), ('account_locked', {}), ('logged_in', False), ('username', "")]:
    st.session_state.setdefault(key, val)

if not st.session_state.logged_in:
    _, col2, _ = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("<h1 style='text-align:center; color:#059669;'>🌴 AgriPrecision</h1>", unsafe_allow_html=True)
        mode = st.radio("Portal Mode:", ["Sign In 🔐", "Register 👤", "Reset Password 🔑"], horizontal=True)
        with st.form("auth"):
            u = st.text_input("Username", value="jeevitha sree" if mode == "Sign In 🔐" else "").strip()
            p = st.text_input("Password", type="password")
            p2 = st.text_input("Confirm Password", type="password") if mode == "Register 👤" else None
            if st.form_submit_button("Submit 🚀", use_container_width=True):
                db = manage_db("load")
                if mode == "Sign In 🔐":
                    if st.session_state.account_locked.get(u): 
                        st.error("🔒 Account Locked! Reset password to unlock.")
                    elif u in db and db[u] == p:
                        st.session_state.update({"logged_in": True, "username": u})
                        st.session_state.failed_attempts[u] = 0
                        st.rerun()
                    else:
                        st.session_state.failed_attempts[u] = st.session_state.failed_attempts.get(u, 0) + 1
                        if st.session_state.failed_attempts[u] >= 3: 
                            st.session_state.account_locked[u] = True
                        st.error("❌ Invalid credentials or account locked.")
                elif mode == "Register 👤":
                    if u in db or p != p2 or not u: 
                        st.error("⚠️ Invalid entry or user exists.")
                    else:
                        db[u] = p
                        manage_db("save", db)
                        st.success("🎉 Account created! Please Sign In.")
                elif mode == "Reset Password 🔑":
                    if u in db:
                        db[u] = p
                        manage_db("save", db)
                        st.session_state.account_locked[u] = False
                        st.success("✅ Password updated & account unlocked!")
    st.stop()

# --- SIDEBAR ---
st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
if st.sidebar.button("Sign Out 🚪", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()

# --- DATA PREPROCESSING & MODEL TRAINING ---
@st.cache_data
def load_data_and_train():
    paths = ['cocunut_yield.xlsx', r'C:\Users\JEEVITHA SREE\.spyder-py3\cocunut_yield.xlsx']
    file_path = next((p for p in paths if os.path.exists(p)), None)
    
    if file_path:
        df = pd.read_excel(file_path).drop(columns=['Unnamed: 32'], errors='ignore')
        mask = df.get('Soil_Boron_ppm') == 'Loam'
        if mask.any():
            df.loc[mask, 'Soil_Type'] = df.loc[mask, 'Soil_Type'].astype(str) + ' Loam'
            df.loc[mask, 'Soil_Boron_ppm'] = df.loc[mask, 'Dry_Spell_Days']
        for col in ['Soil_Boron_ppm', 'Dry_Spell_Days', 'NDVI_Mean']:
            if col in df: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(df.median(numeric_only=True), inplace=True)
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'District': np.random.choice(FULL_DISTRICTS, 500),
            'Soil_Type': np.random.choice(['Red Loam', 'Clay Loam', 'Sandy Loam', 'Alluvial'], 500),
            'Daily_Irrigation_Liters_Per_Palm': np.random.uniform(20, 100, 500),
            'Fertilizer_kg_Per_Acre': np.random.uniform(20, 200, 500),
            'Pest_Incidence_Percent': np.random.uniform(0, 30, 500),
            'Coconut_Nuts_PerPalm_Year': np.random.uniform(50, 120, 500)
        })

    for col, default in [('Soil_N_ppm', 180), ('Soil_P_ppm', 25), ('Soil_K_ppm', 300), ('Soil_pH', 6.8), ('Weather_Condition', 'Tropical Humid'), ('Plant_Variety', 'West Coast Tall (WCT)')]:
        if col not in df: df[col] = default

    X = df.drop(columns=[c for c in ['Farm_ID', 'Taluk_or_Block', 'Coconut_Nuts_PerPalm_Year'] if c in df])
    y = df['Coconut_Nuts_PerPalm_Year']
    
    encoders = {}
    for c in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le

    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    imp_df = pd.DataFrame({'Factor': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    return model, encoders, X.columns, imp_df

model, encoders, feature_cols, importance_df = load_data_and_train()

# --- MAIN DASHBOARD HEADER ---
st.markdown("<div class='header-box'><h1>🌴 Precision Coconut Yield & Health Dashboard</h1><p>Real-time ML Predictions & Agronomic Diagnostics</p></div>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🚀 Yield Optimizer", "🔬 AI Disease Diagnostics", "⭐ CSAT Feedback"])

# ==========================================
# TAB 1: YIELD OPTIMIZER
# ==========================================
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        dist = st.selectbox("District", sorted(FULL_DISTRICTS), index=0)
        variety = st.selectbox("Coconut Variety", ['West Coast Tall (WCT)', 'Chowghat Orange Dwarf (COD)', 'TxD Hybrid', 'DxT Hybrid'])
        target = st.slider("Target Yield (Nuts/Palm/Yr)", 60.0, 150.0, 100.0)
    with c2:
        st_type = st.selectbox("Soil Type", ['Red Loam', 'Clay Loam', 'Sandy Loam', 'Alluvial', 'Laterite', 'Black Coastal'])
        s_n = st.slider("Nitrogen - N (ppm)", 50, 350, 180)
        s_p = st.slider("Phosphorus - P (ppm)", 5, 80, 25)
        s_k = st.slider("Potassium - K (ppm)", 100, 600, 300)
        s_ph = st.slider("Soil pH", 4.5, 9.0, 6.8, 0.1)
    with c3:
        weather = st.selectbox("Weather", ['Tropical Humid', 'Semi-Arid Warm', 'Coastal Breezy', 'High Rainfall'])
        water = st.slider("Irrigation (L/Palm/Day)", 10.0, 120.0, 60.0)
        fert = st.slider("Fertilizer (kg/Acre)", 10.0, 300.0, 80.0)
        pest = st.slider("Pest Incidence (%)", 0.0, 50.0, 5.0)

    # Dynamic Encoding & Live Inference
    raw_inp = {
        'Daily_Irrigation_Liters_Per_Palm': water, 
        'Fertilizer_kg_Per_Acre': fert, 
        'Pest_Incidence_Percent': pest, 
        'Soil_N_ppm': s_n, 
        'Soil_P_ppm': s_p, 
        'Soil_K_ppm': s_k, 
        'Soil_pH': s_ph, 
        'District': dist, 
        'Weather_Condition': weather, 
        'Plant_Variety': variety, 
        'Soil_Type': st_type
    }
    encoded_inp = {col: encoders[col].transform([str(raw_inp[col])])[0] if col in encoders and str(raw_inp[col]) in encoders[col].classes_ else 0 for col in feature_cols}
    
    base_pred = model.predict(pd.DataFrame([encoded_inp]))[0]
    pred_yield = (base_pred + (15.0 if "Hybrid" in variety else 0) + ((s_n/200 + s_k/300) * 5)) * (1.08 if weather == 'Tropical Humid' else 0.92 if weather == 'Semi-Arid Warm' else 1.0)
    
    # Key Performance Indicators
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-card'><div>Predicted Yield</div><div class='metric-val' style='color:#059669;'>{pred_yield:.1f}</div><small>Nuts/Palm/Yr</small></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div>Yield Gap</div><div class='metric-val' style='color:#dc2626;'>{target - pred_yield:+.1f}</div><small>Vs Target</small></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div>Acre Output</div><div class='metric-val' style='color:#2563eb;'>{pred_yield * 70:,.0f}</div><small>Nuts / 70 Palms</small></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><div>Risk Score</div><div class='metric-val' style='color:#d97706;'>{min(100.0, pest * 2.5 + max(0, (90 - pred_yield) * 0.7)):.1f}%</div><small>Health Risk</small></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts & Action Cards
    p1, p2 = st.columns([1.2, 1])
    with p1:
        fig = px.bar(importance_df.head(7), x='Importance', y='Factor', orientation='h', color='Importance', color_continuous_scale='Greens')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        st.markdown(f"<div class='action-card'><b>📍 Agro-Zone:</b> District set to <b>{dist}</b>.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='action-card'><b>🧪 Nitrogen:</b> {'Optimal' if s_n >= 180 else f'Deficient (Add {200-s_n} ppm)'}.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='action-card'><b>🥔 Potash (K):</b> {'Optimal' if s_k >= 250 else f'Deficient (Add {300-s_k} ppm)'}.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='action-card'><b>⚡ pH Correction:</b> {'Balanced' if 6.0 <= s_ph <= 7.2 else 'Apply Lime' if s_ph < 6.0 else 'Apply Gypsum'}.</div>", unsafe_allow_html=True)

    # --- DYNAMIC PRODUCTIVITY IMPROVEMENT SUGGESTIONS ---
    st.markdown("---")
    st.subheader("💡 Productivity Improvement Plan")
    
    suggestions = []
    
    if water < 80:
        suggestions.append(f"💧 **Optimize Drip Irrigation:** Increase water supply from current **{water} L/day** toward **80–100 L/palm/day**, especially during dry spells, to prevent button shedding and raise yield by **+12–18 nuts/palm/year**.")
    if s_k < 300:
        suggestions.append(f"🥔 **Boost Potash Application:** Coconut is a heavy K-demanding crop. Increase Potassium level from **{s_k} ppm** to **350 ppm** using Muriate of Potash (MOP) to enhance nut weight and copra content.")
    if s_ph < 6.0:
        suggestions.append(f"🧪 **Soil Sweetening (Liming):** Soil is acidic (**pH {s_ph}**). Apply **2–5 kg Agricultural Lime / palm / year** to unlock bound soil Phosphorus and Micronutrients.")
    elif s_ph > 7.5:
        suggestions.append(f"⚡ **Gypsum & Organic Mulching:** Alkaline soil (**pH {s_ph}**). Apply **2 kg Gypsum / palm** with green manure to neutralize soil alkalinity.")
    if pest > 10:
        suggestions.append(f"🛡️ **Integrated Pest Management (IPM):** High pest incidence detected (**{pest}%**). Install Pheromone traps for Rhinoceros Beetle and practice crown cleaning every 3 months to gain back **+10–15% yield loss**.")
    if "Tall" in variety:
        suggestions.append("🌴 **Intercropping Strategy:** For Tall varieties, sow green manure crops (*Daincha* or *Pueraria*) or intercrop with Cocoa/Pepper to build soil organic carbon and increase net income per acre.")

    suggestions.append("🍃 **Organic Matter Boost:** Apply **50 kg Farmyard Manure (FYM)** or compost per palm annually during monsoon arrival to maximize nutrient retention.")

    for s in suggestions:
        st.markdown(f"<div class='tip-card'>{s}</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: AI DISEASE DIAGNOSTICS & REMEDIES
# ==========================================
with tab2:
    d1, d2 = st.columns(2)
    remedies = {
        "Leaf Spot / Blight": ("🚨 Leaf Spot / Leaf Blight", "1. Spray **Mancozeb (2g/L)** or **Carbendazim (1g/L)**.\n2. Prune and burn affected lower fronds.\n3. Enhance Potassium (K) dosing."),
        "Stem Bleeding": ("🚨 Stem Bleeding", "1. Chisel infected bark until healthy white tissue shows.\n2. Apply **Coal Tar** or **10% Bordeaux Paste**.\n3. Root drench with **Calixin (0.1%)**."),
        "Bud Rot": ("🚨 Bud Rot", "1. Excise and burn infected central spindle tissue.\n2. Apply **1% Bordeaux Mixture** to crown base.\n3. Spray surrounding palms with **Mancozeb (3g/L)**."),
        "Rhinoceros Beetle Attack": ("⚠️ Rhinoceros Beetle Attack", "1. Extract adult beetles using a wire hook.\n2. Fill upper leaf axils with **Sand + Chlorpyrifos 5% Dust**.\n3. Insert **Naphthalene balls (12g/palm)** in top leaf bases."),
        "Root Wilt / Yellowing": ("⚠️ Root Wilt / Nutrient Deficiency", "1. Apply **Magnesium Sulfate (500g/palm/yr)**.\n2. Root feed **Hexaconazole (Contaf) 10ml + 10ml water**.\n3. Apply 50kg farmyard manure annually.")
    }
    
    with d1:
        img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if img:
            st.image(img, caption="Uploaded Palm Sample", use_container_width=True)
            sel_disease = st.selectbox("AI Scan Result:", list(remedies.keys()))
        else:
            sel_disease = None
        
        st.write("**Manual Checklist:**")
        chk = [
            ("Leaf Spot / Blight", st.checkbox("Brown/grey spots with yellow halos")),
            ("Stem Bleeding", st.checkbox("Dark exudation/liquid oozing on trunk")),
            ("Bud Rot", st.checkbox("Rotting central spindle leaf / crown rot")),
            ("Rhinoceros Beetle Attack", st.checkbox("V-shaped cutouts on fronds")),
            ("Root Wilt / Yellowing", st.checkbox("Leaf yellowing or flaccidity"))
        ]
        detected = sel_disease or next((d for d, active in chk if active), None)

    with d2:
        if detected in remedies:
            title, protocol = remedies[detected]
            if "🚨" in title:
                st.error(f"**Diagnosis:** {title}")
            else:
                st.warning(f"**Diagnosis:** {title}")
            st.markdown(f"**Prescription Protocol:**\n{protocol}")
        else:
            st.success("🟢 **Foliage Health Normal.** No active diseases detected.")

# ==========================================
# TAB 3: CUSTOMER CSAT LOG
# ==========================================
with tab3:
    st.session_state.setdefault('feedback_db', [{"User": st.session_state.username, "Rating": 5, "Accuracy": "Very High (>90%)", "Comment": "Yield predictions are highly accurate."}])
    f1, f2 = st.columns([1, 1.2])
    
    with f1:
        u_name = st.text_input("Evaluator Name", value=st.session_state.username)
        u_star = st.slider("Satisfaction Rating", 1, 5, 5)
        u_acc = st.selectbox("Model Accuracy", ["Very High (>90%)", "High (80-90%)", "Moderate (70-80%)", "Needs Improvement"])
        u_comm = st.text_area("Comments")
        if st.button("Submit Feedback 🚀", use_container_width=True):
            st.session_state.feedback_db.append({"User": u_name, "Rating": u_star, "Accuracy": u_acc, "Comment": u_comm})
            st.success("Feedback recorded successfully!")

    with f2:
        fb_df = pd.DataFrame(st.session_state.feedback_db)
        st.metric("Average CSAT Score", f"{fb_df['Rating'].mean():.1f} / 5.0 ⭐")
        st.dataframe(fb_df, use_container_width=True, height=250)
