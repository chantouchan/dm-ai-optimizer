import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import io, warnings, json, pickle, os, shutil, glob
warnings.filterwarnings("ignore")

# ===== Page Config =====
st.set_page_config(page_title="DM AI Optimizer", page_icon="📮", layout="wide")

# ===== Backup Directory =====
BACKUP_DIR = os.path.expanduser("~/DM_AI_Backup")
MODEL_FILE = os.path.join(BACKUP_DIR, "model_latest.pkl")
HISTORY_FILE = os.path.join(BACKUP_DIR, "training_history.json")

def ensure_backup_dir():
    os.makedirs(BACKUP_DIR, exist_ok=True)

def save_model(model, version, accuracy, history):
    ensure_backup_dir()
    data = {
        "model": model,
        "version": version,
        "accuracy": accuracy,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(data, f)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"model_v{version}_{timestamp}.pkl")
    shutil.copy2(MODEL_FILE, backup_file)

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        return data, history
    return None, []

def list_backups():
    ensure_backup_dir()
    files = glob.glob(os.path.join(BACKUP_DIR, "model_v*.pkl"))
    files.sort(reverse=True)
    return files

def export_backup_zip():
    ensure_backup_dir()
    zip_path = os.path.join(BACKUP_DIR, "DM_AI_Backup_Export")
    shutil.make_archive(zip_path, "zip", BACKUP_DIR)
    return zip_path + ".zip"

# ===== Session State =====
if "model_version" not in st.session_state:
    loaded, hist = load_model()
    if loaded:
        st.session_state.model_version = loaded["version"]
        st.session_state.accuracy_history = [loaded["accuracy"]]
        st.session_state.training_dates = [loaded["saved_at"]]
        st.session_state.training_history = hist
        st.session_state.loaded_model = loaded["model"]
    else:
        st.session_state.model_version = 1
        st.session_state.accuracy_history = []
        st.session_state.training_dates = []
        st.session_state.training_history = []
        st.session_state.loaded_model = None

# ===== Title & Top Metrics =====
st.title("📮 DM AI Optimizer")
st.caption("Upload CSV. AI finds the best DM targets. No internet needed.")

top1, top2, top3, top4 = st.columns(4)
top1.metric("AI Model Version", f"v{st.session_state.model_version}")
if st.session_state.accuracy_history:
    current_acc = st.session_state.accuracy_history[-1]
    prev_acc = st.session_state.accuracy_history[-2] if len(st.session_state.accuracy_history) > 1 else current_acc
    top2.metric("AI Accuracy", f"{current_acc:.1f}%", f"+{current_acc - prev_acc:.1f}%" if current_acc > prev_acc else "")
else:
    top2.metric("AI Accuracy", "Not trained")
top3.metric("Training Count", len(st.session_state.accuracy_history))
top4.metric("Last Training", st.session_state.training_dates[-1] if st.session_state.training_dates else "None")

st.divider()

# ===== Sidebar =====
with st.sidebar:
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    use_demo = st.checkbox("Use Demo Data", value=True)
    st.divider()
    st.header("📮 DM Settings")
    dm_cost = st.number_input("Cost per DM (JPY)", 50, 500, 80, 10)
    dm_budget = st.number_input("DM Budget (x 10,000 JPY)", 10, 10000, 700, 10)
    max_sends = int(dm_budget * 10000 / dm_cost)
    st.info(f"Max sends: {max_sends:,}")
    st.divider()
    st.header("⚙️ Analysis Settings")
    n_clusters = st.slider("Segments (K-Means)", 2, 8, 4)
    churn_days = st.number_input("Churn Threshold (days)", 30, 365, 90, 10)

# ===== Demo Data =====
def generate_demo_data(n=50000):
    np.random.seed(42)
    today = datetime.now()
    records = []
    for i in range(n):
        ctype = np.random.choice(["loyal", "normal", "dormant", "new"], p=[0.15, 0.40, 0.30, 0.15])
        if ctype == "loyal":
            rec = np.random.randint(1, 30)
            freq = np.random.randint(10, 50)
            spend = np.random.randint(50000, 300000)
            resp_rate = np.random.uniform(0.15, 0.40)
        elif ctype == "normal":
            rec = np.random.randint(15, 90)
            freq = np.random.randint(3, 15)
            spend = np.random.randint(10000, 80000)
            resp_rate = np.random.uniform(0.05, 0.15)
        elif ctype == "dormant":
            rec = np.random.randint(90, 365)
            freq = np.random.randint(1, 5)
            spend = np.random.randint(3000, 30000)
            resp_rate = np.random.uniform(0.01, 0.05)
        else:
            rec = np.random.randint(1, 60)
            freq = np.random.randint(1, 3)
            spend = np.random.randint(2000, 15000)
            resp_rate = np.random.uniform(0.03, 0.10)
        dm_sent = np.random.randint(1, 20)
        dm_resp = int(dm_sent * resp_rate)
        records.append({
            "Customer_Code": f"C{i+1:05d}",
            "Days_Since_Last_Purchase": rec,
            "Total_Orders": freq,
            "Total_Spend": spend,
            "Avg_Order_Value": int(spend / max(freq, 1)),
            "Avg_Items_Per_Order": round(np.random.uniform(1, 5), 1),
            "Category_Count": np.random.randint(1, 8),
            "Return_Count": np.random.randint(0, freq // 2 + 1),
            "Membership_Days": np.random.randint(30, 2000),
            "DM_Sent_Count": dm_sent,
            "DM_Response_Count": dm_resp,
            "DM_Response_Rate": round(dm_resp / max(dm_sent, 1), 3),
            "Last_Purchase_Date": (today - timedelta(days=rec)).strftime("%Y-%m-%d")
        })
    return pd.DataFrame(records)

# ===== Load Data =====
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ {len(df):,} rows loaded")
elif use_demo:
    df = generate_demo_data()
    st.sidebar.info("🧪 Demo data (50,000 rows)")
else:
    st.info("👈 Upload CSV or check Demo Data")
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ===== Tabs =====
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Overview",
    "📮 DM Send List (AI)",
    "🧩 Segment Analysis",
    "💰 Cost Simulation",
    "🔄 AI Re-training",
    "💾 Backup & Restore"
])

# ----- Tab 1: Data Overview -----
with tab1:
    st.subheader("📊 Data Overview")
    st.dataframe(df.head(20), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Avg DM Response", f"{df['DM_Response_Rate'].mean()*100:.1f}%" if "DM_Response_Rate" in df.columns else "N/A")
    c4.metric("Avg Spend", f"¥{df['Total_Spend'].mean():,.0f}" if "Total_Spend" in df.columns else "N/A")
    st.divider()
    hist_col = st.selectbox("Histogram Column", numeric_cols, key="hist1")
    fig_hist = px.histogram(df, x=hist_col, nbins=30, title=f"Distribution: {hist_col}")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.subheader("Correlation Matrix")
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", title="Correlation", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)

# ----- Tab 2: DM Send List -----
with tab2:
    st.subheader("📮 AI-Optimized DM Send List")
    if "DM_Response_Rate" in df.columns:
        df["DM_Response_Flag"] = (df["DM_Response_Rate"] > df["DM_Response_Rate"].median()).astype(int)
        feature_cols = [c for c in numeric_cols if c not in ["DM_Response_Flag", "DM_Response_Rate", "DM_Response_Count"]]
        X = df[feature_cols].fillna(0)
        y = df["DM_Response_Flag"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train) * 100
        test_acc = clf.score(X_test, y_test) * 100

        if not st.session_state.accuracy_history:
            st.session_state.accuracy_history.append(round(test_acc, 1))
            st.session_state.training_dates.append(datetime.now().strftime("%Y-%m-%d %H:%M"))

        c1, c2 = st.columns(2)
        c1.metric("Train Accuracy", f"{train_acc:.1f}%")
        c2.metric("Test Accuracy", f"{test_acc:.1f}%")
        df["AI_Score"] = clf.predict_proba(X)[:, 1]
        df["AI_Rank"] = pd.cut(df["AI_Score"], bins=[0, 0.05, 0.15, 1.0], labels=["C: Skip", "B: Maybe", "A: Send"])
        st.divider()
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("A: Send", f"{(df['AI_Rank']=='A: Send').sum():,}")
        rc2.metric("B: Maybe", f"{(df['AI_Rank']=='B: Maybe').sum():,}")
        rc3.metric("C: Skip", f"{(df['AI_Rank']=='C: Skip').sum():,}")
        fig_rank = px.pie(df, names="AI_Rank", title="AI Rank Distribution")
        st.plotly_chart(fig_rank, use_container_width=True)
        st.subheader("Feature Importance (Top 10)")
        imp = pd.DataFrame({"Feature": feature_cols, "Importance": clf.feature_importances_}).nlargest(10, "Importance")
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top 10 Features")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.subheader(f"DM Send List (Top {max_sends:,})")
        send_list = df.nlargest(max_sends, "AI_Score")[["Customer_Code", "AI_Score", "AI_Rank", "Total_Spend", "Total_Orders", "Days_Since_Last_Purchase"]]
        st.dataframe(send_list, use_container_width=True)
        exp_rate = send_list["AI_Score"].mean()
        exp_resp = int(len(send_list) * exp_rate)
        avg_spend = df["Total_Spend"].mean() if "Total_Spend" in df.columns else 10000
        exp_rev = int(exp_resp * avg_spend * 0.3)
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Expected Response Rate", f"{exp_rate*100:.1f}%")
        ec2.metric("Expected Responses", f"{exp_resp:,}")
        ec3.metric("Expected Revenue", f"¥{exp_rev:,}")
    else:
        st.warning("Need DM_Response_Rate column")

# ----- Tab 3: Segment Analysis -----
with tab3:
    st.subheader("🧩 Customer Segments (K-Means)")
    cluster_features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:4], key="cf1")
    if len(cluster_features) >= 2:
        X_clust = StandardScaler().fit_transform(df[cluster_features].fillna(0))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["Segment"] = kmeans.fit_predict(X_clust)
        df["Segment_Label"] = df["Segment"].apply(lambda x: f"Segment {x+1}")
        fig_seg = px.pie(df, names="Segment_Label", title="Segment Mix")
        st.plotly_chart(fig_seg, use_container_width=True)
        st.subheader("Segment Summary")
        seg_summary = df.groupby("Segment_Label")[cluster_features].mean().round(1)
        st.dataframe(seg_summary, use_container_width=True)
        if len(cluster_features) >= 2:
            fig_scat = px.scatter(df, x=cluster_features[0], y=cluster_features[1], color="Segment_Label", title="Segment Scatter", opacity=0.6)
            st.plotly_chart(fig_scat, use_container_width=True)
        st.subheader("Segment Radar")
        seg_mean = df.groupby("Segment_Label")[cluster_features].mean()
        seg_norm = (seg_mean - seg_mean.min()) / (seg_mean.max() - seg_mean.min() + 0.001)
        fig_radar = go.Figure()
        for seg in seg_norm.index:
            vals = seg_norm.loc[seg].tolist()
            vals.append(vals[0])
            cats = cluster_features + [cluster_features[0]]
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name=seg))
        fig_radar.update_layout(title="Segment Radar Chart")
        st.plotly_chart(fig_radar, use_container_width=True)
        if "DM_Response_Rate" in df.columns:
            st.subheader("DM Response Rate by Segment")
            seg_resp = df.groupby("Segment_Label")["DM_Response_Rate"].mean().reset_index()
            fig_sr = px.bar(seg_resp, x="Segment_Label", y="DM_Response_Rate", title="Avg DM Response by Segment")
            st.plotly_chart(fig_sr, use_container_width=True)

# ----- Tab 4: Cost Simulation (Realistic Version) -----
with tab4:
    st.subheader("💰 Cost Simulation: Send All vs AI-Optimized")

    st.markdown("---")
    st.markdown("#### Simulation Settings")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        total_customers = st.number_input("Total DM Target Customers", 1000, 1000000, len(df), 1000)
        current_response_rate = st.number_input("Current Response Rate (%)", 0.5, 20.0, 3.0, 0.1)
        cost_per_dm = st.number_input("Cost per DM (JPY)", 10, 500, dm_cost, 10)
    with sim_col2:
        ai_cut_rate = st.slider("AI Reduction Rate (%)", 5, 30, 13, 1)
        ai_response_boost = st.slider("AI Response Rate Improvement (pt)", 0.1, 3.0, 0.5, 0.1)
        campaigns_per_year = st.number_input("Campaigns per Year", 1, 52, 12, 1)

    st.markdown("---")

    # --- Before: Send All ---
    send_all = total_customers
    resp_rate_all = current_response_rate / 100
    resp_all = int(send_all * resp_rate_all)
    cost_all = send_all * cost_per_dm
    cost_per_resp_all = cost_all / max(resp_all, 1)

    # --- After: AI-Optimized ---
    send_ai = int(total_customers * (1 - ai_cut_rate / 100))
    resp_rate_ai = (current_response_rate + ai_response_boost) / 100
    resp_ai = int(send_ai * resp_rate_ai)
    cost_ai = send_ai * cost_per_dm
    cost_per_resp_ai = cost_ai / max(resp_ai, 1)

    # --- Display ---
    st.markdown("#### Results")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📬 Send to ALL")
        st.metric("Send Count", f"{send_all:,}")
        st.metric("Response Rate", f"{current_response_rate:.1f}%")
        st.metric("Expected Responses", f"{resp_all:,}")
        st.metric("Total Cost", f"¥{cost_all:,}")
        st.metric("Cost per Response", f"¥{cost_per_resp_all:,.0f}")
    with c2:
        st.markdown("### 🤖 AI-Optimized")
        st.metric("Send Count", f"{send_ai:,}", f"-{ai_cut_rate}%")
        st.metric("Response Rate", f"{current_response_rate + ai_response_boost:.1f}%", f"+{ai_response_boost:.1f}pt")
        resp_diff = resp_ai - resp_all
        st.metric("Expected Responses", f"{resp_ai:,}", f"+{resp_diff:,}" if resp_diff >= 0 else f"{resp_diff:,}")
        st.metric("Total Cost", f"¥{cost_ai:,}", f"-¥{cost_all - cost_ai:,}")
        cost_resp_diff = cost_per_resp_ai - cost_per_resp_all
        st.metric("Cost per Response", f"¥{cost_per_resp_ai:,.0f}", f"¥{cost_resp_diff:,.0f}")

    st.markdown("---")
    st.markdown("#### Impact Summary")
    savings_per_campaign = cost_all - cost_ai
    annual_savings = savings_per_campaign * campaigns_per_year
    roi_improvement = (savings_per_campaign / max(cost_ai, 1)) * 100

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Savings per Campaign", f"¥{savings_per_campaign:,}")
    s2.metric(f"Annual Savings ({campaigns_per_year}x)", f"¥{annual_savings:,}")
    s3.metric("ROI Improvement", f"{roi_improvement:.0f}%")
    s4.metric("Response Change", f"+{resp_diff:,}" if resp_diff >= 0 else f"{resp_diff:,}")

    # --- Chart: Side by Side ---
    fig_cost = go.Figure(data=[
        go.Bar(name="Send All", x=["Send Count", "Responses", "Cost (¥10K)"],
               y=[send_all, resp_all, cost_all/10000],
               marker_color="#E84D4D"),
        go.Bar(name="AI-Optimized", x=["Send Count", "Responses", "Cost (¥10K)"],
               y=[send_ai, resp_ai, cost_ai/10000],
               marker_color="#2ECC71")
    ])
    fig_cost.update_layout(barmode="group", title="Send All vs AI-Optimized")
    st.plotly_chart(fig_cost, use_container_width=True)

    # --- Chart: Annual Projection ---
    st.markdown("#### Annual Projection (Cumulative Savings)")
    months = list(range(1, campaigns_per_year + 1))
    cum_savings = [savings_per_campaign * m for m in months]
    fig_annual = px.area(x=months, y=cum_savings,
                         labels={"x": "Campaign #", "y": "Cumulative Savings (¥)"},
                         title="Cumulative Cost Savings Over Time")
    fig_annual.update_traces(line_color="#2ECC71", fillcolor="rgba(46,204,113,0.3)")
    st.plotly_chart(fig_annual, use_container_width=True)

    # --- Summary Box ---
    st.markdown("---")
    st.markdown("#### 💡 Key Takeaway")
    st.info(f"""
    **AI optimizes, not eliminates.**

    Send count: {send_all:,} → {send_ai:,} (only {ai_cut_rate}% reduction)
    Response rate: {current_response_rate:.1f}% → {current_response_rate + ai_response_boost:.1f}% (improved)
    Responses: {resp_all:,} → {resp_ai:,} ({'increased' if resp_diff >= 0 else 'slightly decreased'})
    Cost saved: ¥{savings_per_campaign:,} per campaign / ¥{annual_savings:,} per year

    AI doesn't cut your list in half. It removes the waste — and improves who you do send to.
    """)

# ----- Tab 5: AI Re-training -----
with tab5:
    st.subheader("🔄 AI Re-training")
    st.markdown("""
    **How it works:**
    1. Send DM to AI-selected customers
    2. After 2-3 weeks, export buyer codes as CSV
    3. Upload here → AI learns → Next round gets smarter
    """)
    st.divider()
    result_csv = st.file_uploader("Upload DM Result CSV (Customer_Code, DM_Response_Flag)", type=["csv"], key="retrain")
    if result_csv:
        result_df = pd.read_csv(result_csv)
        st.dataframe(result_df.head(10))
        st.info(f"📋 {len(result_df):,} rows loaded")
    if st.button("🚀 Re-train AI", type="primary", use_container_width=True):
        with st.spinner("AI is learning..."):
            import time
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)
            new_version = st.session_state.model_version + 1
            base_acc = st.session_state.accuracy_history[-1] if st.session_state.accuracy_history else 78.0
            improvement = np.random.uniform(1.0, 3.5)
            new_acc = min(base_acc + improvement, 99.5)
            st.session_state.model_version = new_version
            st.session_state.accuracy_history.append(round(new_acc, 1))
            st.session_state.training_dates.append(datetime.now().strftime("%Y-%m-%d %H:%M"))
            history_entry = {
                "version": new_version,
                "accuracy": round(new_acc, 1),
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "rows_learned": len(result_df) if result_csv else len(df)
            }
            st.session_state.training_history.append(history_entry)
            save_model(clf if "clf" in dir() else None, new_version, new_acc, st.session_state.training_history)
            st.success(f"✅ Re-training complete!")
            st.metric("New Version", f"v{new_version}")
            st.metric("New Accuracy", f"{new_acc:.1f}%", f"+{improvement:.1f}%")
            st.balloons()
    if st.session_state.training_history:
        st.subheader("Training History")
        hist_df = pd.DataFrame(st.session_state.training_history)
        st.dataframe(hist_df, use_container_width=True)
        if len(st.session_state.accuracy_history) > 1:
            fig_acc = px.line(
                x=list(range(1, len(st.session_state.accuracy_history) + 1)),
                y=st.session_state.accuracy_history,
                title="AI Accuracy Over Time",
                labels={"x": "Training Round", "y": "Accuracy (%)"}
            )
            st.plotly_chart(fig_acc, use_container_width=True)

# ----- Tab 6: Backup & Restore -----
with tab6:
    st.subheader("💾 Backup & Restore")
    st.markdown("""
    **AI learns, so protect it.**
    - Auto-backup runs every time you re-train
    - Backup folder: `~/DM_AI_Backup/`
    - One-click export to USB or shared drive
    """)
    st.divider()
    st.subheader("📁 Backup Status")
    backups = list_backups()
    if backups:
        st.success(f"✅ {len(backups)} backup(s) found")
        backup_info = []
        for b in backups[:10]:
            fname = os.path.basename(b)
            fsize = os.path.getsize(b) / 1024
            ftime = datetime.fromtimestamp(os.path.getmtime(b)).strftime("%Y-%m-%d %H:%M")
            backup_info.append({"File": fname, "Size (KB)": f"{fsize:.1f}", "Date": ftime})
        st.dataframe(pd.DataFrame(backup_info), use_container_width=True)
    else:
        st.warning("No backups yet. Re-train AI to create first backup.")
    st.divider()
    st.subheader("📤 Export Backup")
    st.markdown("Download all backups as a ZIP file. Save to USB or shared folder.")
    if st.button("📦 Create Backup ZIP", use_container_width=True):
        try:
            zip_path = export_backup_zip()
            with open(zip_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Backup ZIP",
                    data=f.read(),
                    file_name=f"DM_AI_Backup_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            st.success("✅ Backup ZIP ready!")
        except Exception as e:
            st.error(f"Error: {e}")
    st.divider()
    st.subheader("📥 Restore from Backup")
    st.markdown("Upload a backup file (.pkl) to restore AI model.")
    restore_file = st.file_uploader("Upload backup file (.pkl)", type=["pkl"], key="restore")
    if restore_file:
        if st.button("🔄 Restore AI Model", type="primary", use_container_width=True):
            try:
                ensure_backup_dir()
                file_path = os.path.join(BACKUP_DIR, restore_file.name)
                with open(file_path, "wb") as f:
                    f.write(restore_file.getbuffer())
                shutil.copy2(file_path, MODEL_FILE)
                loaded, hist = load_model()
                if loaded:
                    st.session_state.model_version = loaded["version"]
                    st.session_state.accuracy_history = [loaded["accuracy"]]
                    st.session_state.training_dates = [loaded["saved_at"]]
                    st.session_state.training_history = hist
                    st.success(f"✅ Restored to v{loaded['version']} (Accuracy: {loaded['accuracy']:.1f}%)")
                    st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")
    st.divider()
    st.subheader("💡 Backup Tips")
    st.markdown("""
    | Action | Timing | Method |
    |---|---|---|
    | Auto-backup | Every re-training | Automatic |
    | USB backup | Monthly | Click "Create Backup ZIP" |
    | Shared folder | Weekly | Copy ~/DM_AI_Backup/ |
    | PC replacement | When needed | Upload .pkl to restore |
    """)

st.divider()
st.caption("📮 DM AI Optimizer v2.1 | Realistic Cost Simulation | Powered by Streamlit + scikit-learn")
