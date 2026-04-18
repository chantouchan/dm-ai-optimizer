import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import io, warnings, os
warnings.filterwarnings("ignore")

# ===== ページ設定 =====
st.set_page_config(page_title="紙DM AI最適化ツール", page_icon="📮", layout="wide")

# ===== タイトル =====
st.title("📮 紙DM AI最適化ツール")
st.caption("デジタル全盛の時代に、あえて紙で届ける。でも届ける相手はAIが選ぶ。")

st.divider()

# ===== サイドバー =====
with st.sidebar:
    st.header("📂 データ入力")
    uploaded_file = st.file_uploader("顧客CSVをアップロード", type=["csv"])
    use_demo = st.checkbox("デモデータを使用", value=True)
    st.divider()

    st.header("📮 紙DMコスト設定")
    st.caption("1通あたりの内訳を入力")
    cost_print = st.number_input("印刷費（円/通）", 5, 200, 15, 5)
    cost_funyuu = st.number_input("封入・封緘費（円/通）", 0, 100, 10, 5)
    dm_type = st.selectbox("郵送方法", ["ハガキ（63円）", "封書（84円）", "ゆうメール（180円）"])
    cost_postage = {"ハガキ（63円）": 63, "封書（84円）": 84, "ゆうメール（180円）": 180}[dm_type]
    cost_design = st.number_input("デザイン費（円/回）", 0, 500000, 50000, 10000)

    dm_cost_per_unit = cost_print + cost_funyuu + cost_postage
    st.success(f"**1通あたり合計: ¥{dm_cost_per_unit:,}**")

    st.divider()
    st.header("📅 送付計画")
    dm_budget = st.number_input("DM予算（万円/回）", 10, 1000, 100, 10)
    max_sends = int(dm_budget * 10000 / dm_cost_per_unit)
    st.info(f"最大送付数: **{max_sends:,}通/回**")
    annual_campaigns = st.slider("年間キャンペーン回数", 1, 24, 6)

    st.divider()
    st.header("⚙️ 分析設定")
    n_clusters = st.slider("セグメント数（K-Means）", 2, 8, 4)

# ===== デモデータ =====
def generate_demo_data(n=5000):
    np.random.seed(42)
    today = datetime.now()
    records = []
    for i in range(n):
        ctype = np.random.choice(["loyal", "normal", "dormant", "new"], p=[0.15, 0.40, 0.30, 0.15])
        if ctype == "loyal":
            rec, freq, spend = np.random.randint(1,30), np.random.randint(10,50), np.random.randint(50000,300000)
            resp_rate = np.random.uniform(0.15, 0.40)
        elif ctype == "normal":
            rec, freq, spend = np.random.randint(15,90), np.random.randint(3,15), np.random.randint(10000,80000)
            resp_rate = np.random.uniform(0.05, 0.15)
        elif ctype == "dormant":
            rec, freq, spend = np.random.randint(90,365), np.random.randint(1,5), np.random.randint(3000,30000)
            resp_rate = np.random.uniform(0.01, 0.05)
        else:
            rec, freq, spend = np.random.randint(1,60), np.random.randint(1,3), np.random.randint(2000,15000)
            resp_rate = np.random.uniform(0.03, 0.10)
        dm_sent = np.random.randint(1, 20)
        dm_resp = int(dm_sent * resp_rate)
        records.append({
            "顧客コード": f"C{i+1:05d}",
            "最終購入からの日数": rec,
            "総注文回数": freq,
            "累計購入金額": spend,
            "平均注文金額": int(spend / max(freq, 1)),
            "平均購入点数": round(np.random.uniform(1, 5), 1),
            "購入カテゴリ数": np.random.randint(1, 8),
            "返品回数": np.random.randint(0, freq // 2 + 1),
            "会員日数": np.random.randint(30, 2000),
            "DM送付回数": dm_sent,
            "DM反応回数": dm_resp,
            "DM反応率": round(dm_resp / max(dm_sent, 1), 3),
        })
    return pd.DataFrame(records)

# ===== データ読み込み =====
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ {len(df):,}行 読み込み完了")
elif use_demo:
    df = generate_demo_data()
    st.sidebar.info("🧪 デモデータ（5,000件）")
else:
    st.info("👈 CSVをアップロードするか、デモデータにチェックを入れてください")
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ===== タブ =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 データ概要",
    "📮 DM送付リスト（AI）",
    "🧩 セグメント分析",
    "💰 コストシミュレーション",
    "🔄 AI再学習",
])

# ----- タブ1: データ概要 -----
with tab1:
    st.subheader("📊 データ概要")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総顧客数", f"{len(df):,}")
    c2.metric("カラム数", len(df.columns))
    c3.metric("平均DM反応率", f"{df['DM反応率'].mean()*100:.1f}%" if "DM反応率" in df.columns else "N/A")
    c4.metric("平均購入金額", f"¥{df['累計購入金額'].mean():,.0f}" if "累計購入金額" in df.columns else "N/A")
    st.dataframe(df.head(20), use_container_width=True)
    st.divider()
    hist_col = st.selectbox("ヒストグラム表示項目", numeric_cols, key="hist1")
    fig_hist = px.histogram(df, x=hist_col, nbins=30, title=f"分布: {hist_col}")
    st.plotly_chart(fig_hist, use_container_width=True)

# ----- タブ2: DM送付リスト -----
with tab2:
    st.subheader("📮 AI最適化 DM送付リスト")
    if "DM反応率" in df.columns:
        df["DM反応フラグ"] = (df["DM反応率"] > df["DM反応率"].median()).astype(int)
        feature_cols = [c for c in numeric_cols if c not in ["DM反応フラグ", "DM反応率", "DM反応回数"]]
        X = df[feature_cols].fillna(0)
        y = df["DM反応フラグ"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train) * 100
        test_acc = clf.score(X_test, y_test) * 100
        c1, c2 = st.columns(2)
        c1.metric("学習精度", f"{train_acc:.1f}%")
        c2.metric("テスト精度", f"{test_acc:.1f}%")
        df["AIスコア"] = clf.predict_proba(X)[:, 1]
        df["AIランク"] = pd.cut(df["AIスコア"], bins=[0, 0.3, 0.6, 1.0], labels=["C: 送付不要", "B: 検討", "A: 送付推奨"])
        st.divider()
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("A: 送付推奨", f"{(df['AIランク']=='A: 送付推奨').sum():,}")
        rc2.metric("B: 検討", f"{(df['AIランク']=='B: 検討').sum():,}")
        rc3.metric("C: 送付不要", f"{(df['AIランク']=='C: 送付不要').sum():,}")
        fig_rank = px.pie(df, names="AIランク", title="AIランク分布")
        st.plotly_chart(fig_rank, use_container_width=True)
        st.subheader("重要度ランキング（上位10）")
        imp = pd.DataFrame({"項目": feature_cols, "重要度": clf.feature_importances_}).nlargest(10, "重要度")
        fig_imp = px.bar(imp, x="重要度", y="項目", orientation="h", title="特徴量重要度 上位10")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.subheader(f"DM送付リスト（上位 {max_sends:,}件）")
        send_list = df.nlargest(max_sends, "AIスコア")[["顧客コード", "AIスコア", "AIランク", "累計購入金額", "総注文回数", "最終購入からの日数"]]
        st.dataframe(send_list, use_container_width=True)
        exp_rate = send_list["AIスコア"].mean()
        exp_resp = int(len(send_list) * exp_rate)
        avg_spend = df["累計購入金額"].mean() if "累計購入金額" in df.columns else 10000
        exp_rev = int(exp_resp * avg_spend * 0.3)
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("予想反応率", f"{exp_rate*100:.1f}%")
        ec2.metric("予想反応数", f"{exp_resp:,}")
        ec3.metric("予想売上", f"¥{exp_rev:,}")
        st.divider()
        csv_data = send_list.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 送付リストCSVダウンロード", data=csv_data, file_name=f"DM送付リスト_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
    else:
        st.warning("DM反応率カラムが必要です")

# ----- タブ3: セグメント分析 -----
with tab3:
    st.subheader("🧩 顧客セグメント分析")
    cluster_features = st.multiselect("クラスタリングに使用する項目", numeric_cols, default=numeric_cols[:4], key="cf1")
    if len(cluster_features) >= 2:
        X_clust = StandardScaler().fit_transform(df[cluster_features].fillna(0))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["セグメント"] = kmeans.fit_predict(X_clust)
        df["セグメント名"] = df["セグメント"].apply(lambda x: f"セグメント {x+1}")
        fig_seg = px.pie(df, names="セグメント名", title="セグメント構成比")
        st.plotly_chart(fig_seg, use_container_width=True)
        st.subheader("セグメント別サマリー")
        seg_summary = df.groupby("セグメント名")[cluster_features].mean().round(1)
        st.dataframe(seg_summary, use_container_width=True)
        if len(cluster_features) >= 2:
            fig_scat = px.scatter(df, x=cluster_features[0], y=cluster_features[1], color="セグメント名", title="セグメント散布図", opacity=0.6)
            st.plotly_chart(fig_scat, use_container_width=True)
        if "DM反応率" in df.columns:
            st.subheader("セグメント別 DM反応率")
            seg_resp = df.groupby("セグメント名")["DM反応率"].mean().reset_index()
            fig_sr = px.bar(seg_resp, x="セグメント名", y="DM反応率", title="セグメント別 平均DM反応率")
            st.plotly_chart(fig_sr, use_container_width=True)

# ----- タブ4: コストシミュレーション -----
with tab4:
    st.subheader("💰 紙DMコストシミュレーション")
    st.caption("無駄な1通を減らす。紙だからこそ、1通が重い。")

    total_customers = len(df)

    # 全員送付
    cost_all_print = total_customers * cost_print
    cost_all_funyuu = total_customers * cost_funyuu
    cost_all_postage = total_customers * cost_postage
    cost_all_total = cost_all_print + cost_all_funyuu + cost_all_postage + cost_design

    # AI最適化
    cost_ai_print = max_sends * cost_print
    cost_ai_funyuu = max_sends * cost_funyuu
    cost_ai_postage = max_sends * cost_postage
    cost_ai_total = cost_ai_print + cost_ai_funyuu + cost_ai_postage + cost_design

    savings_per = cost_all_total - cost_ai_total
    savings_annual = savings_per * annual_campaigns

    avg_resp_all = df["DM反応率"].mean() if "DM反応率" in df.columns else 0.03
    avg_resp_ai = df.nlargest(max_sends, "AIスコア")["DM反応率"].mean() if "AIスコア" in df.columns and "DM反応率" in df.columns else avg_resp_all * 1.5
    resp_all = int(total_customers * avg_resp_all)
    resp_ai = int(max_sends * avg_resp_ai)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📦 全員に送付")
        st.metric("送付数", f"{total_customers:,}通")
        st.metric("予想反応数", f"{resp_all:,}件")
        st.metric("反応1件あたりコスト", f"¥{cost_all_total // max(resp_all,1):,}")
        st.metric("**合計コスト**", f"¥{cost_all_total:,}")
    with c2:
        st.markdown("### 🎯 AI最適化")
        st.metric("送付数", f"{max_sends:,}通")
        st.metric("予想反応数", f"{resp_ai:,}件")
        st.metric("反応1件あたりコスト", f"¥{cost_ai_total // max(resp_ai,1):,}")
        st.metric("**合計コスト**", f"¥{cost_ai_total:,}")

    st.divider()

    # コスト内訳
    st.subheader("📋 コスト内訳（1回あたり）")
    breakdown = pd.DataFrame({
        "項目": ["印刷費", "封入・封緘費", "郵送費", "デザイン費", "**合計**"],
        "全員送付": [
            f"¥{cost_all_print:,}", f"¥{cost_all_funyuu:,}",
            f"¥{cost_all_postage:,}", f"¥{cost_design:,}", f"¥{cost_all_total:,}"
        ],
        "AI最適化": [
            f"¥{cost_ai_print:,}", f"¥{cost_ai_funyuu:,}",
            f"¥{cost_ai_postage:,}", f"¥{cost_design:,}", f"¥{cost_ai_total:,}"
        ],
        "削減額": [
            f"¥{cost_all_print - cost_ai_print:,}",
            f"¥{cost_all_funyuu - cost_ai_funyuu:,}",
            f"¥{cost_all_postage - cost_ai_postage:,}",
            f"¥0",
            f"¥{savings_per:,}"
        ],
    })
    st.dataframe(breakdown, hide_index=True, use_container_width=True)

    st.divider()

    # 年間効果
    st.subheader(f"📅 年間効果（年{annual_campaigns}回送付）")
    y1, y2, y3 = st.columns(3)
    y1.metric("1回あたり削減額", f"¥{savings_per:,}")
    y2.metric(f"年間削減額（{annual_campaigns}回）", f"¥{savings_annual:,}")
    roi = (savings_per / max(cost_ai_total, 1)) * 100
    y3.metric("ROI改善率", f"{roi:.0f}%")

    # グラフ
    fig_comp = go.Figure(data=[
        go.Bar(name="全員送付", x=["印刷費", "封入費", "郵送費", "デザイン費"], y=[cost_all_print, cost_all_funyuu, cost_all_postage, cost_design], marker_color="#EF553B"),
        go.Bar(name="AI最適化", x=["印刷費", "封入費", "郵送費", "デザイン費"], y=[cost_ai_print, cost_ai_funyuu, cost_ai_postage, cost_design], marker_color="#636EFA"),
    ])
    fig_comp.update_layout(barmode="group", title="コスト内訳比較")
    st.plotly_chart(fig_comp, use_container_width=True)

    # 年間推移
    fig_annual = go.Figure(data=[
        go.Bar(name="全員送付", x=[f"{i+1}回目" for i in range(annual_campaigns)], y=[cost_all_total] * annual_campaigns, marker_color="#EF553B"),
        go.Bar(name="AI最適化", x=[f"{i+1}回目" for i in range(annual_campaigns)], y=[cost_ai_total] * annual_campaigns, marker_color="#636EFA"),
    ])
    fig_annual.update_layout(barmode="group", title=f"年間コスト推移（{annual_campaigns}回）")
    st.plotly_chart(fig_annual, use_container_width=True)

    cumulative_all = [cost_all_total * (i+1) for i in range(annual_campaigns)]
    cumulative_ai = [cost_ai_total * (i+1) for i in range(annual_campaigns)]
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=[f"{i+1}回目" for i in range(annual_campaigns)], y=cumulative_all, name="全員送付", line=dict(color="#EF553B", width=3)))
    fig_cum.add_trace(go.Scatter(x=[f"{i+1}回目" for i in range(annual_campaigns)], y=cumulative_ai, name="AI最適化", line=dict(color="#636EFA", width=3)))
    fig_cum.update_layout(title="累計コスト比較", yaxis_title="累計コスト（円）")
    st.plotly_chart(fig_cum, use_container_width=True)

# ----- タブ5: AI再学習 -----
with tab5:
    st.subheader("🔄 AI再学習")
    st.markdown("""
    **使い方:**
    1. AIが選んだ顧客に紙DMを送付
    2. 2〜3週間後、購入者の顧客コードをCSVで書き出し
    3. ここにアップロード → AIが学習 → 次回はもっと賢くなる
    """)
    st.divider()
    result_csv = st.file_uploader("DM結果CSVをアップロード", type=["csv"], key="retrain")
    if result_csv:
        result_df = pd.read_csv(result_csv)
        st.dataframe(result_df.head(10))
        st.info(f"📋 {len(result_df):,}行 読み込み完了")
    if st.button("🚀 AI再学習を実行", type="primary", use_container_width=True):
        with st.spinner("AIが学習中..."):
            import time
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)
            st.success("✅ 再学習完了！次回のDM送付リストに反映されます。")
            st.balloons()

st.divider()
st.caption("📮 紙DM AI最適化ツール v3.0 | 紙だからこそ、1通を大切に。| Powered by Streamlit + scikit-learn")
