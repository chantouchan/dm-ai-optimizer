import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, date
import io, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="紙DM AI最適化ツール", page_icon="📮", layout="wide")
st.title("📮 紙DM AI最適化ツール")
st.caption("デジタル全盛の時代に、あえて紙で届ける。でも届ける相手はAIが選ぶ。")

# === デモデータ（10万件） ===
def generate_demo_data(n=100000):
    np.random.seed(42)
    data = pd.DataFrame({
        "顧客ID": [f"C{str(i).zfill(6)}" for i in range(1, n+1)],
        "年齢": np.random.randint(20, 80, n),
        "性別": np.random.choice(["男性","女性"], n),
        "居住地域": np.random.choice(["北海道","東北","関東","中部","近畿","中国","四国","九州"], n),
        "累計購入回数": np.random.poisson(5, n),
        "累計購入金額": np.random.exponential(30000, n).astype(int),
        "最終購入日からの日数": np.random.exponential(120, n).astype(int),
        "過去DM反応回数": np.random.poisson(1.5, n),
        "メルマガ開封率": np.round(np.random.beta(2, 5, n), 2),
        "Web訪問回数_直近30日": np.random.poisson(3, n),
    })
    score = (
        (data["累計購入回数"] > 3).astype(int) * 2
        + (data["最終購入日からの日数"] < 90).astype(int) * 2
        + (data["過去DM反応回数"] > 1).astype(int) * 3
        + (data["メルマガ開封率"] > 0.3).astype(int)
        + (data["Web訪問回数_直近30日"] > 2).astype(int)
    )
    prob = 1 / (1 + np.exp(-0.5 * (score - 4)))
    data["DM反応フラグ"] = (np.random.random(n) < prob).astype(int)
    return data

# === サイドバー ===
st.sidebar.header("📁 データ設定")
use_demo = st.sidebar.checkbox("デモデータを使用（10万件）", value=True)
if use_demo:
    df = generate_demo_data(100000)
    st.sidebar.success("デモデータ（10万件）を読み込みました")
else:
    uploaded = st.sidebar.file_uploader("CSVアップロード", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"{len(df):,}件 読み込みました")
    else:
        st.info("CSVをアップロードするか、デモデータを使用してください。")
        st.stop()

st.sidebar.header("💰 紙DMコスト設定")
st.sidebar.caption("1通あたりの内訳")
cost_print = st.sidebar.number_input("印刷費（円）", value=15, min_value=0, step=1)
cost_enclose = st.sidebar.number_input("封入・封緘費（円）", value=10, min_value=0, step=1)
mail_type = st.sidebar.selectbox("郵送方法", ["ハガキ（63円）","封書（84円）","ゆうメール（180円）"])
cost_postage = {"ハガキ（63円）":63,"封書（84円）":84,"ゆうメール（180円）":180}[mail_type]
cost_design = st.sidebar.number_input("デザイン費（総額・円）", value=50000, min_value=0, step=1000)
cost_per_mail = cost_print + cost_enclose + cost_postage
st.sidebar.metric("1通あたり単価", f"¥{cost_per_mail:,}")
annual_campaigns = st.sidebar.number_input("年間キャンペーン回数", value=4, min_value=1, max_value=24, step=1)

st.sidebar.header("🔬 セグメント設定")
n_clusters = st.sidebar.slider("セグメント数", 2, 8, 4)

# === AI モデル学習 ===
features = ["年齢","累計購入回数","累計購入金額","最終購入日からの日数","過去DM反応回数","メルマガ開封率","Web訪問回数_直近30日"]
X = df[features]
y = df["DM反応フラグ"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
df["AI送付スコア"] = model.predict_proba(X)[:, 1]
df["AI送付対象"] = (df["AI送付スコア"] >= 0.5).astype(int)
ai_target = df[df["AI送付対象"] == 1]

# === タブ ===
tab1, tab2, tab3, tab4 = st.tabs(["📊 データ概要","🎯 AI DM送付リスト","👥 セグメント分析","💴 コストシミュレーション"])

with tab1:
    st.header("📊 データ概要")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("総顧客数", f"{len(df):,}人")
    c2.metric("平均購入回数", f"{df['累計購入回数'].mean():.1f}回")
    c3.metric("平均購入金額", f"¥{df['累計購入金額'].mean():,.0f}")
    c4.metric("DM反応率", f"{df['DM反応フラグ'].mean()*100:.1f}%")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x="年齢", color="性別", nbins=30, title="年齢分布"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(df, x="居住地域", color="DM反応フラグ", title="地域別DM反応"), use_container_width=True)
    st.subheader("データプレビュー（先頭100件）")
    st.dataframe(df.head(100), hide_index=True, height=300)

with tab2:
    st.header("🎯 AI DM送付リスト")
    st.caption("AIが反応しそうな顧客を自動で選別します")
    c1,c2,c3 = st.columns(3)
    c1.metric("モデル精度", f"{acc*100:.1f}%")
    c2.metric("AI選別 送付対象", f"{len(ai_target):,}人")
    c3.metric("全件からの削減率", f"{(1 - len(ai_target)/len(df))*100:.1f}%")
    importance_df = pd.DataFrame({"特徴量": features, "重要度": model.feature_importances_}).sort_values("重要度", ascending=True)
    st.plotly_chart(px.bar(importance_df, x="重要度", y="特徴量", orientation="h", title="特徴量の重要度"), use_container_width=True)
    fig_score = px.histogram(df, x="AI送付スコア", nbins=50, title="AI送付スコアの分布")
    fig_score.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="閾値 0.5")
    st.plotly_chart(fig_score, use_container_width=True)
    st.dataframe(ai_target.head(100), hide_index=True, height=300)
    st.download_button("📥 送付リストCSVダウンロード", ai_target.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"DM送付リスト_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

with tab3:
    st.header("👥 セグメント分析")
    cluster_features = ["累計購入回数","累計購入金額","最終購入日からの日数","過去DM反応回数","メルマガ開封率"]
    X_scaled = StandardScaler().fit_transform(df[cluster_features])
    df["セグメント"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)
    df["セグメント名"] = df["セグメント"].map(lambda x: f"セグメント{x+1}")
    seg = df.groupby("セグメント名").agg(
        顧客数=("顧客ID","count"), 平均購入回数=("累計購入回数","mean"),
        平均購入金額=("累計購入金額","mean"), DM反応率=("DM反応フラグ","mean")
    ).round(1)
    seg["DM反応率"] = (seg["DM反応率"] * 100).round(1).astype(str) + "%"
    st.dataframe(seg, height=250)
    fig_seg = px.scatter(df.sample(5000, random_state=42), x="累計購入金額", y="累計購入回数",
                         color="セグメント名", title="セグメント散布図（サンプル5,000件）")
    st.plotly_chart(fig_seg, use_container_width=True)

with tab4:
    st.header("💴 コストシミュレーション")
    st.caption("全件送付 vs AI最適化の比較")
    total_count = len(df)
    ai_count = len(ai_target)
    reduced = total_count - ai_count

    st.subheader("📬 送付通数の比較")
    c1,c2,c3 = st.columns(3)
    c1.metric("全件送付", f"{total_count:,}通")
    c2.metric("AI最適化", f"{ai_count:,}通")
    c3.metric("削減通数", f"{reduced:,}通", delta=f"-{reduced/total_count*100:.1f}%")

    st.subheader("💰 1回あたりのコスト内訳")
    cost_all_print = total_count * cost_print
    cost_all_enclose = total_count * cost_enclose
    cost_all_postage = total_count * cost_postage
    cost_all_total = total_count * cost_per_mail + cost_design
    cost_ai_print = ai_count * cost_print
    cost_ai_enclose = ai_count * cost_enclose
    cost_ai_postage = ai_count * cost_postage
    cost_ai_total = ai_count * cost_per_mail + cost_design
    saving_once = cost_all_total - cost_ai_total

    breakdown = pd.DataFrame({
        "項目": ["印刷費", "封入・封緘費", "郵送費", "デザイン費", "合計"],
        "全件送付": [f"¥{cost_all_print:,}", f"¥{cost_all_enclose:,}", f"¥{cost_all_postage:,}", f"¥{cost_design:,}", f"¥{cost_all_total:,}"],
        "AI最適化": [f"¥{cost_ai_print:,}", f"¥{cost_ai_enclose:,}", f"¥{cost_ai_postage:,}", f"¥{cost_design:,}", f"¥{cost_ai_total:,}"],
        "削減額": [f"¥{cost_all_print - cost_ai_print:,}", f"¥{cost_all_enclose - cost_ai_enclose:,}",
                   f"¥{cost_all_postage - cost_ai_postage:,}", "¥0", f"¥{saving_once:,}"]
    })
    st.dataframe(breakdown, hide_index=True, width=800)

    st.subheader("📊 コスト比較グラフ")
    fig_cost = go.Figure(data=[
        go.Bar(name="全件送付", x=["印刷費","封入費","郵送費","デザイン費"],
               y=[cost_all_print, cost_all_enclose, cost_all_postage, cost_design]),
        go.Bar(name="AI最適化", x=["印刷費","封入費","郵送費","デザイン費"],
               y=[cost_ai_print, cost_ai_enclose, cost_ai_postage, cost_design])
    ])
    fig_cost.update_layout(barmode="group", title="1回あたりコスト内訳比較")
    st.plotly_chart(fig_cost, use_container_width=True)

    st.subheader("📅 年間コスト比較")
    annual_all = cost_all_total * annual_campaigns
    annual_ai = cost_ai_total * annual_campaigns
    annual_saving = annual_all - annual_ai
    c1,c2,c3 = st.columns(3)
    c1.metric("全件送付（年間）", f"¥{annual_all:,}")
    c2.metric("AI最適化（年間）", f"¥{annual_ai:,}")
    c3.metric("年間削減額", f"¥{annual_saving:,}", delta=f"-{annual_saving/annual_all*100:.1f}%")

    campaigns = list(range(1, annual_campaigns + 1))
    fig_annual = go.Figure()
    fig_annual.add_trace(go.Scatter(x=campaigns, y=[cost_all_total * c for c in campaigns], name="全件送付", mode="lines+markers"))
    fig_annual.add_trace(go.Scatter(x=campaigns, y=[cost_ai_total * c for c in campaigns], name="AI最適化", mode="lines+markers"))
    fig_annual.update_layout(title="累積コスト推移", xaxis_title="キャンペーン回数", yaxis_title="累積コスト（円）")
    st.plotly_chart(fig_annual, use_container_width=True)

st.divider()
st.caption("© 2026 紙DM AI最適化ツール")
