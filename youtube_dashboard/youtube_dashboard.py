import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="🎥 YouTube Revenue & KPI Optimizer", layout="wide")
st.sidebar.markdown("### 📄 Pages")
st.sidebar.write("Open 'Deep Dive Analytics' from the left sidebar navigation.")
sns.set_style("whitegrid")

# -------------------- LOAD DATA --------------------
@st.cache
def load_data():
    df = pd.read_csv("youtube_data_cleaned.csv")
    if 'Video Publish Time' in df.columns:
        df['Video Publish Time'] = pd.to_datetime(df['Video Publish Time'], errors='coerce')
    return df

# Load full dataset (for ML model training & predictions)
df_full = load_data()
df = df_full.copy()

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔎 Filter Data")
if 'Video Publish Time' in df.columns:
    min_date = df['Video Publish Time'].min().date()
    max_date = df['Video Publish Time'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        df = df[(df['Video Publish Time'] >= start_date) & (df['Video Publish Time'] <= end_date)]

# -------------------- KPI METRICS --------------------
st.markdown("## 📊 Key Channel Metrics")
col1, col2, col3 = st.columns(3)
if 'Estimated Revenue (USD)' in df.columns:
    col1.metric("💰 Total Revenue", f"${df['Estimated Revenue (USD)'].sum():,.2f}")
if 'Views' in df.columns:
    col2.metric("📺 Total Views", f"{df['Views'].sum():,}")
if 'Subscribers' in df.columns:
    col3.metric("👥 Subscribers", f"{df['Subscribers'].sum():,}")

# -------------------- REVENUE DISTRIBUTION --------------------
if 'Estimated Revenue (USD)' in df.columns:
    st.subheader("💵 Revenue Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Estimated Revenue (USD)'], bins=30, kde=True, color='#ff5252', ax=ax)
    st.pyplot(fig)

# -------------------- REVENUE VS VIEWS --------------------
if 'Views' in df.columns and 'Estimated Revenue (USD)' in df.columns:
    st.subheader("📈 Revenue vs Views")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Views', y='Estimated Revenue (USD)', data=df, alpha=0.6, ax=ax)
    ax.set_xscale('log')
    st.pyplot(fig)

# -------------------- TOP 10 VIDEOS --------------------
if 'Estimated Revenue (USD)' in df.columns:
    st.subheader("🏅 Top 10 Videos by Revenue")
    cols_to_display = ['Video Publish Time', 'Estimated Revenue (USD)']
    for col in ['Views', 'Watch Time (hours)', 'Video Thumbnail CTR (%)']:
        if col in df.columns:
            cols_to_display.append(col)
    st.dataframe(df.nlargest(10, 'Estimated Revenue (USD)')[cols_to_display])

# -------------------- HEATMAP --------------------
if 'Video Publish Time' in df.columns and 'Estimated Revenue (USD)' in df.columns:
    st.subheader("📅 Best Posting Times (Revenue Heatmap)")
    df['Weekday'] = df['Video Publish Time'].dt.day_name()
    df['Hour'] = df['Video Publish Time'].dt.hour
    heat_data = df.pivot_table(values='Estimated Revenue (USD)', index='Weekday', columns='Hour', aggfunc='mean').fillna(0)
    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(heat_data, cmap='Reds', ax=ax)
    st.pyplot(fig)

# -------------------- TRAIN ML MODEL ON FULL DATA --------------------
if all(col in df_full.columns for col in ['Views', 'Watch Time (hours)', 'Video Thumbnail CTR (%)', 'Estimated Revenue (USD)']):
    st.subheader("🤖 Revenue Prediction & What‑If Simulator")

    features = ['Views', 'Watch Time (hours)', 'Video Thumbnail CTR (%)']
    X = df_full[features]  # use full dataset for training
    y = df_full['Estimated Revenue (USD)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # --- Feature Importance ---
    st.markdown("### Feature Importance (Drivers of Revenue)")
    importances = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
    st.bar_chart(importances.set_index("Feature"))

    # --- What-If Inputs ---
    st.markdown("### Try Your Own KPI Values")
    views_input = st.number_input("Views", min_value=0, value=int(df_full['Views'].median()))
    wt_input = st.number_input("Watch Time (hours)", min_value=0, value=int(df_full['Watch Time (hours)'].median()))
    ctr_input = st.slider("Thumbnail CTR (%)", 0.0, 100.0, float(df_full['Video Thumbnail CTR (%)'].median()))

    # Prediction with correct order & DataFrame
    input_df = pd.DataFrame([[views_input, wt_input, ctr_input]], columns=features)
    predicted_rev = model.predict(input_df)
    st.success(f"Predicted Revenue: ${predicted_rev[0]:,.2f}")

    # --- Recommendation ---
    top_feature = importances.sort_values("Importance", ascending=False).iloc[0]['Feature']
    st.info(f"💡 Recommendation: Focus on improving **{top_feature}** — it has the most impact on revenue in your dataset.")
