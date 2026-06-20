import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="📊 Deep Dive Analytics", layout="wide")
sns.set_style("whitegrid")

@st.cache
def load_data():
    df = pd.read_csv("youtube_data_cleaned.csv")
    if 'Video Publish Time' in df.columns:
        df['Video Publish Time'] = pd.to_datetime(df['Video Publish Time'], errors='coerce')
    return df

df = load_data()

st.title("🔍 Deep Dive YouTube Analytics")

# Filter by date if available
if 'Video Publish Time' in df.columns:
    min_date = df['Video Publish Time'].min().date()
    max_date = df['Video Publish Time'].max().date()
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        df = df[(df['Video Publish Time'] >= start_date) & (df['Video Publish Time'] <= end_date)]

# ---- Correlation Heatmap of All Numeric Features ----
if st.checkbox("Show correlation heatmap of all numeric features", value=True):
    st.subheader("🔗 Correlation Heatmap")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax)
    st.pyplot(fig)

# CTR Distribution
if 'Video Thumbnail CTR (%)' in df.columns:
    st.subheader("🎯 CTR (Click-Through Rate) Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Video Thumbnail CTR (%)'], bins=25, kde=True, color='orange', ax=ax)
    st.pyplot(fig)

# Engagement over Time
if all(col in df.columns for col in ['Likes', 'Shares', 'New Comments']):
    st.subheader("📢 Engagement Over Time")
    df['Total Engagement'] = df['Likes'] + df['Shares'] + df['New Comments']
    fig, ax = plt.subplots(figsize=(12,5))
    df.groupby(df['Video Publish Time'].dt.date)['Total Engagement'].sum().plot(ax=ax)
    ax.set_ylabel("Total Engagement")
    st.pyplot(fig)

# Watch Time vs Revenue
if all(col in df.columns for col in ['Watch Time (hours)', 'Estimated Revenue (USD)']):
    st.subheader("⏱ Watch Time vs Revenue")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Watch Time (hours)', y='Estimated Revenue (USD)', data=df, alpha=0.6, ax=ax, color='green')
    st.pyplot(fig)

# Best Day to Post
if 'Video Publish Time' in df.columns and 'Estimated Revenue (USD)' in df.columns:
    st.subheader("📅 Average Revenue by Day of Week")
    df['Weekday'] = df['Video Publish Time'].dt.day_name()
    avg_rev_day = df.groupby('Weekday')['Estimated Revenue (USD)'].mean().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    )
    fig, ax = plt.subplots()
    avg_rev_day.plot(kind='bar', color='purple', ax=ax)
    ax.set_ylabel("Average Revenue ($)")
    st.pyplot(fig)
