"""
dashboard.py
Streamlit dashboard for YouTube channel analytics.
Run with: streamlit run src/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from preprocess import load_raw_data, engineer_features, prepare_for_modelling, get_feature_columns
from model import train_and_compare_models, get_feature_importance, fit_best_model, predict_new_video


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 YouTube Channel Analytics Dashboard")
st.markdown("*Predicting what makes videos perform — built with YouTube Data API + scikit-learn*")


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = load_raw_data("data/raw/videos.csv")
    df = engineer_features(df)
    return df


@st.cache_resource
def train_model(df):
    X, y, feature_names = prepare_for_modelling(df)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, feature_names, X, y


try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ No data found. Run `python src/fetch_data.py` first to fetch YouTube data.")
    st.stop()


# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

channels = ["All"] + sorted(df["channel_name"].unique().tolist())
selected_channel = st.sidebar.selectbox("Channel", channels)

if selected_channel != "All":
    filtered_df = df[df["channel_name"] == selected_channel]
else:
    filtered_df = df

year_range = st.sidebar.slider(
    "Year range",
    min_value=int(df["publish_year"].min()),
    max_value=int(df["publish_year"].max()),
    value=(int(df["publish_year"].min()), int(df["publish_year"].max()))
)
filtered_df = filtered_df[
    (filtered_df["publish_year"] >= year_range[0]) &
    (filtered_df["publish_year"] <= year_range[1])
]

exclude_shorts = st.sidebar.checkbox("Exclude YouTube Shorts", value=True)
if exclude_shorts:
    filtered_df = filtered_df[filtered_df["is_short"] == 0]

st.sidebar.markdown(f"**{len(filtered_df)} videos** matching filters")


# ── TAB LAYOUT ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview",
    "🔍 Title Analysis",
    "⏱️ Timing & Duration",
    "🤖 ML Model",
    "🎯 Video Predictor"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Channel Overview")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Videos", len(filtered_df))
    col2.metric("Total Views", f"{filtered_df['view_count'].sum():,.0f}")
    col3.metric("Avg Views/Video", f"{filtered_df['view_count'].mean():,.0f}")
    col4.metric("Median Views/Video", f"{filtered_df['view_count'].median():,.0f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        # Views over time
        monthly = filtered_df.groupby(
            filtered_df["published_at"].dt.tz_localize(None).dt.to_period("M")
        )["view_count"].sum().reset_index()
        monthly["published_at"] = monthly["published_at"].astype(str)

        fig = px.bar(
            monthly, x="published_at", y="view_count",
            title="Total Views by Month (upload date)",
            labels={"published_at": "Month", "view_count": "Views"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    with col_right:
        # Top 10 videos
        top_videos = filtered_df.nlargest(10, "view_count")[["title", "view_count", "channel_name"]]
        top_videos["title_short"] = top_videos["title"].str[:50] + "..."
        fig = px.bar(
            top_videos, x="view_count", y="title_short",
            orientation="h",
            title="Top 10 Videos by Views",
            color="channel_name",
            labels={"view_count": "Views", "title_short": ""},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    # Channel comparison (if multiple channels)
    if selected_channel == "All" and df["channel_name"].nunique() > 1:
        st.subheader("Channel Comparison")
        channel_stats = df.groupby("channel_name").agg(
            videos=("video_id", "count"),
            total_views=("view_count", "sum"),
            avg_views=("view_count", "mean"),
            median_views=("view_count", "median"),
            avg_likes=("like_count", "mean"),
            avg_like_rate=("like_rate", "mean"),
        ).reset_index()

        st.dataframe(
            channel_stats.style.format({
                "total_views": "{:,.0f}",
                "avg_views": "{:,.0f}",
                "median_views": "{:,.0f}",
                "avg_likes": "{:,.0f}",
                "avg_like_rate": "{:.3f}",
            }),
            width="stretch"
        )

        fig = px.box(
            df, x="channel_name", y="view_count",
            title="View Count Distribution by Channel",
            log_y=True,
            labels={"channel_name": "Channel", "view_count": "Views (log scale)"}
        )
        st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TITLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Title Analysis")
    st.markdown("Which title characteristics correlate with more views?")

    col_left, col_right = st.columns(2)

    with col_left:
        # Title length vs views
        scatter_data = filtered_df[["title_length", "log_view_count", "title", "channel_name"]].dropna()
        fig = px.scatter(
            scatter_data, x="title_length", y="log_view_count",
            color="channel_name" if selected_channel == "All" else None,
            title="Title Length vs Log(Views)",
            labels={"title_length": "Title character count", "log_view_count": "Log(Views)"},
            hover_data=["title"]
        )
        if len(scatter_data) > 2:
            x_vals = scatter_data["title_length"].values
            y_vals = scatter_data["log_view_count"].values
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            fig.add_scatter(x=x_line, y=p(x_line), mode="lines", name="Trend",
                line=dict(color="red", width=2, dash="dash"))
        st.plotly_chart(fig, width="stretch")

    with col_right:
        # Hype words vs views
        fig = px.box(
            filtered_df,
            x="title_hype_word_count",
            y="log_view_count",
            title="Hype Words in Title vs Log(Views)",
            labels={"title_hype_word_count": "Number of hype words", "log_view_count": "Log(Views)"}
        )
        st.plotly_chart(fig, width="stretch")

    # Binary title feature comparison
    st.subheader("Title Feature Impact on Average Views")

    title_features = {
        "Has Number": "title_has_number",
        "Has Emoji": "title_has_emoji",
        "Has Brackets": "title_has_brackets",
        "Is Question": "title_is_question",
        "Negative Sentiment": "title_negative_sentiment",
        "Has Custom Thumbnail": "has_custom_thumbnail",
    }

    impact_data = []
    for label, col in title_features.items():
        if col in filtered_df.columns:
            avg_with = filtered_df[filtered_df[col] == 1]["view_count"].mean()
            avg_without = filtered_df[filtered_df[col] == 0]["view_count"].mean()
            impact_data.append({
                "Feature": label,
                "With feature": avg_with,
                "Without feature": avg_without,
                "Lift": (avg_with - avg_without) / max(avg_without, 1) * 100
            })

    impact_df = pd.DataFrame(impact_data).sort_values("Lift", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="With feature", x=impact_df["Feature"], y=impact_df["With feature"]))
    fig.add_trace(go.Bar(name="Without feature", x=impact_df["Feature"], y=impact_df["Without feature"]))
    fig.update_layout(
        barmode="group",
        title="Average Views: With vs Without Each Title Feature",
        yaxis_title="Average Views"
    )
    st.plotly_chart(fig, width="stretch")

    # Lift table
    st.dataframe(
        impact_df[["Feature", "With feature", "Without feature", "Lift"]].style.format({
            "With feature": "{:,.0f}",
            "Without feature": "{:,.0f}",
            "Lift": "{:+.1f}%"
        }),
        width="stretch"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TIMING & DURATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Timing & Duration Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        # Best day of week
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_stats = filtered_df.groupby("publish_day_of_week")["view_count"].agg(["mean", "count"])
        day_stats.index = [day_names[i] for i in day_stats.index]
        day_stats = day_stats.reset_index()
        day_stats.columns = ["Day", "Avg Views", "Video Count"]

        fig = px.bar(
            day_stats, x="Day", y="Avg Views",
            title="Average Views by Upload Day",
            text="Video Count",
            labels={"Day": "Day of Week", "Avg Views": "Average Views"}
        )
        fig.update_traces(textposition="outside", texttemplate="n=%{text}")
        st.plotly_chart(fig, width="stretch")

    with col_right:
        # Best hour
        hour_stats = filtered_df.groupby("publish_hour")["view_count"].agg(["mean", "count"]).reset_index()
        hour_stats.columns = ["Hour", "Avg Views", "Count"]

        fig = px.line(
            hour_stats, x="Hour", y="Avg Views",
            title="Average Views by Upload Hour (UTC)",
            markers=True
        )
        st.plotly_chart(fig, width="stretch")

    # Duration vs views
    duration_data = filtered_df[filtered_df["duration_minutes"] < 60].copy()
    dur_scatter = duration_data[["duration_minutes", "log_view_count", "title", "channel_name"]].dropna()
    fig = px.scatter(
        dur_scatter,
        x="duration_minutes", y="log_view_count",
        color="channel_name" if selected_channel == "All" else None,
        title="Video Duration vs Log(Views) — videos under 60 min",
        labels={"duration_minutes": "Duration (minutes)", "log_view_count": "Log(Views)"},
        hover_data=["title"]
    )
    if len(dur_scatter) > 2:
        x_d = dur_scatter["duration_minutes"].values
        y_d = dur_scatter["log_view_count"].values
        zd = np.polyfit(x_d, y_d, 1)
        pd_line = np.poly1d(zd)
        xd_line = np.linspace(x_d.min(), x_d.max(), 100)
        fig.add_scatter(x=xd_line, y=pd_line(xd_line), mode="lines", name="Trend",
            line=dict(color="red", width=2, dash="dash"))
    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Machine Learning Model")
    st.markdown("""
    Predicts **log(views per day)** from title features, timing, duration, and channel size.
    Uses Random Forest Regression with 5-fold cross-validation.
    """)

    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            model, feature_names, X, y = train_model(df)

            # CV scores
            from sklearn.model_selection import cross_val_score
            cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")
            cv_rmse = np.sqrt(-cross_val_score(
                model, X, y, cv=5, scoring="neg_mean_squared_error"
            ))

        col1, col2, col3 = st.columns(3)
        col1.metric("CV R² (mean)", f"{cv_r2.mean():.3f}")
        col2.metric("CV R² (std)", f"± {cv_r2.std():.3f}")
        col3.metric("CV RMSE (mean)", f"{cv_rmse.mean():.3f}")

        st.info(f"""
        **Interpreting R²:** An R² of {cv_r2.mean():.2f} means the model explains
        {cv_r2.mean()*100:.0f}% of the variance in video performance from these features alone.
        The remaining variance comes from factors we can't measure (algorithm recommendations,
        external events, quality of content, thumbnail appeal).
        """)

        # Feature importances
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig = px.bar(
            importance_df.head(15),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Features Predicting Video Performance",
            color="Importance",
            color_continuous_scale="viridis"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

        st.session_state["model"] = model
        st.session_state["feature_names"] = feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — VIDEO PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Video Predictor")
    st.markdown("Simulate a hypothetical video and predict its daily view rate.")

    if "model" not in st.session_state:
        st.warning("Train the model first in the ML Model tab.")
    else:
        model = st.session_state["model"]
        feature_names = st.session_state["feature_names"]

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Video Details")
            title_input = st.text_input("Video title", placeholder="My Honest Review of This Figure...")
            duration = st.slider("Duration (minutes)", 1, 60, 12)
            upload_day = st.selectbox("Upload day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            upload_hour = st.slider("Upload hour (UTC)", 0, 23, 14)
            subscribers = st.number_input("Channel subscribers", value=10000, step=1000)

        with col_right:
            st.subheader("Title Features")
            has_number = st.checkbox("Title contains a number")
            has_emoji = st.checkbox("Title contains an emoji")
            has_brackets = st.checkbox("Title contains brackets [ ] or ( )")
            is_question = st.checkbox("Title ends with ?")
            has_thumbnail = st.checkbox("Has custom thumbnail", value=True)
            tag_count = st.slider("Number of tags", 0, 30, 10)

        if st.button("Predict Performance", type="primary"):
            import re

            # Build feature dict from inputs
            title = title_input or ""
            day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

            hype_words = [
                "unboxing", "haul", "review", "figure", "merch", "limited",
                "exclusive", "rare", "collection", "grail", "worth it",
                "honest", "worst", "best", "ranked", "tier list"
            ]
            hype_count = sum(1 for w in hype_words if w.lower() in title.lower())

            features = {
                "publish_month": 6,
                "publish_day_of_week": day_map[upload_day],
                "publish_hour": upload_hour,
                "uploaded_on_weekend": 1 if upload_day in ["Sat", "Sun"] else 0,
                "title_length": len(title),
                "title_word_count": len(title.split()),
                "title_upper_ratio": sum(1 for c in title if c.isupper()) / max(len(title), 1),
                "title_has_number": int(has_number),
                "title_has_emoji": int(has_emoji),
                "title_has_brackets": int(has_brackets),
                "title_is_question": int(is_question),
                "title_has_pipe": int("|" in title),
                "title_hype_word_count": hype_count,
                "title_negative_sentiment": int(any(w in title.lower() for w in ["worst", "bad", "terrible"])),
                "title_is_question_word": int(any(title.lower().startswith(w) for w in ["how", "why", "what", "when"])),
                "description_length": 300,
                "description_has_links": 1,
                "description_has_timestamps": 1,
                "duration_minutes": duration,
                "is_short": int(duration <= 1),
                "is_long": int(duration >= 20),
                "tag_count": tag_count,
                "has_custom_thumbnail": int(has_thumbnail),
                "log_subscriber_count": np.log1p(subscribers),
            }

            predicted_views_per_day = predict_new_video(model, feature_names, features)

            st.divider()
            st.subheader("Prediction")

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted views/day", f"{predicted_views_per_day:,.0f}")
            col2.metric("Predicted views (30 days)", f"{predicted_views_per_day * 30:,.0f}")
            col3.metric("Predicted views (1 year)", f"{predicted_views_per_day * 365:,.0f}")

            # Compare to channel average
            channel_avg = df["views_per_day"].median()
            pct_diff = (predicted_views_per_day - channel_avg) / channel_avg * 100
            if pct_diff > 0:
                st.success(f"📈 This video is predicted to perform **{pct_diff:.0f}% above** the median video in your dataset.")
            else:
                st.warning(f"📉 This video is predicted to perform **{abs(pct_diff):.0f}% below** the median video in your dataset.")

            st.caption("Note: predictions are based on structural features only (title, timing, duration). Content quality, thumbnails, and algorithmic promotion are major factors the model cannot capture.")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with YouTube Data API v3 · scikit-learn · Streamlit · pandas · plotly")