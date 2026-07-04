<<<<<<< HEAD
# YouTube Channel Analytics Dashboard

A machine learning dashboard that predicts YouTube video performance and surfaces insights about what makes videos succeed — built with the YouTube Data API, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![Streamlit](https://img.shields.io/badge/streamlit-1.29+-red)

## What It Does

- **Fetches** video metadata and statistics from one or more YouTube channels via the YouTube Data API v3
- **Engineers features** from titles, upload timing, duration, descriptions, and tags
- **Trains regression models** (Ridge, Lasso, Random Forest, Gradient Boosting) to predict views per day
- **Compares channels** across key performance metrics
- **Predicts** expected performance for hypothetical new videos via an interactive UI

## Dashboard Tabs

| Tab | What You'll See |
|-----|----------------|
| Overview | Total views, top videos, monthly trends, channel comparison |
| Title Analysis | Which title features correlate with more views |
| Timing & Duration | Best upload days, hours, and video lengths |
| ML Model | Feature importance, cross-validated R², model comparison |
| Video Predictor | Enter a hypothetical video and get a predicted views/day |

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourusername/youtube-analytics.git
cd youtube-analytics

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Get a YouTube API key

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → Enable **YouTube Data API v3**
3. Create Credentials → API Key

### 3. Configure

```bash
cp .env.example .env
# Edit .env and paste your API key
```

### 4. Fetch data

Edit the `CHANNELS` list in `src/fetch_data.py` with your channel name(s), then:

```bash
python src/fetch_data.py
```

This saves to `data/raw/videos.csv` (not committed to git).

### 5. Run the dashboard

```bash
streamlit run src/dashboard.py
```

Open `http://localhost:8501` in your browser.

## Project Structure

```
youtube-analytics/
├── src/
│   ├── fetch_data.py      # YouTube API data collection
│   ├── preprocess.py      # Feature engineering
│   ├── model.py           # ML model training & evaluation
│   └── dashboard.py       # Streamlit app
├── data/
│   └── raw/               # API data (gitignored)
├── .env.example           # API key template
├── requirements.txt
└── README.md
```

## Features Engineered

**Title features:** character length, word count, uppercase ratio, contains number/emoji/brackets/question mark, hype word count, negative sentiment words, question words

**Timing features:** day of week, hour of day, month, weekend flag

**Video features:** duration in minutes, Shorts flag, long-form flag, tag count, custom thumbnail flag

**Channel features:** log-transformed subscriber count

**Target variable:** log(views per day) — normalises for video age so a 1-week-old video and a 5-year-old video are comparable

## ML Approach

- **Problem type:** regression (predicting continuous view rate)
- **Models compared:** Ridge Regression, Lasso Regression, Random Forest, Gradient Boosting
- **Validation:** 5-fold cross-validation
- **Target:** log(views per day) — log-transformed to handle the heavy right-skew of view counts
- **Evaluation:** R² and RMSE on held-out folds

## API Quota

The YouTube Data API has a daily quota of 10,000 units. Each channel fetch uses approximately 1-3 units per video (playlist items + video details). Fetching 200 videos per channel costs ~400-600 units — well within the free daily limit.

## Limitations

The model predicts from structural features only. Factors not captured include:
- Actual thumbnail quality and visual appeal
- Content quality and production value
- YouTube algorithm recommendations
- External events (trending topics, viral moments)
- Creator community size and loyalty

Expect R² values of 0.2–0.5 — this is normal and expected given these unobservable factors.

## Skills Demonstrated

- REST API integration (YouTube Data API v3)
- Data collection pipeline with rate limiting
- Feature engineering from unstructured text (titles, descriptions)
- Regression modelling with scikit-learn
- Cross-validation and model evaluation
- Interactive data visualisation with Plotly
- Web app deployment with Streamlit

---

Built as a portfolio project demonstrating end-to-end ML pipeline development.
=======
# YouTube-Channel-Analytics-Dashboard
Dashboard predicting which video topics/titles/thumbnails correlate with higher views.
>>>>>>> 02d88812b1cb0dae8da49c5f9af39bf1105f6559
