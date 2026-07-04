"""
model.py
Trains regression models to predict YouTube video performance.
Compares multiple models and reports feature importance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import load_raw_data, engineer_features, prepare_for_modelling


def train_and_compare_models(X, y):
    """
    Trains multiple regression models, evaluates with k-fold CV,
    and returns a comparison DataFrame.
    """
    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        # Cross-validated R² scores
        cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
        # Cross-validated RMSE (negative MSE → positive RMSE)
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))

        results.append({
            "Model": name,
            "CV R² Mean": cv_r2.mean(),
            "CV R² Std": cv_r2.std(),
            "CV RMSE Mean": cv_rmse.mean(),
            "CV RMSE Std": cv_rmse.std(),
        })
        print(f"{name}: R²={cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    return pd.DataFrame(results), models


def get_feature_importance(model, feature_names, model_name="Random Forest"):
    """
    Extracts and returns feature importances for tree-based models.
    For linear models, uses absolute coefficients.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline — get from the model step
        inner = model.named_steps["model"]
        if hasattr(inner, "coef_"):
            importances = np.abs(inner.coef_)
        else:
            importances = inner.feature_importances_
    else:
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    return importance_df


def plot_feature_importance(importance_df, top_n=15, save_path=None):
    """Plots feature importances as a horizontal bar chart."""
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(top))
    ax.barh(top["Feature"][::-1], top["Importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features Predicting Video Performance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def fit_best_model(X, y, models):
    """
    Fits the Random Forest on full data and returns it.
    (Best for feature importance interpretation)
    """
    best = models["Random Forest"]
    best.fit(X, y)
    return best


def predict_new_video(model, feature_names, video_features: dict):
    """
    Predicts log_views_per_day for a hypothetical video.
    video_features: dict of feature_name → value
    Returns the predicted views per day (back-transformed from log).
    """
    row = pd.DataFrame([{f: video_features.get(f, 0) for f in feature_names}])
    log_pred = model.predict(row)[0]
    views_per_day = np.expm1(log_pred)
    return views_per_day


if __name__ == "__main__":
    # Load and preprocess
    df = load_raw_data()
    df = engineer_features(df)
    X, y, feature_names = prepare_for_modelling(df)

    print(f"Training on {len(X)} videos with {len(feature_names)} features\n")

    # Train and compare
    results_df, models = train_and_compare_models(X, y)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    # Fit best model and get importances
    best_model = fit_best_model(X, y, models)
    importance_df = get_feature_importance(best_model, feature_names)

    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))

    plot_feature_importance(importance_df, save_path="feature_importance.png")
    print("\nFeature importance plot saved.")