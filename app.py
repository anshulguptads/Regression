"""
==========================================================================
  Real Estate Price Prediction — Regression Model Showdown
  Streamlit Dashboard  |  Synthetic Data  |  Storyline Format
==========================================================================
  Business Domain : Real Estate Price Prediction
  Thesis          : Lasso > Ridge > Linear Regression (on sparse, 
                    high-dimensional, multicollinear data)
  Bonus           : Decision Tree family regressors benchmarked alongside
==========================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regression Showdown — Real Estate",
    page_icon="🏠",
    layout="wide",
)

# ── Custom Styling ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    h1 {color: #1a1a2e;}
    h2 {color: #16213e; border-bottom: 2px solid #e94560; padding-bottom: 6px;}
    h3 {color: #0f3460;}
    .stMetric > div {background: #f8f9fa; border-radius: 10px; padding: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════
# CHAPTER 0 — TITLE & INTRODUCTION
# ════════════════════════════════════════════════════════════════════════
st.title("🏠 Real Estate Price Prediction — Regression Model Showdown")
st.markdown(
    """
    > **Hypothesis:** In a real-estate dataset with many features but only a 
    > handful of true price drivers, **Lasso** will outperform **Ridge**, 
    > which in turn will outperform plain **Linear Regression**.  
    > We also benchmark **Decision Tree family** algorithms to see where 
    > tree-based models land.
    """
)
st.divider()

# ════════════════════════════════════════════════════════════════════════
# CHAPTER 1 — SIDEBAR CONTROLS
# ════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Experiment Controls")
n_samples = st.sidebar.slider("Number of Properties", 500, 5000, 1000, 100)
n_features = st.sidebar.slider("Total Features", 50, 300, 200, 10)
n_informative = st.sidebar.slider("True Price Drivers", 3, 15, 5, 1)
noise_level = st.sidebar.slider("Market Noise (×$1K σ)", 10, 120, 60, 5)
test_size = st.sidebar.slider("Test Set %", 10, 40, 20, 5)
random_state = st.sidebar.number_input("Random Seed", 0, 999, 42)

st.sidebar.divider()
st.sidebar.markdown("**Regularization Strength**")
ridge_alpha = st.sidebar.slider("Ridge α", 1.0, 500.0, 100.0, 5.0)
lasso_alpha = st.sidebar.slider("Lasso α", 100.0, 10000.0, 5000.0, 100.0)

# ════════════════════════════════════════════════════════════════════════
# CHAPTER 2 — SYNTHETIC DATA GENERATION
# ════════════════════════════════════════════════════════════════════════
st.header("📊 Chapter 1 — The Dataset")

@st.cache_data
def generate_real_estate_data(
    n_samples, n_features, n_informative, noise_level, random_state
):
    """
    Generates synthetic real-estate data engineered so that:
      • Only a few features are true price drivers   → favours Lasso
      • Many features are multicollinear copies       → destabilises OLS
      • Moderate noise                               → rewards regularisation
    
    Design ensures:  RMSE_Lasso < RMSE_Ridge < RMSE_OLS
    """
    rng = np.random.default_rng(random_state)

    # --- Base random features ---
    X = rng.standard_normal((n_samples, n_features))

    # --- Inject heavy multicollinearity ---
    # Make many features near-exact copies of the informative ones
    n_collinear = min(int(n_features * 0.4), n_features - n_informative)
    for i in range(n_collinear):
        src = rng.choice(n_informative)
        tgt = n_informative + i
        X[:, tgt] = X[:, src] + rng.normal(0, 0.05, n_samples)

    # --- Sparse true coefficients (only n_informative features matter) ---
    true_coefs = np.zeros(n_features)
    true_coefs[:n_informative] = (
        rng.uniform(50_000, 150_000, n_informative)
        * rng.choice([-1, 1], n_informative)
    )

    # --- Target: realistic house prices ---
    noise_std = noise_level * 1000  # slider value × 1000
    y = X @ true_coefs + rng.normal(0, noise_std, n_samples)
    y = y - y.min() + 200_000  # shift to $200k+ range

    # --- Pretty feature names (real-estate flavour) ---
    feature_names = [
        "Sq_Ft", "Lot_Size", "Bedrooms", "Bathrooms", "Garage_Spaces",
        "Year_Built", "Renovation_Year", "Pool_Flag", "Distance_CBD",
        "School_Rating", "Crime_Index", "Walk_Score", "Transit_Score",
        "Floors", "Basement_SqFt", "Fireplace_Count", "HOA_Monthly",
        "Property_Tax", "Median_Income_Area", "Avg_Neighbour_Price",
    ]
    while len(feature_names) < n_features:
        feature_names.append(f"Feature_{len(feature_names)+1}")

    df = pd.DataFrame(X, columns=feature_names[:n_features])
    df["Price"] = np.round(y, 2)

    return df, true_coefs


df, true_coefs = generate_real_estate_data(
    n_samples, n_features, n_informative, noise_level, random_state
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Properties", f"{n_samples:,}")
col2.metric("Total Features", n_features)
col3.metric("True Drivers", n_informative)
col4.metric("Price Range", f"${df['Price'].min():,.0f}–${df['Price'].max():,.0f}")

with st.expander("🔍 Preview Dataset (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# --- Sparsity & Multicollinearity Visual ---
st.subheader("Why This Dataset Favours Lasso")

fig_intro, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Coefficient sparsity
sorted_coefs = np.sort(np.abs(true_coefs))[::-1]
colors = ["#e94560" if c > np.percentile(np.abs(true_coefs), 75) else "#c4c4c4"
          for c in sorted_coefs]
ax1.bar(range(len(sorted_coefs)), sorted_coefs, color=colors, edgecolor="none")
ax1.set_title("True Coefficient Magnitudes (sorted)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Feature Rank")
ax1.set_ylabel("|Coefficient|")
ax1.axhline(y=0.5, color="#0f3460", linestyle="--", alpha=0.5, label="Near-zero zone")
ax1.legend()

# Correlation heatmap (top 15 features)
corr = df.iloc[:, :15].corr()
sns.heatmap(corr, ax=ax2, cmap="coolwarm", center=0, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, xticklabels=True, yticklabels=True)
ax2.set_title("Feature Correlation (first 15)", fontsize=12, fontweight="bold")
ax2.tick_params(axis="both", labelsize=7)
plt.tight_layout()
st.pyplot(fig_intro)

st.info(
    "🔑 **Key Insight:** Most coefficients are near zero (sparse signal) and "
    "features are correlated. OLS will overfit the noise; Ridge will shrink but "
    "keep all features; Lasso will zero out the irrelevant ones."
)

st.divider()

# ════════════════════════════════════════════════════════════════════════
# CHAPTER 3 — MODEL TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════════════
st.header("🤖 Chapter 2 — The Model Showdown")

X = df.drop("Price", axis=1).values
y = df["Price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=random_state
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# --- Define all models ---
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=ridge_alpha, random_state=random_state),
    "Lasso Regression": Lasso(alpha=lasso_alpha, random_state=random_state, max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=random_state),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=random_state, learning_rate=0.1),
    "AdaBoost": AdaBoostRegressor(n_estimators=200, random_state=random_state, learning_rate=0.1),
}

# --- Train & Evaluate ---
results = []
predictions = {}

for name, model in models.items():
    # Use scaled data for linear models, raw for tree-based
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "RMSE ($)": round(rmse, 2), "R² Score": round(r2, 4)})
    predictions[name] = y_pred

results_df = pd.DataFrame(results).sort_values("RMSE ($)").reset_index(drop=True)
results_df.index = results_df.index + 1  # 1-based ranking

# --- Category column ---
def get_category(name):
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        return "Linear"
    return "Tree-Based"

results_df["Family"] = results_df["Model"].apply(get_category)

# --- Display Results Table ---
st.subheader("📋 Performance Scoreboard (sorted by RMSE ↑ = best first)")
st.dataframe(
    results_df.style
    .background_gradient(subset=["RMSE ($)"], cmap="RdYlGn_r")
    .background_gradient(subset=["R² Score"], cmap="RdYlGn")
    .format({"RMSE ($)": "${:,.2f}", "R² Score": "{:.4f}"}),
    use_container_width=True,
    height=300,
)

# --- Highlight the thesis ---
linear_results = results_df[results_df["Family"] == "Linear"].sort_values("RMSE ($)")
if len(linear_results) >= 3:
    ranking = linear_results["Model"].tolist()
    r2_vals = dict(zip(linear_results["Model"], linear_results["R² Score"]))
    if ranking[0] == "Lasso Regression" and ranking[1] == "Ridge Regression":
        gap1 = r2_vals["Ridge Regression"] - r2_vals["Linear Regression"]
        gap2 = r2_vals["Lasso Regression"] - r2_vals["Ridge Regression"]
        st.success(
            f"✅ **THESIS CONFIRMED:** Lasso > Ridge > Linear Regression in R²!  \n"
            f"R² Gaps — Ridge over OLS: **+{gap1:.4f}** | Lasso over Ridge: **+{gap2:.4f}**"
        )
    else:
        st.warning(
            "⚠️ Thesis not perfectly confirmed with current settings. "
            "Try increasing features, reducing informative features, or adjusting α values."
        )

st.divider()

# ════════════════════════════════════════════════════════════════════════
# CHAPTER 4 — VISUAL COMPARISONS
# ════════════════════════════════════════════════════════════════════════
st.header("📈 Chapter 3 — Visual Deep Dive")

tab1, tab2, tab3, tab4 = st.tabs(
    ["RMSE Comparison", "R² Comparison", "Actual vs Predicted", "Coefficient Analysis"]
)

# ── Tab 1: RMSE Bar ─────────────────────────────────────────────────────
with tab1:
    fig1, ax = plt.subplots(figsize=(10, 5))
    colors_bar = ["#e94560" if f == "Linear" else "#0f3460"
                  for f in results_df["Family"]]
    bars = ax.barh(results_df["Model"], results_df["RMSE ($)"], color=colors_bar,
                   edgecolor="white", height=0.6)
    ax.set_xlabel("RMSE ($)", fontsize=12)
    ax.set_title("Root Mean Squared Error — Lower is Better", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, results_df["RMSE ($)"]):
        ax.text(bar.get_width() + max(results_df["RMSE ($)"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", fontsize=10)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e94560", label="Linear"),
                       Patch(color="#0f3460", label="Tree-Based")],
              loc="lower right")
    plt.tight_layout()
    st.pyplot(fig1)

# ── Tab 2: R² Bar ───────────────────────────────────────────────────────
with tab2:
    fig2, ax = plt.subplots(figsize=(10, 5))
    colors_bar2 = ["#e94560" if f == "Linear" else "#0f3460"
                   for f in results_df["Family"]]
    bars2 = ax.barh(results_df["Model"], results_df["R² Score"], color=colors_bar2,
                    edgecolor="white", height=0.6)
    ax.set_xlabel("R² Score", fontsize=12)
    ax.set_title("R² Score — Higher is Better", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars2, results_df["R² Score"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.legend(handles=[Patch(color="#e94560", label="Linear"),
                       Patch(color="#0f3460", label="Tree-Based")],
              loc="lower right")
    plt.tight_layout()
    st.pyplot(fig2)

# ── Tab 3: Actual vs Predicted Scatter ──────────────────────────────────
with tab3:
    selected_models = st.multiselect(
        "Select models to compare",
        list(predictions.keys()),
        default=["Linear Regression", "Lasso Regression", "Gradient Boosting"],
    )
    if selected_models:
        n_cols = min(len(selected_models), 3)
        n_rows = (len(selected_models) + n_cols - 1) // n_cols
        fig3, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten() if len(selected_models) > 1 else [axes]
        for idx, model_name in enumerate(selected_models):
            ax = axes[idx]
            ax.scatter(y_test, predictions[model_name], alpha=0.4, s=15,
                       color="#0f3460", edgecolor="none")
            lims = [min(y_test.min(), predictions[model_name].min()),
                    max(y_test.max(), predictions[model_name].max())]
            ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
            ax.set_title(model_name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Actual Price ($)")
            ax.set_ylabel("Predicted Price ($)")
            ax.legend(fontsize=8)
        # Hide empty subplots
        for idx in range(len(selected_models), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)

# ── Tab 4: Coefficient Analysis ─────────────────────────────────────────
with tab4:
    st.subheader("How Each Linear Model Treats Features")
    feature_names = df.drop("Price", axis=1).columns.tolist()

    coef_data = pd.DataFrame({
        "Feature": feature_names,
        "OLS": models["Linear Regression"].coef_,
        "Ridge": models["Ridge Regression"].coef_,
        "Lasso": models["Lasso Regression"].coef_,
    })

    col_a, col_b = st.columns(2)

    with col_a:
        # Number of zeroed-out features
        n_zero_lasso = (np.abs(coef_data["Lasso"]) < 1e-6).sum()
        n_zero_ridge = (np.abs(coef_data["Ridge"]) < 1e-6).sum()
        n_zero_ols = (np.abs(coef_data["OLS"]) < 1e-6).sum()

        st.markdown("**Features effectively zeroed out (|coef| < 1e-6)**")
        zero_df = pd.DataFrame({
            "Model": ["OLS", "Ridge", "Lasso"],
            "Zeroed Features": [n_zero_ols, n_zero_ridge, n_zero_lasso],
            "Active Features": [n_features - n_zero_ols,
                                n_features - n_zero_ridge,
                                n_features - n_zero_lasso],
        })
        st.dataframe(zero_df, use_container_width=True, hide_index=True)
        st.info(
            f"🎯 Lasso eliminated **{n_zero_lasso}** out of {n_features} features, "
            f"keeping only **{n_features - n_zero_lasso}** active. "
            f"Ridge kept **{n_features - n_zero_ridge}**. OLS kept all **{n_features}**."
        )

    with col_b:
        fig4, ax = plt.subplots(figsize=(8, 5))
        top_k = 20
        top_features = coef_data.reindex(
            coef_data["OLS"].abs().sort_values(ascending=False).index
        ).head(top_k)

        x_pos = np.arange(top_k)
        width = 0.25
        ax.bar(x_pos - width, top_features["OLS"], width, label="OLS", color="#c4c4c4")
        ax.bar(x_pos, top_features["Ridge"], width, label="Ridge", color="#0f3460")
        ax.bar(x_pos + width, top_features["Lasso"], width, label="Lasso", color="#e94560")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_features["Feature"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"Top {top_k} Feature Coefficients Compared", fontsize=12, fontweight="bold")
        ax.set_ylabel("Coefficient Value")
        ax.legend()
        ax.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig4)

st.divider()

# ════════════════════════════════════════════════════════════════════════
# CHAPTER 5 — KEY TAKEAWAYS
# ════════════════════════════════════════════════════════════════════════
st.header("🎓 Chapter 4 — Key Takeaways")

best_model = results_df.iloc[0]["Model"]
best_rmse = results_df.iloc[0]["RMSE ($)"]
best_r2 = results_df.iloc[0]["R² Score"]

st.markdown(
    f"""
    | Insight | Detail |
    |---|---|
    | **Best Overall Model** | {best_model} (RMSE: ${best_rmse:,.2f}, R²: {best_r2:.4f}) |
    | **Among Linear Models** | Lasso's L1 penalty zeroes out noise features → lowest error |
    | **Ridge vs OLS** | Ridge shrinks but retains all features → better than OLS, worse than Lasso |
    | **Tree-Based Models** | Capture non-linearity but may overfit on noisy irrelevant features |
    | **Sparsity Advantage** | When true signal is sparse, feature selection > feature shrinkage > no regularisation |
    """
)

st.markdown("---")
st.caption(
    "Built with Streamlit • Synthetic data via scikit-learn • "
    "Domain: Real Estate Price Prediction • "
    "Designed to demonstrate regularisation advantages on sparse, multicollinear data"
)
