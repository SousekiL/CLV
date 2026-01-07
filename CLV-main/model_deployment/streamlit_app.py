# adding necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import lifetimes
from lifetimes import ParetoNBDFitter
from lifetimes import GammaGammaFitter

from sklearn.cluster import KMeans

import altair as alt

np.random.seed(42)

st.markdown("""
# Customer Lifetime Prediction App

Upload the RFM data and get your customer lifetime prediction on the fly.
""")

# --- Replace broken images with always-available placeholder images ---
# Banner image
st.image("https://picsum.photos/1200/350", use_container_width=True)

# Sidebar logo
st.sidebar.image("https://picsum.photos/240/160", width=120)
st.sidebar.markdown("**Made by Felix Liu (demo)**")

st.sidebar.title("Input Features")

days = st.sidebar.slider(
    "Select The No. Of Days", min_value=1, max_value=365, step=1, value=30
)
profit = st.sidebar.slider(
    "Select the Profit Margin", min_value=0.01, max_value=0.09, step=0.01, value=0.05
)

st.sidebar.markdown("### Selected Input Features")
features = pd.DataFrame({"Days": [days], "Profit": [profit]})
st.sidebar.write(features)

st.sidebar.markdown("""
Before uploading the file, please select the input features first.

**Note:** Only use a CSV file with columns:
- frequency
- recency
- T
- monetary_value
""")


# --- Provide a local, always-available sample dataset ---
def make_sample_rfm(n_customers=300, seed=42):
    rng = np.random.default_rng(seed)
    customer_id = np.arange(10001, 10001 + n_customers)

    # Create plausible RFM-like data
    T = rng.integers(30, 365, size=n_customers).astype(
        float
    )  # observation window length (days)
    frequency = rng.poisson(lam=3.0, size=n_customers).astype(float)
    frequency = np.clip(frequency, 0, None)

    # recency must be <= T; allow 0 (common edge case)
    recency = rng.integers(0, 365, size=n_customers).astype(float)
    recency = np.minimum(recency, T)

    # monetary_value > 0 for Gamma-Gamma usage; create some small positive values
    monetary_value = rng.gamma(shape=2.0, scale=50.0, size=n_customers).astype(float)
    monetary_value = np.clip(monetary_value, 1.0, None)

    df = pd.DataFrame(
        {
            "CustomerID": customer_id,
            "frequency": frequency,
            "recency": recency,
            "T": T,
            "monetary_value": monetary_value,
        }
    )

    return df


sample_df = make_sample_rfm(n_customers=300, seed=42)
sample_csv = sample_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sample CSV (RFM format)",
    data=sample_csv,
    file_name="sample_rfm.csv",
    mime="text/csv",
)

st.markdown("---")
data = st.file_uploader("Upload your RFM CSV file", type=["csv"])


def load_and_score(uploaded_file, day=days, profit_m=profit):
    input_data = pd.read_csv(uploaded_file)

    # Optional: if the CSV has an index column or extra first column, your old code used iloc[:,1:]
    # Keep this only if you know the first col is junk. Otherwise comment it out.
    # input_data = pd.DataFrame(input_data.iloc[:, 1:])

    # Basic validation
    required_cols = {"frequency", "recency", "T", "monetary_value"}
    missing = required_cols - set(input_data.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(list(missing))}")
        return None

    # Pareto/NBD model
    pareto_model = ParetoNBDFitter(penalizer_coef=0.1)
    pareto_model.fit(input_data["frequency"], input_data["recency"], input_data["T"])

    input_data["p_alive"] = pareto_model.conditional_probability_alive(
        input_data["frequency"], input_data["recency"], input_data["T"]
    )
    input_data["p_not_alive"] = 1 - input_data["p_alive"]

    t = day
    input_data["predicted_purchases"] = (
        pareto_model.conditional_expected_number_of_purchases_up_to_time(
            t, input_data["frequency"], input_data["recency"], input_data["T"]
        )
    )

    # Gamma-Gamma model: requires frequency > 0 and monetary_value > 0
    input_data = input_data.copy()
    input_data = input_data[
        (input_data["frequency"] > 0) & (input_data["monetary_value"] > 0)
    ].reset_index(drop=True)

    ggf_model = GammaGammaFitter(penalizer_coef=0.1)
    ggf_model.fit(input_data["frequency"], input_data["monetary_value"])

    input_data["expected_avg_sales_"] = ggf_model.conditional_expected_average_profit(
        input_data["frequency"], input_data["monetary_value"]
    )

    input_data["predicted_clv"] = ggf_model.customer_lifetime_value(
        pareto_model,
        input_data["frequency"],
        input_data["recency"],
        input_data["T"],
        input_data["monetary_value"],
        time=t,
        freq="D",
        discount_rate=0.01,
    )

    input_data["profit_margin"] = input_data["predicted_clv"] * profit_m

    # KMeans segmentation (handle NaN/inf defensively)
    col = [
        "predicted_purchases",
        "expected_avg_sales_",
        "predicted_clv",
        "profit_margin",
    ]
    new_df = input_data[col].replace([np.inf, -np.inf], np.nan).dropna()

    # Align input_data to the rows kept for clustering
    input_data = input_data.loc[new_df.index].reset_index(drop=True)
    new_df = new_df.reset_index(drop=True)

    k_model = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    k_model.fit(new_df)

    labels = pd.Series(k_model.labels_, name="Labels")
    input_data = pd.concat([input_data, labels], axis=1)

    label_mapper = {0: "Low", 3: "Medium", 1: "High", 2: "V_High"}
    input_data["Labels"] = input_data["Labels"].map(label_mapper)

    return input_data


if data is not None:
    st.markdown("## Customer Lifetime Prediction Result")
    result = load_and_score(data)

    if result is not None:
        st.dataframe(result, use_container_width=True)

        # Bar chart by label
        fig = (
            alt.Chart(result)
            .mark_bar()
            .encode(
                y=alt.Y("Labels:N", sort="-x"),
                x=alt.X("count(Labels):Q"),
            )
        )
        text = fig.mark_text(align="left", baseline="middle", dx=3).encode(
            text="count(Labels):Q"
        )
        st.altair_chart(fig + text, use_container_width=True)

        # Download result
        out_csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results CSV",
            data=out_csv,
            file_name="customer_lifetime_prediction_result.csv",
            mime="text/csv",
        )
else:
    st.info("Please upload a CSV file (or download the sample above to test).")
