# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# ──────────────────────────────────────────────────────────────────────────────
# File paths (adjust if needed)
RAW_CSV           = "C:/Users/sinwa/Desktop/swiggy_project/data/raw_data/swiggy.csv"
CLEANED_CSV       = "C:/Users/sinwa/Desktop/swiggy_project/data/processed/cleaned_data.csv"
NUMERICAL_CSV     = "C:/Users/sinwa/Desktop/swiggy_project/data/processed/numerical_data.csv"
ENCODER_PKL       = "C:/Users/sinwa/Desktop/swiggy_project/data/processed/encoder.pkl"
KNN_MODEL_PKL     = "C:/Users/sinwa/Desktop/swiggy_project/data/processed/nn_model.pkl" 
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Swiggy Recommender", layout="wide")
st.title("🍽️ Swiggy Restaurant Recommendation App")

# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean_raw_data():
    """
    Load swiggy.csv, clean numeric columns (rating, rating_count, cost),
    drop rows missing name/city/cuisine, and save to CLEANED_CSV.
    """
    df = pd.read_csv(RAW_CSV)

    # Replace known bad entries with NaN
    df.replace(["--", "Too Few Ratings"], np.nan, inplace=True)

    # Clean rating_count: strip non-digits
    df["rating_count"] = (
        df["rating_count"]
          .astype(str)
          .str.replace(r"[^0-9]", "", regex=True)
    )

    # Clean cost: strip non-digits (e.g. "₹250 for two" → "250")
    df["cost"] = (
        df["cost"]
          .astype(str)
          .str.replace(r"[^0-9]", "", regex=True)
    )

    # Convert rating, rating_count, cost to numeric floats
    for col in ["rating", "rating_count", "cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute any remaining NaNs in numeric columns with their median
    for col in ["rating", "rating_count", "cost"]:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    # Drop rows missing name, city, or cuisine
    df = df.drop_duplicates().dropna(subset=["name", "city", "cuisine"]).reset_index(drop=True)

    # Save cleaned data
    df.to_csv(CLEANED_CSV, index=False)
    return df

@st.cache_data
def load_numerical_matrix():
    """
    Load numerical_data.csv. If it doesn’t exist yet, build it by:
    - One-hot encoding ‘city’ and ‘cuisine’
    - Combining with rating/rating_count/cost
    - Saving to NUMERICAL_CSV
    """
    try:
        df_num = pd.read_csv(NUMERICAL_CSV)
        return df_num
    except FileNotFoundError:
        # Build numerical_data.csv
        df_cleaned = pd.read_csv(CLEANED_CSV)

        # Numeric columns
        numeric_cols = ["rating", "rating_count", "cost"]
        df_numeric = df_cleaned[numeric_cols].reset_index(drop=True)

        # One-hot encode ‘city’ and ‘cuisine’
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded_array = encoder.fit_transform(df_cleaned[["city", "cuisine"]])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(["city", "cuisine"]),
            index=df_cleaned.index
        )

        # Concatenate numeric + encoded
        df_num = pd.concat([df_numeric, encoded_df.reset_index(drop=True)], axis=1)
        df_num.to_csv(NUMERICAL_CSV, index=False)

        # Save encoder for future use
        with open(ENCODER_PKL, "wb") as f:
            pickle.dump(encoder, f)

        return df_num

@st.cache_data
def get_city_cuisine_unique():
    """Return sorted lists of unique cities and cuisines from cleaned_data.csv."""
    df = pd.read_csv(CLEANED_CSV)
    cities = sorted(df["city"].unique())
    cuisines = sorted(df["cuisine"].unique())
    return cities, cuisines

def recommend_same_city_and_cuisine(df_cleaned, df_num, chosen_idx: int, n_recs: int = 5):
    """
    Given a chosen index into df_cleaned (0..N-1), return top n_recs indexes
    of restaurants that share both city and cuisine, based on cosine similarity
    over df_num features.
    """
    chosen_city = df_cleaned.loc[chosen_idx, "city"]
    chosen_cuisine = df_cleaned.loc[chosen_idx, "cuisine"]

    # Build masks
    mask_same_city = df_cleaned["city"] == chosen_city
    mask_same_cuis = df_cleaned["cuisine"].str.contains(chosen_cuisine, regex=False)

    mask = mask_same_city & mask_same_cuis
    indices_sub = np.where(mask)[0]

    if len(indices_sub) <= 1:
        return []

    # Subset feature matrix
    df_num_sub = df_num.loc[indices_sub].reset_index(drop=True)

    # Fit NearestNeighbors on this subset
    nn_sub = NearestNeighbors(n_neighbors=n_recs + 1, metric="cosine")
    nn_sub.fit(df_num_sub)

    # Find local position of chosen_idx
    pos_in_sub = np.where(indices_sub == chosen_idx)[0][0]

    # Query neighbors (includes itself)
    distances, neighbors_local = nn_sub.kneighbors(df_num_sub.iloc[[pos_in_sub]], n_recs + 1)
    recs_local = [r for r in neighbors_local[0] if r != pos_in_sub][:n_recs]

    rec_indices = indices_sub[recs_local]
    return rec_indices


# ──────────────────────────────────────────────────────────────────────────────
# 1) Load & clean raw data
df_cleaned = load_and_clean_raw_data()

# 2) Load/build numerical feature matrix
df_num = load_numerical_matrix()

# 3) Load unique city/cuisine for filters
cities, cuisines = get_city_cuisine_unique()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: filter preferences
st.sidebar.header("🔍 Filter Preferences")

selected_city = st.sidebar.selectbox("City", ["All"] + cities)
selected_cuisine = st.sidebar.selectbox("Cuisine", ["All"] + cuisines)

min_rating, max_rating = float(df_cleaned["rating"].min()), float(df_cleaned["rating"].max())
selected_rating = st.sidebar.slider(
    "Minimum Rating",
    min_value=min_rating,
    max_value=max_rating,
    value=min_rating,
    step=0.1
)

min_cost, max_cost = float(df_cleaned["cost"].min()), float(df_cleaned["cost"].max())
selected_cost = st.sidebar.slider(
    "Maximum Cost",
    min_value=min_cost,
    max_value=max_cost,
    value=max_cost,
    step=1.0
)

# ──────────────────────────────────────────────────────────────────────────────
# Apply filters to cleaned DataFrame
filtered_df = df_cleaned.copy()

if selected_city != "All":
    filtered_df = filtered_df[filtered_df["city"] == selected_city]
if selected_cuisine != "All":
    filtered_df = filtered_df[filtered_df["cuisine"].str.contains(selected_cuisine, regex=False)]

filtered_df = filtered_df[
    (filtered_df["rating"] >= selected_rating) &
    (filtered_df["cost"] <= selected_cost)
].reset_index(drop=True)

st.subheader("📋 Filtered Restaurants")
st.write(f"Found **{len(filtered_df)}** restaurants matching your filters.")

if len(filtered_df) == 0:
    st.warning("No restaurants match those filters. Try adjusting them.")
else:
    # Show preview of filtered list
    st.dataframe(filtered_df[["name", "city", "cuisine", "rating", "cost"]].reset_index(drop=True).head(10))

    st.markdown("---")
    st.subheader("🍽️ Choose a Restaurant for Recommendations")

    # Restaurant select box from filtered results
    restaurant_list = filtered_df["name"].tolist()
    selected_restaurant = st.selectbox("Select Restaurant", [""] + restaurant_list)

    if selected_restaurant:
        # Find the global index of selected_restaurant
        chosen_idx = int(df_cleaned[df_cleaned["name"] == selected_restaurant].index[0])

        # Get top-5 same-city & same-cuisine recommendations
        rec_indices = recommend_same_city_and_cuisine(df_cleaned, df_num, chosen_idx, n_recs=5)


# ──────────────────────────────────────────────────────────────────────────────
    if len(rec_indices) == 0:
        st.info("No other restaurants found in the same city & cuisine.")
    if selected_restaurant:
        # Find the global index of selected_restaurant
        chosen_idx = int(df_cleaned[df_cleaned["name"] == selected_restaurant].index[0])

        # Get top-5 same-city & same-cuisine recommendations
        rec_indices = recommend_same_city_and_cuisine(df_cleaned, df_num, chosen_idx, n_recs=5)

        # Check if any recommendations were found
        if len(rec_indices) == 0:
            st.info("No other restaurants found in the same city & cuisine.")
        else:
            recommended_df = df_cleaned.loc[rec_indices].reset_index(drop=True)
            st.subheader("🔍 Recommended Restaurants")
            st.write(recommended_df[["name", "city", "cuisine", "rating", "cost"]])
    
