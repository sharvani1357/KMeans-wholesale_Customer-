import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)

# -----------------------------
# Dataset Upload Section
# -----------------------------
st.sidebar.header("📂 Dataset Options")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset (Optional)", type=["csv"]
)

# -----------------------------
# Load Dataset
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded")
else:
    df = pd.read_csv("data/raw/Wholesale customers data.csv")
    st.sidebar.info("ℹ️ Using default Wholesale dataset")

# -----------------------------
# Data Preprocessing
# -----------------------------
st.subheader("🧹 Data Preprocessing")

# Keep only numerical columns
df = df.select_dtypes(include=["int64", "float64"])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna(df.mean(), inplace=True)

st.write("✔️ Numerical columns selected")
st.write("✔️ Missing values handled using mean")
st.write("✔️ Duplicate records removed")

numeric_cols = df.columns.tolist()

# -----------------------------
# Sidebar – Clustering Controls
# -----------------------------
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox("Select Feature 1", numeric_cols)
feature_2 = st.sidebar.selectbox(
    "Select Feature 2", [col for col in numeric_cols if col != feature_1]
)

k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)

random_state = st.sidebar.number_input(
    "Random State (Optional)", min_value=0, max_value=1000, value=42
)

run_button = st.sidebar.button("🟦 Run Clustering")

# -----------------------------
# Run Clustering
# -----------------------------
if run_button:
    selected_features = df[[feature_1, feature_2]]

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_features)

    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(scaled_data)

    df["Cluster"] = clusters
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # -----------------------------
    # Visualization Section
    # -----------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df[feature_1],
        df[feature_2],
        c=df["Cluster"],
        cmap="viridis",
        alpha=0.7
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        s=200,
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Clusters")
    ax.legend()

    st.pyplot(fig)

    # -----------------------------
    # Cluster Summary Section
    # -----------------------------
    st.subheader("📋 Cluster Summary")

    summary = (
        df.groupby("Cluster")[[feature_1, feature_2]]
        .mean()
        .reset_index()
    )

    counts = df["Cluster"].value_counts().sort_index().values
    summary.insert(1, "Customer Count", counts)

    st.dataframe(summary, use_container_width=True)

    # -----------------------------
    # Business Interpretation Section
    # -----------------------------
    st.subheader("💡 Business Interpretation")

    for i in range(k):
        avg_f1 = summary.loc[summary["Cluster"] == i, feature_1].values[0]
        avg_f2 = summary.loc[summary["Cluster"] == i, feature_2].values[0]

        if avg_f1 > df[feature_1].mean() and avg_f2 > df[feature_2].mean():
            msg = "High-spending customers across selected categories."
        elif avg_f1 < df[feature_1].mean() and avg_f2 < df[feature_2].mean():
            msg = "Budget-conscious customers with lower spending."
        else:
            msg = "Moderate spenders with selective purchasing behaviour."

        st.write(f"🟢 **Cluster {i}:** {msg}")

    # -----------------------------
    # User Guidance
    # -----------------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted using similar business strategies."
    )

else:
    st.warning("⬅️ Upload a dataset or select features and click **Run Clustering**")
