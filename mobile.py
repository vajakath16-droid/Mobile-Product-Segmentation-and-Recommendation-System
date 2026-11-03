
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Product Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #f6f8fc 0%, #edf3fb 100%);}
    [data-testid="stSidebar"] {background: #111827;}
    [data-testid="stSidebar"] * {color: #f9fafb;}
    .hero {
        padding: 1.5rem 1.8rem; border-radius: 18px; color: white;
        background: linear-gradient(110deg, #172554, #2563eb, #06b6d4);
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2.15rem;}
    .hero p {margin: .5rem 0 0 0; opacity: .9;}
    div[data-testid="stMetric"] {
        background: white; padding: 1rem; border-radius: 14px;
        border: 1px solid #e2e8f0; box-shadow: 0 4px 14px #0f172a0b;
    }
    .insight {
        padding: 1rem; background: white; border-radius: 12px;
        border-left: 4px solid #2563eb; margin-bottom: .7rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📊 Product Intelligence</h1>
  <p>Product segmentation · Performance insights · Customer preferences · Smart recommendations</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Data loading --------------------
@st.cache_data
def load_data(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


def make_demo_data(n=350, seed=42):
    rng = np.random.default_rng(seed)
    categories = ["Electronics", "Home & Kitchen", "Fashion", "Beauty", "Sports"]
    customer_segments = ["Budget", "Value", "Premium"]
    rows = []

    for i in range(n):
        category = rng.choice(categories)
        segment = rng.choice(customer_segments, p=[0.42, 0.38, 0.20])
        base_price = {
            "Electronics": 180,
            "Home & Kitchen": 65,
            "Fashion": 42,
            "Beauty": 28,
            "Sports": 55,
        }[category]
        multiplier = {"Budget": 0.55, "Value": 1.0, "Premium": 2.1}[segment]
        price = max(5, rng.normal(base_price * multiplier, base_price * 0.22))
        rating = float(np.clip(
            rng.normal({"Budget": 3.7, "Value": 4.1, "Premium": 4.4}[segment], 0.45),
            1, 5
        ))
        units = int(max(1, rng.lognormal(
            4.0 - 0.28 * np.log1p(price / 40) + 0.20 * (rating - 3), 0.55
        )))
        rows.append({
            "product_id": f"P{i + 1:04d}",
            "product_name": f"{category.split()[0]} Product {i + 1:03d}",
            "category": category,
            "brand": category,
            "price": round(price, 2),
            "units_sold": units,
            "revenue": round(price * units, 2),
            "rating": round(rating, 1),
            "reviews": int(max(0, rng.normal(units * 0.18, 8))),
            "customer_segment": segment,
        })

    return pd.DataFrame(rows)


with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    uploaded = st.file_uploader("Upload product CSV", type=["csv"])
    st.caption("No file? Explore the dashboard with generated demo data.")
    use_demo = st.checkbox("Use demo dataset", value=uploaded is None)
    st.divider()
    st.markdown("### Segmentation")
    n_clusters = st.slider("Number of product clusters", 2, 8, 4)
    st.markdown("### Filters")

if uploaded is not None:
    try:
        raw = load_data(uploaded.getvalue())
        if raw.empty or len(raw.columns) == 0:
            st.error("The CSV file is empty or has no column headers. Please upload a valid CSV.")
            st.stop()
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty. Please upload a valid CSV file.")
        st.stop()
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif use_demo:
    raw = make_demo_data()
else:
    st.info("Upload a product CSV or enable 'Use demo dataset' in the sidebar.")
    st.stop()


# Normalize column names
raw.columns = [
    str(col).strip().lower().replace(" ", "_").replace("-", "_")
    for col in raw.columns
]

# Map common CSV column names
column_aliases = {
    "price_usd": "price",
    "unit_price": "price",
    "selling_price": "price",
    "product_price": "price",
    "discounted_price": "price",
    "actual_price": "price",
    "retail_price": "price",
    "product": "product_name",
    "name": "product_name",
    "segment": "customer_segment",
    "quantity": "units_sold",
    "units": "units_sold",
    "sales": "units_sold",
    "review_count": "reviews",
    "number_of_reviews": "reviews",
    "product_category": "category",
}

for old_name, new_name in column_aliases.items():
    if old_name in raw.columns and new_name not in raw.columns:
        raw = raw.rename(columns={old_name: new_name})

# Check price column
if "price" not in raw.columns:
    st.error("Price column not found. Available columns:")
    st.write(raw.columns.tolist())
    st.stop()

# Create required columns
if "product_name" not in raw.columns:
    raw["product_name"] = [
        f"Product {i + 1}" for i in range(len(raw))
    ]

if "units_sold" not in raw.columns:
    raw["units_sold"] = 1

if "customer_segment" not in raw.columns:
    raw["customer_segment"] = "Unknown"

if "brand" not in raw.columns:
    raw["brand"] = "Unknown"

if "rating" not in raw.columns:
    raw["rating"] = np.nan

if "reviews" not in raw.columns:
    raw["reviews"] = 0

if "category" not in raw.columns:
    raw["category"] = "Uncategory"

# Convert numeric columns
raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
raw["units_sold"] = pd.to_numeric(
    raw["units_sold"], errors="coerce"
)

raw = raw.dropna(subset=["price", "units_sold"]).copy()

if raw.empty:
    st.error("No valid price data found in the CSV file.")
    st.stop()

# Calculate revenue
raw["revenue"] = raw["price"] * raw["units_sold"]

# Create sensible defaults for optional columns
if "product_name" not in raw.columns:
    raw["product_name"] = [f"Product {i + 1}" for i in range(len(raw))]
if "product_id" not in raw.columns:
    raw["product_id"] = [f"P{i + 1:04d}" for i in range(len(raw))]
if "category" not in raw.columns:
    raw["category"] = "Electronics"
if "brand" not in raw.columns:
    raw["brand"] = "Unknown"
if "customer_segment" not in raw.columns:
    raw["customer_segment"] = "Unknown"
if "units_sold" not in raw.columns:
    raw["units_sold"] = 1
if "rating" not in raw.columns:
    raw["rating"] = np.nan
if "reviews" not in raw.columns:
    raw["reviews"] = 0

# Convert numeric columns safely
for col in ["price", "units_sold", "rating", "reviews"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

raw = raw.dropna(subset=["price", "units_sold"]).copy()
raw = raw[(raw["price"] >= 0) & (raw["units_sold"] >= 0)].copy()

if raw.empty:
    st.error("No valid rows found. Check that price and units_sold contain valid non-negative numbers.")
    st.stop()

raw["revenue"] = raw["price"] * raw["units_sold"]
raw["rating"] = raw["rating"].clip(1, 5)
raw["reviews"] = raw["reviews"].fillna(0).clip(lower=0)
raw["category"] = raw["category"].fillna("Uncategorized").astype(str)
raw["customer_segment"] = raw["customer_segment"].fillna("Unknown").astype(str)
raw["product_name"] = raw["product_name"].fillna("Unnamed product").astype(str)
raw["brand"] = raw["brand"].fillna("Unknown").astype(str)

categories = sorted(raw["category"].unique())
segments = sorted(raw["customer_segment"].unique())
with st.sidebar:
    selected_categories = st.multiselect("Product categories", categories, default=categories)
    selected_segments = st.multiselect("Customer segments", segments, default=segments)
    min_price, max_price = float(raw.price.min()), float(raw.price.max())
    if min_price == max_price:
        price_range = (min_price, max_price)
        st.caption(f"Price: {min_price:.2f}")
    else:
        price_range = st.slider("Price range", min_price, max_price,
                                (min_price, max_price), key="price_filter")

df = raw[
    raw["category"].isin(selected_categories)
    & raw["customer_segment"].isin(selected_segments)
    & raw["price"].between(price_range[0], price_range[1])
].copy()

if df.empty:
    st.warning("No products match these filters. Adjust the sidebar selections.")
    st.stop()

# -------------------- Clustering --------------------
feature_cols = ["price", "units_sold", "revenue"]
if df["rating"].notna().sum() >= 2:
    feature_cols.append("rating")
cluster_features = df[feature_cols].copy()
cluster_features = cluster_features.fillna(cluster_features.median(numeric_only=True)).fillna(0)
scaled = StandardScaler().fit_transform(cluster_features)
k = min(n_clusters, len(df))
if len(df) >= 3 and k >= 2:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
else:
    labels = np.zeros(len(df), dtype=int)
df["cluster"] = [f"Cluster {x+1}" for x in labels]
cluster_summary = df.groupby("cluster", as_index=False).agg(
    products=("product_id", "nunique"),
    avg_price=("price", "mean"),
    units_sold=("units_sold", "sum"),
    revenue=("revenue", "sum"),
    avg_rating=("rating", "mean"),
    reviews=("reviews", "sum"),
).sort_values("revenue", ascending=False)
df["performance_score"] = (
    df["revenue"].rank(pct=True) * .45
    + df["units_sold"].rank(pct=True) * .35
    + df["rating"].fillna(df["rating"].median() if df["rating"].notna().any() else 3).rank(pct=True) * .20
)

# -------------------- KPIs --------------------
total_revenue = df["revenue"].sum()
total_units = df["units_sold"].sum()
avg_price = df["price"].mean()
avg_rating = df["rating"].mean()
best = df.loc[df["revenue"].idxmax()]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Revenue", f"{total_revenue:,.2f}")
m2.metric("Units Sold", f"{total_units:,.0f}")
m3.metric("Products", f"{df['product_id'].nunique():,}")
m4.metric("Average Price", f"{avg_price:,.2f}")
m5.metric("Average Rating", "N/A" if pd.isna(avg_rating) else f"{avg_rating:.2f} / 5")

tab_overview, tab_clusters, tab_performance, tab_customers, tab_recommend, tab_data = st.tabs([
    "🏠 Overview", "🧩 Product Segments", "📈 Performance", "👥 Customer Insights",
    "✨ Recommendations", "🗂️ Data Explorer"
])

with tab_overview:
    st.subheader("Business overview")
    c1, c2 = st.columns(2)
    category_summary = df.groupby("category", as_index=False).agg(
        revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
    with c1:
        fig = px.bar(category_summary.sort_values("revenue", ascending=False),
                     x="category", y="revenue", color="revenue",
                     title="Revenue by Category", text_auto=".2s",
                     color_continuous_scale="Blues")
        fig.update_layout(xaxis_title="", yaxis_title="Revenue", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(df, names="category", values="revenue", hole=.52,
                     title="Revenue Mix by Category")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(df, x="price", y="units_sold", size="revenue",
                         color="cluster", hover_name="product_name",
                         hover_data=["category", "rating", "revenue"],
                         title="Price vs Sales Performance")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.bar(cluster_summary, x="cluster", y="revenue", color="cluster",
                     title="Revenue Contribution by Cluster", text_auto=".2s")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick business insights")
    top_category = category_summary.sort_values("revenue", ascending=False).iloc[0]
    low_product = df.sort_values("revenue").iloc[0]
    st.markdown(f"""<div class="insight"><b>🏆 Top category:</b> {top_category['category']} contributes
    {top_category['revenue']:,.2f} in revenue.</div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="insight"><b>⭐ Best revenue product:</b> {best['product_name']}
    ({best['revenue']:,.2f} revenue, {best['units_sold']:,.0f} units sold).</div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="insight"><b>🔎 Review opportunity:</b> {low_product['product_name']}
    has the lowest revenue in the filtered data. Check pricing, visibility, stock, and customer feedback.</div>""",
                unsafe_allow_html=True)

with tab_clusters:
    st.subheader("Product segmentation")
    st.caption("K-means clusters are calculated from price, units sold, revenue, and rating when available.")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = px.scatter(df, x="price", y="units_sold", color="cluster",
                         size="revenue", hover_name="product_name",
                         hover_data=["category", "rating", "revenue"],
                         title="Cluster Map: Price vs Units Sold")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(cluster_summary, x="cluster", y="products", color="cluster",
                     title="Products in Each Cluster", text_auto=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    display_summary = cluster_summary.copy()
    display_summary["avg_price"] = display_summary["avg_price"].round(2)
    display_summary["revenue"] = display_summary["revenue"].round(2)
    display_summary["avg_rating"] = display_summary["avg_rating"].round(2)
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Cluster interpretation")
    for _, row in cluster_summary.iterrows():
        cluster_rows = df[df["cluster"] == row["cluster"]]
        label = ("High-value / high-revenue" if row["revenue"] >= cluster_summary["revenue"].quantile(.67)
                 else "Mid-tier" if row["revenue"] >= cluster_summary["revenue"].quantile(.33)
                 else "Growth opportunity")
        st.markdown(
            f"- **{row['cluster']} — {label}:** {int(row['products'])} products, "
            f"average price {row['avg_price']:.2f}, total revenue {row['revenue']:,.2f}."
        )
    st.download_button("Download cluster analysis CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       "product_cluster_analysis.csv", "text/csv")

with tab_performance:
    st.subheader("High- and low-performing products")
    ranked = df.sort_values("performance_score", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top performers")
        st.dataframe(ranked[["product_name", "category", "price", "units_sold",
                             "revenue", "rating", "cluster"]].head(10),
                     use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### Products needing attention")
        st.dataframe(ranked[["product_name", "category", "price", "units_sold",
                             "revenue", "rating", "cluster"]].tail(10).sort_values("revenue"),
                     use_container_width=True, hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(df, x="price", y="revenue", color="category",
                         size="units_sold", hover_name="product_name",
                         title="Price vs Revenue")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(df, x="category", y="price", color="category",
                     title="Price Distribution by Category")
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Category-level performance")
    cat_perf = df.groupby("category", as_index=False).agg(
        products=("product_id", "nunique"),
        revenue=("revenue", "sum"),
        units=("units_sold", "sum"),
        avg_price=("price", "mean"),
        avg_rating=("rating", "mean"),
    ).sort_values("revenue", ascending=False)
    st.dataframe(cat_perf.round(2), use_container_width=True, hide_index=True)

with tab_customers:
    st.subheader("Customer preference patterns")
    c1, c2 = st.columns(2)
    with c1:
        segment_perf = df.groupby("customer_segment", as_index=False).agg(
            revenue=("revenue", "sum"), units=("units_sold", "sum"),
            products=("product_id", "nunique"), avg_price=("price", "mean"))
        fig = px.bar(segment_perf, x="customer_segment", y="revenue",
                     color="customer_segment", title="Revenue by Customer Segment",
                     text_auto=".2s")
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        preference = df.groupby(["customer_segment", "category"], as_index=False).agg(
            revenue=("revenue", "sum"))
        fig = px.density_heatmap(preference, x="category", y="customer_segment",
                                 z="revenue", histfunc="sum",
                                 title="Category Preference Heatmap",
                                 color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    if df["rating"].notna().any():
        fig = px.scatter(df.dropna(subset=["rating"]), x="rating", y="units_sold",
                         color="customer_segment", size="revenue",
                         hover_name="product_name",
                         title="Customer Rating vs Units Sold")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(segment_perf.round(2), use_container_width=True, hide_index=True)
    st.info("Customer preference patterns use the customer_segment column when available. "
            "If your file lacks customer-level purchase history, segment insights are product-level proxies.")

with tab_recommend:
    st.subheader("✨ Product recommendation assistant")
    st.write("Choose a budget, category, and preferred rating. Recommendations rank products using price fit, rating, and sales performance.")
    c1, c2, c3 = st.columns(3)
    with c1:
        budget = st.number_input("Maximum budget", min_value=0.0,
                                 value=float(max(100, df["price"].quantile(.6))), step=10.0)
    with c2:
        rec_category = st.selectbox("Preferred category", ["Any"] + sorted(df["category"].unique().tolist()))
    with c3:
        min_rating = st.slider("Minimum rating", 1.0, 5.0, 3.5, 0.5)

    recs = df[df["price"] <= budget].copy()
    if rec_category != "Any":
        recs = recs[recs["category"] == rec_category]
    if recs["rating"].notna().any():
        recs = recs[recs["rating"].fillna(0) >= min_rating]
    if recs.empty:
        st.warning("No products match your preferences. Increase the budget or lower the minimum rating.")
    else:
        recs["rating_norm"] = recs["rating"].fillna(3).clip(1, 5) / 5
        recs["sales_norm"] = recs["units_sold"].rank(pct=True)
        recs["value_norm"] = 1 - (recs["price"] / max(budget, 1)).clip(0, 1)
        recs["recommendation_score"] = (
            .45 * recs["rating_norm"] + .35 * recs["sales_norm"] + .20 * recs["value_norm"])
        recs = recs.sort_values("recommendation_score", ascending=False).head(10)
        for _, r in recs.iterrows():
            with st.container(border=True):
                a, b, c, d = st.columns([3, 1, 1, 1])
                a.markdown(f"**{r['product_name']}**")
                a.caption(f"{r['category']} · {r['cluster']}")
                b.metric("Price", f"{r['price']:,.2f}")
                c.metric("Rating", "N/A" if pd.isna(r["rating"]) else f"{r['rating']:.1f} / 5")
                d.metric("Match score", f"{r['recommendation_score']*100:.0f}%")
                st.caption(f"Sales: {r['units_sold']:,.0f} units · Revenue: {r['revenue']:,.2f}")
        st.download_button("Download recommendations",
                           recs.to_csv(index=False).encode("utf-8"),
                           "product_recommendations.csv", "text/csv")

with tab_data:
    st.subheader("Data explorer")
    st.write(f"Showing **{len(df):,}** filtered rows from **{len(raw):,}** valid source rows.")
    search = st.text_input("Search product name")
    view = df.copy()
    if search:
        view = view[view["product_name"].str.contains(search, case=False, na=False)]
    columns = st.multiselect("Columns to display", view.columns.tolist(), default=[
        "product_id", "product_name", "category", "price", "units_sold",
        "revenue", "rating", "customer_segment", "cluster"])
    st.dataframe(view[columns], use_container_width=True, hide_index=True)
    st.download_button("Download filtered data CSV",
                       view.to_csv(index=False).encode("utf-8"),
                       "filtered_product_data.csv", "text/csv")

st.divider()
st.caption("Product Intelligence Dashboard · For decision support. Clusters and recommendations depend on the uploaded data and selected filters.")
