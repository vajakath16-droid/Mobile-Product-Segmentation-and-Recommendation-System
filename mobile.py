import pandas as pd, plotly.express as px, streamlit as st

st.set_page_config(page_title="Mobile Segmentation & Recommender", layout="wide")

@st.cache_data
def load():
    return (pd.read_csv("products_segmented.csv"), pd.read_csv("sim_content.csv", index_col=0),
            pd.read_csv("sim_cf.csv", index_col=0), pd.read_csv("reviews_slim.csv"))

products, sim_c, sim_cf, reviews = load()
SEGS = ["Budget", "Mid-range", "Premium (higher-rated)", "Premium (lower-rated)"]

st.title("Mobile Product Segmentation & Recommendation System")
tab1, tab2, tab3 = st.tabs(["Overview", "Segments", "Recommender"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{len(reviews):,}"); c2.metric("Models", products.shape[0])
    c3.metric("Brands", products.brand.nunique()); c4.metric("Avg rating", f"{reviews.rating.mean():.2f}")
    st.plotly_chart(px.histogram(reviews, x="rating", color="sentiment", barmode="group"), use_container_width=True)
    st.plotly_chart(px.box(reviews, x="brand", y="price_usd", color="brand"), use_container_width=True)

with tab2:
    seg = st.multiselect("Segments", SEGS, default=SEGS)
    d = products[products.segment.isin(seg)]
    st.plotly_chart(px.scatter(d, x="pc1", y="pc2", color="segment", text="model", size="n_reviews",
                               category_orders={"segment": SEGS}), use_container_width=True)
    st.plotly_chart(px.scatter(d, x="avg_price", y="avg_rating", color="segment", hover_name="model"), use_container_width=True)
    st.dataframe(d.groupby("segment")[["avg_price", "avg_rating", "pct_positive", "n_reviews"]].mean().round(2))

with tab3:
    model = st.selectbox("Pick a phone you like", sorted(products.model))
    n = st.slider("How many recommendations", 1, 10, 5)
    alpha = st.slider("Weight on feature similarity (vs. collaborative filtering)", 0.0, 1.0, 0.7)
    same = st.checkbox("Only same segment")
    score = (alpha * sim_c[model] + (1 - alpha) * sim_cf[model]).drop(model)
    out = products.set_index("model").loc[score.index, ["brand", "avg_price", "avg_rating", "segment"]].assign(score=score.round(3))
    if same: out = out[out.segment == products.set_index("model").loc[model, "segment"]]
    st.dataframe(out.sort_values("score", ascending=False).head(n).round(2))
