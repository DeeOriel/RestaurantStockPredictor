
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Restaurant Stock Predictor",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Restaurant Stock Predictor")
st.caption(
    "Sample prototype for Steers and Debonairs Pizza | "
    "All data and recipes are illustrative, not official."
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


@st.cache_data
def load_data():
    sales_path = DATA_DIR / "sample_sales.csv"
    recipes_path = DATA_DIR / "sample_recipes.csv"
    inventory_path = DATA_DIR / "sample_inventory.csv"

    required_files = [
        sales_path,
        recipes_path,
        inventory_path
    ]

    missing_files = [
        str(path.relative_to(ROOT))
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required data files: "
            + ", ".join(missing_files)
        )

    sales = pd.read_csv(
        sales_path,
        parse_dates=["date"]
    )
    recipes = pd.read_csv(recipes_path)
    inventory = pd.read_csv(inventory_path)

    return sales, recipes, inventory


sales, recipes, inventory = load_data()

with st.sidebar:
    st.header("Forecast settings")
    brand = st.selectbox("Restaurant brand", ["Steers", "Debonairs Pizza"])
    forecast_date = st.date_input("Forecast date", value=pd.Timestamp.today().date() + pd.Timedelta(days=1))
    promotion = st.checkbox("Promotion planned for forecast date", value=False)
    holiday_period = st.checkbox("Holiday / unusually busy period", value=False)
    st.caption("For this prototype, one sample branch is modelled. Replace with approved branch data before operational use.")

def add_features(df):
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["weekend"] = (out["dow"] >= 5).astype(int)
    out["promotion"] = out["promotion"].astype(int)
    out["holiday_period"] = out["holiday_period"].astype(int)
    return out

def forecast_item(item_df, target_date, promo, holiday):
    item_df = item_df.sort_values("date").copy()
    item_df = add_features(item_df)
    feature_cols = ["dow", "month", "dayofyear", "weekend", "promotion", "holiday_period"]
    X = item_df[feature_cols]
    y = item_df["units_sold"]
    # Use a chronological holdout for a simple quality check when enough rows exist.
    split = max(1, int(len(item_df) * 0.8))
    model = RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=42)
    if len(item_df) > 60:
        model.fit(X.iloc[:split], y.iloc[:split])
        val_pred = model.predict(X.iloc[split:])
        mae = mean_absolute_error(y.iloc[split:], val_pred)
        model.fit(X, y)
    else:
        mae = float("nan")
        model.fit(X, y)
    dt = pd.Timestamp(target_date)
    row = pd.DataFrame([{
        "dow": dt.dayofweek, "month": dt.month, "dayofyear": dt.dayofyear,
        "weekend": int(dt.dayofweek >= 5), "promotion": int(promo),
        "holiday_period": int(holiday)
    }])
    pred = max(0, int(round(model.predict(row)[0])))
    return pred, mae

brand_sales = sales[sales["brand"] == brand].copy()
forecast_rows = []
for item in brand_sales["menu_item"].unique():
    item_df = brand_sales[brand_sales["menu_item"] == item]
    pred, mae = forecast_item(item_df, forecast_date, promotion, holiday_period)
    forecast_rows.append({"Menu item": item, "Predicted units": pred, "Validation MAE (units)": round(mae, 1) if np.isfinite(mae) else None})

forecast_df = pd.DataFrame(forecast_rows)
st.subheader(f"Sales forecast — {brand}")
c1, c2, c3 = st.columns(3)
c1.metric("Menu items forecast", len(forecast_df))
c2.metric("Predicted total item sales", int(forecast_df["Predicted units"].sum()))
c3.metric("Forecast date", pd.Timestamp(forecast_date).strftime("%d %b %Y"))
st.dataframe(forecast_df, use_container_width=True, hide_index=True)
st.bar_chart(forecast_df.set_index("Menu item")["Predicted units"])


# Ingredient demand based on recipe bill of materials
brand_recipes = recipes[recipes["brand"] == brand].copy()
demand_rows = []

for _, rec in brand_recipes.iterrows():
    matching_forecast = forecast_df.loc[
        forecast_df["Menu item"] == rec["menu_item"],
        "Predicted units"
    ]

    if matching_forecast.empty:
        continue

    predicted_units = int(matching_forecast.iloc[0])

    demand_rows.append({
        "brand": brand,
        "ingredient": rec["ingredient"],
        "required_qty": (
            predicted_units * float(rec["quantity_per_item"])
        ),
        "unit": rec["unit"],
        "menu_item": rec["menu_item"]
    })

demand_df = pd.DataFrame(
    demand_rows,
    columns=[
        "brand", "ingredient", "required_qty",
        "unit", "menu_item"
    ]
)

if not demand_df.empty:
    ingredient_demand = demand_df.groupby(
        ["brand", "ingredient", "unit"],
        as_index=False
    )["required_qty"].sum()

    stock = inventory[inventory["brand"] == brand].copy()

    # Prevent duplicate ingredient columns during the merge.
    stock = stock[
        ["brand", "ingredient", "on_hand",
         "safety_stock", "unit_cost_pula"]
    ].drop_duplicates(subset=["brand", "ingredient"])

    order_df = ingredient_demand.merge(
        stock,
        on=["brand", "ingredient"],
        how="left"
    )

    for col in ["on_hand", "safety_stock", "unit_cost_pula"]:
        order_df[col] = pd.to_numeric(
            order_df[col], errors="coerce"
        ).fillna(0)

    order_df["recommended_order"] = np.maximum(
        0,
        np.ceil(
            order_df["required_qty"]
            + order_df["safety_stock"]
            - order_df["on_hand"]
        )
    )

    order_df["estimated_cost_pula"] = (
        order_df["recommended_order"]
        * order_df["unit_cost_pula"]
    )

    order_df = order_df.rename(columns={
        "ingredient": "Ingredient",
        "required_qty": "Forecast requirement",
        "unit": "Unit",
        "on_hand": "Current stock",
        "safety_stock": "Safety stock",
        "unit_cost_pula": "Unit cost (P)",
        "recommended_order": "Suggested order",
        "estimated_cost_pula": "Estimated order cost (P)"
    })

    st.subheader("Ingredient requirements and order recommendations")
    st.caption(
        "Calculation: max(0, forecast requirement + safety stock "
        "− usable stock). Supplier pack sizes and incoming deliveries "
        "are not yet modelled."
    )

    show_cols = [
        "Ingredient",
        "Forecast requirement",
        "Unit",
        "Current stock",
        "Safety stock",
        "Suggested order",
        "Unit cost (P)",
        "Estimated order cost (P)"
    ]

    # Defensive check: report missing columns instead of crashing.
    missing_cols = [
        col for col in show_cols
        if col not in order_df.columns
    ]

    if missing_cols:
        st.error(
            f"Order recommendation columns are missing: {missing_cols}"
        )
        st.write("Available columns:", order_df.columns.tolist())
    else:
        st.dataframe(
            order_df[show_cols].round(2),
            use_container_width=True,
            hide_index=True
        )

        total_cost = order_df["Estimated order cost (P)"].sum()
        st.metric("Estimated order cost", f"P{total_cost:,.2f}")

        st.download_button(
            "Download order recommendation CSV",
            order_df[show_cols].to_csv(index=False).encode("utf-8"),
            file_name=(
                f"{brand.lower().replace(' ', '_')}"
                "_order_recommendation.csv"
            ),
            mime="text/csv"
        )
else:
    st.info("No sample recipe rows exist for this brand.")

with st.expander("Data and model notes"):
    st.markdown("""
    - The dataset is synthetic and generated only to demonstrate the workflow.
    - Recipes, stock levels, unit costs, and sales volumes are not official Steers or Debonairs information.
    - The initial model is a Random Forest using calendar, promotion, and holiday features.
    - Validation MAE is a simple chronological holdout estimate; model quality must be assessed on real data.
    - Before use in a live kitchen, add stock movements, deliveries, waste, expiry, supplier pack sizes, and recipe yields.
    """)
