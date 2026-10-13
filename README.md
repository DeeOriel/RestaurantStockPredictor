# Restaurant Stock Predictor (Sample Prototype)

A Streamlit prototype that forecasts daily menu-item sales and calculates illustrative ingredient order recommendations for Steers and Debonairs Pizza.

## Important
All data, recipes, inventory values, and prices are synthetic examples. They are not official brand information and must not be used to place real orders.

## Run on Windows
1. Install Python 3.11+.
2. Open Command Prompt in this folder.
3. Create and activate a virtual environment:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bat
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bat
   streamlit run app.py
   ```

## Included data
- `data/sample_sales.csv` — daily sample sales for menu items across 2025-01-01 through 2026-10-12.
- `data/sample_recipes.csv` — illustrative ingredient quantities per menu item.
- `data/sample_inventory.csv` — illustrative stock levels, safety stock, and unit costs.

## Next improvements
- Upload sales, recipes, and inventory CSV files through the UI.
- Model incoming purchase orders, supplier pack sizes, expiry dates, waste, and stockout days.
- Compare forecasting models and report backtest performance.
- Support multiple branches and user roles.
