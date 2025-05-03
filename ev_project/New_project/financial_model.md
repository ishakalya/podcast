# Financial Model

## Step 4: Financial Modelling

### a. Market Identification

*   **Target Market:** Indian EV market, with a focus on state-wise adoption trends as per available data (see data/electric_vehicle_sales_by_state.csv).

### b. Data Collection

*   **Data Sources:**
    - Government of India Vahan Dashboard (aggregated in project dataset)
    - Project data files: data/electric_vehicle_sales_by_state.csv, data/electric_vehicle_sales_by_makers.csv
*   **Key Statistics:**
    - Data covers monthly EV sales by state and maker from April 2021 to March 2023.
    - Example: In 2021, total EV sales (all states): [refer to code output].
    - Major categories: 2-Wheelers, 4-Wheelers.

### c. Market Forecast/Prediction

*   **Model Used:** Simple Linear Regression (scikit-learn) on annual total EV sales (see src/prototype.py).
*   **Forecast Summary:**
    - Annual EV sales by year are aggregated and a linear trend is fitted.
    - Predicted EV sales for next year (e.g., 2024): [see code output, e.g., 12345 units].
    - This forecast is based on historical sales growth and assumes continued market expansion.
*   **Reference:** Refer to `src/prototype.py` for the prediction implementation details.

### d. Financial Equation

*   **Unit Cost:** Rs. 5,00,000 (example per EV, can be adjusted)
*   **Sales Volume:** Predicted sales for next year (e.g., 12345 units, from model)
*   **Fixed Costs:** Rs. 2,00,00,000 (example annual fixed costs)
*   **Revenue Equation:** `Total Revenue (y) = (Unit Price * Sales Volume (x)) - Fixed Costs`
*   **Example Calculation:**
    - Using predicted sales volume from the model and example costs:
    - Total Revenue = (5,00,000 × [Predicted Sales Volume]) - 2,00,00,000
    - See src/prototype.py for the actual calculation and output.