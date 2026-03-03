# Big Mac Index — Linear Regression (Gradient Descent, 1‑Variable)

**Goal:** Predict **how “expensive” countries are** using the **Economist Big Mac Index**.  
We model the relationship between **GDP per capita** and the **price of a Big Mac (USD)** with **linear regression trained via gradient descent** (from scratch), and validate against a **scikit‑learn** baseline.

> Example result (from this run):  
> - **GDP ≈ $5,296 → Predicted Big Mac ≈ $2.92**  
> - **GDP ≈ $82,715 → Predicted Big Mac ≈ $5.12**  
> These align with the Big Mac Index intuition: **richer countries → higher burger prices** (but the slope is moderate; Big Macs typically range in low single digits).

---

## 🔍 What is the Big Mac Index?
The Big Mac Index (by *The Economist*) is a light‑hearted measure of **purchasing power parity (PPP)**: the same product (a Big Mac) priced across countries to compare **relative cost of living** and **currency valuation**.

---

## 📁 Repository Structure

```
big-mac-index-ml/
│
├── data/
│   └── bigmac.csv               # Big Mac Index CSV (place your Kaggle file here)
│
├── notebooks/
│   └── bigmac_regression.ipynb  # Main notebook (EDA, training, plots)
│
├── src/
│   ├── gradient_descent.py      # 1D Linear Regression via Gradient Descent (stable/normalized)
│   ├── metrics.py               # (optional) RMSE, R^2 helpers
│   └── utils.py                 # (optional) shared helpers
│
├── reports/
│   └── figures/                 # Plots saved from notebook
│
├── requirements.txt
└── README.md
```

---

## 🧰 Environment & Setup

**Prereqs**
- Anaconda Distribution (Python, Jupyter): https://www.anaconda.com/products/distribution  
- Git (installed)  
- Python 3.10+

**Install Python libs** (if not included):
```bash
pip install numpy pandas matplotlib scikit-learn
```

**Dataset**  
- Download the **Big Mac Index** CSV from Kaggle and save as `data/bigmac.csv`.  
- This project expects at least these columns:
  - `GDP_bigmac` → GDP per capita (USD)
  - `dollar_price` → Big Mac price in USD

> If your file uses different names, update `x_col` and `y_col` in the notebook.

---

## 📓 How to Run

1. Open **Anaconda Navigator** → **Launch Jupyter Notebook**  
2. Open `notebooks/bigmac_regression.ipynb`  
3. Run all cells:
   - Load & clean data
   - Train **from‑scratch GD** model on **standardized** data
   - Train **scikit‑learn** baseline (closed‑form OLS) on **raw** data
   - Visualize the fit and check metrics
   - Make predictions for chosen GDP values

---

## 🧮 Modeling Details

### Model
We use **1‑variable linear regression**:
\[ \hat{y} = \theta_0 + \theta_1 x \]
- \( x \): `GDP_bigmac` (GDP per capita in USD)  
- \( y \): `dollar_price` (Big Mac price, USD)

### Cost (MSE with ½ factor)
\[ J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 \]

### Gradient Descent updates
\[
\begin{aligned}
\theta_0 &:= \theta_0 - \alpha \cdot \frac{1}{m} \sum_i (\hat{y}^{(i)} - y^{(i)}) \\
\theta_1 &:= \theta_1 - \alpha \cdot \frac{1}{m} \sum_i (\hat{y}^{(i)} - y^{(i)}) \cdot x^{(i)}
\end{aligned}
\]

### Stability (important!)
To avoid overflow/divergence, the GD implementation:
- **Standardizes** \( x \) and \( y \) (z‑score) internally for training
- **Maps learned parameters back** to original scale for final reporting
- Includes **early stopping** and **finite‑value checks**

> The scikit‑learn baseline runs on **raw** (unstandardized) data for a direct sanity‑check.

---

## ✅ What You Should See
- A scatter plot of **GDP per capita vs Big Mac price** with a red regression line (mapped back to dollars).  
- A **loss curve** (cost vs. iterations) that **decreases smoothly**.  
- **Similar parameters/predictions** between your GD model (post inverse‑scaling) and the scikit‑learn baseline.

---

## 🧪 Example Predictions

```text
GD Pred @ GDP = 5,296   → ≈ $2.92
GD Pred @ GDP = 82,715  → ≈ $5.12
```

**Interpretation:** Countries with higher GDP per capita tend to have **more expensive Big Macs** (but the effect size is moderate—Big Macs usually range in **low single digits**).

---

## 📈 Extensions (Try These Next)
1. **Multi‑variable regression**  
   Add predictors like `dollar_ex`, `local_price`, `adj_price`, and compare R² / RMSE.
2. **Year filtering / Cohort analysis**  
   Fit separate lines for different years to see how relationships shift.
3. **Outlier handling**  
   Trim extreme GDP tails (e.g., 0.5th–99.5th percentiles) and compare fits.
4. **Mapping**  
   Plot a world map coloring countries by Big Mac price (e.g., with `plotly` or `geopandas`).
5. **Classification**  
   Bucket countries into “below $3 / $3–$5 / above $5” and try a simple classifier.

---

## 🧷 Troubleshooting
- **Overflow or NaNs in GD:** Use the provided **normalized** GD class (already in `src/gradient_descent.py`).  
- **“Expected 2D array, got 1D array” (scikit‑learn):** Use **2‑D features**: `X = df[[x_col]]` or `x.reshape(-1,1)`.  
- **Predictions are huge (thousands):** Likely wrong columns or predicting in standardized space and not inverse‑transforming. Ensure:
  - `y_col = "dollar_price"` (USD, single digits)
  - `x_col = "GDP_bigmac"` (per‑capita, thousands/tens of thousands)
  - For GD predictions on **raw GDP**, convert raw → scaled → predict → **inverse‑scale** to USD.

---

## 📜 License
This repository is for learning/demonstration. If you publish or share, please retain attribution to the Big Mac Index dataset source and this project’s authors.

---

## 🙌 Acknowledgements
- *The Economist* — **Big Mac Index** (concept and data collection).  
- scikit‑learn — baseline linear regression (closed‑form OLS).  
- NumPy/Pandas/Matplotlib — data wrangling & visualization.

---

### Quick Start (TL;DR)
```bash
# 1) Put dataset:
#    data/bigmac.csv   (must contain GDP_bigmac, dollar_price)

# 2) Install libs
pip install numpy pandas matplotlib scikit-learn

# 3) Run the notebook
#    notebooks/bigmac_regression.ipynb

# 4) View results in reports/figures (optional)
```
