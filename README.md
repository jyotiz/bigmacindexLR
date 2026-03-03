# Big Mac Index — Linear Regression (Gradient Descent, 1‑Variable)

**Goal:** Predict **how “expensive” countries are** using the **Economist Big Mac Index**.
We model the relationship between **GDP per capita** and the **price of a Big Mac (USD)** with **linear regression trained via gradient descent** (from scratch), and validate against a **scikit‑learn** baseline.

> Example result (from this run):
> - **GDP ≈ $5,296 → Predicted Big Mac ≈ $2.92**  
> - **GDP ≈ $82,715 → Predicted Big Mac ≈ $5.12**

---

## 🔍 What is the Big Mac Index?
The Big Mac Index (by *The Economist*) is a light‑hearted measure of **purchasing power parity (PPP)**: the same product (a Big Mac) priced across countries to compare **relative cost of living** and **currency valuation**.

---

## 📁 Repository Structure

```
big-mac-index-ml/
├── data/
│   └── bigmac.csv
├── notebooks/
│   └── bigmac_regression.ipynb
├── src/
│   ├── gradient_descent.py
│   ├── metrics.py
│   └── utils.py
├── reports/
│   └── figures/
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
- This project expects at least these columns: `GDP_bigmac`, `dollar_price`.

---

## 📓 How to Run

1. Open **Anaconda Navigator** → **Launch Jupyter Notebook**
2. Open `notebooks/bigmac_regression.ipynb`
3. Run all cells.

---

## 🧮 Modeling Details

### Model
We use **1‑variable linear regression**:

$$
\hat{y} = \theta_0 + \theta_1 x
$$


### Gradient Descent Updates

$$
\theta_0 := \theta_0 - \alpha\, \frac{1}{m} \sum_{i=1}^{m}
\left( \theta_0 + \theta_1 x^{(i)} - y^{(i)} \right)
$$

$$
\theta_1 := \theta_1 - \alpha\, \frac{1}{m} \sum_{i=1}^{m}
\left( \theta_0 + \theta_1 x^{(i)} - y^{(i)} \right) x^{(i)}
$$

### Cost (Mean Squared Error)


$$
J(\theta_0, \theta_1)=
$$

$$
\frac{1}{2m} \sum_{i=1}^{m}
\left( \theta_0 + \theta_1 x^{(i)} - y^{(i)} \right)^2
$$



> **Rendering tips (GitHub):** Ensure there is a **blank line before and after** each `$$` block; do **not** put any other text on the same line as `$$`.

---

## ✅ What You Should See
- Scatter of **GDP per capita vs Big Mac price** with a red regression line.
- Loss curve decreasing smoothly.
- Similar predictions between GD (after inverse‑scaling) and the scikit‑learn baseline.

---

## 🧪 Example Predictions

```
GD Pred @ GDP = 5,296   → ≈ $2.92
GD Pred @ GDP = 82,715  → ≈ $5.12
```

---

## 🧷 Troubleshooting
- **Math not rendering on GitHub?** Keep each `$$` block isolated with blank lines; no trailing characters after `$$`.
- **sklearn 2‑D requirement:** Use `X = df[[x_col]]` or `x.reshape(-1,1)`.
- **Huge predictions:** Verify you’re using `dollar_price` (USD) and `GDP_bigmac` (per‑capita) and inverse‑transform if you trained on standardized values.

---

## 🙌 Acknowledgements
- *The Economist* — Big Mac Index.
- scikit‑learn, NumPy, Pandas, Matplotlib.
