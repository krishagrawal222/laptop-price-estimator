# Laptop Price Estimator

A capstone project on laptop prices: data analysis in Google Colab, price prediction with machine learning, and a web app that estimates the price of a laptop from its specifications.

**Live web app:** _add your Netlify link here_

## What is in this repository

| Path | What it is |
|---|---|
| `index.html` | The finished web app. It is one file with the dataset and the trained model inside, so it needs no server. |
| `data/laptop_price_inr_cleaned.csv` | The cleaned dataset: 1,275 laptops, 25 columns, prices in Indian rupees. |
| `notebooks/Laptop_Price_Capstone.ipynb` | The Google Colab data analysis (overview, univariate, bivariate and multivariate analysis, correlation, outliers, insights). |
| `webapp/template.html` | The page design and JavaScript for the web app. |
| `webapp/build.py` | Trains the model on the CSV and writes `index.html`. |
| `webapp/requirements.txt` | Python libraries needed by `build.py`. |

## The dataset
Based on the *Laptop Price* dataset by Muhammet Varli on Kaggle. Changes made:
- Prices converted from euros to rupees (1 EUR = ₹110).
- 28 duplicate listings removed.
- Text columns (screen resolution, CPU, memory, RAM, weight) split into numeric columns.
- `Price_Category` added: Budget (under ₹60,000), Mid-range (₹60,000 to ₹1,20,000), Premium (above ₹1,20,000).

Prices come from an older European dataset, so the estimates are indicative and may not match today's Indian market prices.

## The model
A Random Forest regressor (60 trees) trained on laptop specifications. On a held-out 20% of the laptops it scored R² 0.843 with an average error of about ₹20,700, and it picked the correct price category 84.3% of the time. The web app uses a final model trained on all 1,275 laptops.

## Run the analysis
Open `notebooks/Laptop_Price_Capstone.ipynb` in Google Colab, run the first cells, and upload `data/laptop_price_inr_cleaned.csv` when asked.

## Run or rebuild the web app
To use it, open `index.html` in any browser.

To retrain and rebuild it from the CSV:
```
cd webapp
pip install -r requirements.txt
python build.py
```
This overwrites `index.html` in the repository root.

## Host it
The site is a single static file, so any static host works.
- **GitHub Pages:** Settings, Pages, deploy from the `main` branch and `/ (root)`.
- **Netlify or Vercel:** import this repository, leave the build command empty and set the publish directory to `.` (the root).
