# Milestone 1 Progress Report: E-Commerce Customer Churn Prediction

**Course Project - Milestone 1**  
**Author**: Masab Rehman  
**Repository**: [GitHub Link](https://github.com/Husnainraza-b/ML-CustomerChurnPredictor-Project)  

---

## 1. Introduction and Revisions to Problem Statement

### 1.1 Project Objective
The objective of this project is to build an end-to-end Machine Learning pipeline to predict customer churn in an e-commerce platform. Specifically, given a set of customer attributes, app interaction patterns, and transaction histories, the system must classify whether a customer is likely to churn (`Churn = 1`) or remain active (`Churn = 0`).

### 1.2 Relevance
In the highly competitive e-commerce sector, customer acquisition costs (CAC) are 5 to 25 times more expensive than customer retention costs. A minor reduction in customer churn can result in a significant increase in profit margins. By proactively predicting which users are about to leave, the marketing and customer success teams can target them with retention offers, loyalty rewards, and personalized feedback loops.

### 1.3 Revisions to Original Problem Statement
Initially, the project aimed to apply standard machine learning models directly to the raw dataset to maximize overall classification accuracy. However, during exploratory data analysis (EDA), two major challenges were identified:
1. **Extreme Class Imbalance**: The ratio of active customers to churned customers is roughly 5:1. Minimizing error rate (maximizing accuracy) on such a dataset causes models to heavily bias toward the majority class (retained customers), resulting in a very high rate of False Negatives (missing actual churners).
2. **No Scikit-Learn Restriction**: The project constraints require all algorithms (Logistic Regression, K-Nearest Neighbors, Naive Bayes, metrics, and data splitters) to be implemented completely from scratch using only `NumPy` and `Pandas`.

Consequently, our revised problem statement shifts the primary performance objective from **Accuracy** to **Recall** (capturing the maximum number of true churners) and **F1-Score** (balancing precision and recall), and incorporates a mandatory class balancing step (random undersampling) in the preprocessing pipeline.

---

## 2. Dataset Description

### 2.1 Data Source and Volume
The dataset used is the Kaggle E-Commerce Customer Churn Dataset. 
* **Raw Shape**: 5,630 rows, 20 columns.
* **Processed Shape**: 5,630 rows, 30 columns (after one-hot encoding categorical features).
* **Target Feature**: `Churn` (Binary: `1` for Churned, `0` for Retained).

### 2.2 Feature Categorization

The features in the dataset are classified as follows:

| Feature Name | Type | Description | Range / Categories |
| :--- | :--- | :--- | :--- |
| **CustomerID** | Numerical (ID) | Unique identifier for each customer | Dropped during preprocessing |
| **Tenure** | Continuous | Duration of relationship in months | 0 to 61 months (contains missing values) |
| **WarehouseToHome** | Continuous | Distance from warehouse to home | 5 to 127 km (contains missing values) |
| **HourSpendOnApp** | Continuous | Hours spent on the app by the customer | 0 to 5 hours (contains missing values) |
| **OrderAmountHikeFromlastYear** | Continuous | Percentage increase in order values | 11% to 26% (contains missing values) |
| **CouponUsed** | Continuous | Number of coupons used in the last month | 0 to 16 coupons (contains missing values) |
| **OrderCount** | Continuous | Total number of orders placed last month | 1 to 16 orders (contains missing values) |
| **DaySinceLastOrder** | Continuous | Days elapsed since the last order | 0 to 46 days (contains missing values) |
| **CashbackAmount** | Continuous | Monthly cashback rewards earned | $0.00 to $324.99 |
| **PreferredLoginDevice** | Categorical | Device used most to log in | Mobile Phone, Computer, Phone |
| **PreferredPaymentMode** | Categorical | Payment method preferred by customer | Credit Card, Debit Card, E-Wallet, COD, UPI |
| **Gender** | Categorical | Customer gender | Male, Female |
| **PreferedOrderCat** | Categorical | Category of goods preferred | Laptop, Mobile Phone, Fashion, Grocery, etc. |
| **MaritalStatus** | Categorical | Marital status of the customer | Single, Married, Divorced |

### 2.3 Quality Issues Identified
1. **Missing Data**: Significant missing values were discovered in continuous columns:
   * `Tenure` (264 missing)
   * `WarehouseToHome` (251 missing)
   * `HourSpendOnApp` (255 missing)
   * `OrderAmountHikeFromlastYear` (265 missing)
   * `CouponUsed` (256 missing)
   * `OrderCount` (258 missing)
   * `DaySinceLastOrder` (307 missing)
2. **Outliers**: Features like `Tenure` and `CashbackAmount` contain values that lie far outside the 1.5 $\times$ IQR range, which can distort gradient descent steps.
3. **Imbalance**: A major target skewness was identified. Only 948 out of 5,630 rows are positive instances of churn (16.8% churn rate).

---

## 3. Preprocessing Pipeline

To clean the raw excel sheet and make it suitable for from-scratch model training, we built a modular preprocessing pipeline consisting of:

1. **Imputation (`clean_data.py`)**:
   * We applied **median imputation** on all columns containing missing values. Median was preferred over mean to avoid bias introduced by the right-skewed distributions and extreme outliers.
2. **Feature Removal (`clean_data.py`)**:
   * The `CustomerID` column was removed as it contains no predictive information.
3. **One-Hot Encoding (`encoders.py`)**:
   * All categorical features were converted to dummy variables.
   * `drop_first=True` was applied to eliminate redundant columns and prevent multicollinearity.
4. **Min-Max Scaling (`encoders.py`)**:
   * Continuous variables were scaled to `[0, 1]` using the formula:
     \[X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}\]
   * This ensures that our from-scratch Gradient Descent optimizer converges efficiently without oscillation.
5. **Class Balancing (`imbalance.py`)**:
   * We downsampled the majority class (`Churn = 0`) to match the minority class count (`948`).
   * This resulted in a balanced dataset of **1,896 samples** (948 churned, 948 retained).

---

## 4. Complete Exploratory Data Analysis (EDA)

The project includes an exploratory notebook (`notebooks/01_exploratory_data_analysis.ipynb`) which contains the following 6 visualizations and written insights:

1. **Churn Imbalance Countplot**: Shows the count of active vs. churned users.
   * *Insight*: Highlights that the raw dataset is heavily biased toward retained users, indicating that standard accuracy would be a misleading validation metric.
2. **Tenure Distribution Histogram**: Displays the tenure range of customers.
   * *Insight*: Most customers are within their first few months, indicating a high risk of early-stage churn.
3. **Cashback Amount Boxplot (by Churn)**: Compares cashback earned by active vs. churned users.
   * *Insight*: Active users tend to have higher cashback, indicating that cashback rewards are an effective retention tool.
4. **Outlier Check Boxplots**: Pairwise boxplots of continuous numerical variables.
   * *Insight*: Visually proves the presence of significant outlier points, validating the choice of median imputation.
5. **Balanced Churn Countplot**: Countplot showing class distributions after applying random undersampling.
   * *Insight*: Confirms both classes contain exactly 948 samples, eliminating model training bias.
6. **Correlation Heatmap**: Pairwise Pearson correlation heatmap of features.
   * *Insight*: Shows a strong negative correlation between `Tenure` and `Churn`, indicating that long-term customers are highly unlikely to leave.

---

## 5. Model Results and Comparison

We implemented two classifiers completely from scratch:
1. **Logistic Regression (Baseline)**: Implemented using vectorized gradient descent with a learning rate of `0.1` and `2000` iterations.
2. **K-Nearest Neighbors (KNN, Additional Model)**: Implemented with $k=5$ using vectorized Euclidean distance calculations.

### 5.1 Dataset Splits
The balanced dataset (1,896 samples) was split into:
* **Training Set**: 1,327 samples (70%)
* **Validation Set**: 284 samples (15%)
* **Test Set**: 285 samples (15%)

### 5.2 Model Comparison Results

Using our custom metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`), we evaluated both models on the test set:

| Dataset / Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | | | | |
| *Logistic Regression* | 78.45% | 77.99% | 78.94% | 78.46% |
| *KNN (k=5)* | **85.00%** | **83.55%** | **86.97%** | **85.23%** |
| **Validation Set** | | | | |
| *Logistic Regression* | 77.82% | 80.31% | 72.86% | 76.40% |
| *KNN (k=5)* | **79.23%** | **81.89%** | **74.29%** | **77.90%** |
| **Test Set** | | | | |
| *Logistic Regression* | **78.25%** | **79.45%** | 78.38% | **78.91%** |
| *KNN (k=5)* | 77.19% | 76.10% | **81.76%** | 78.83% |

### 5.3 Comparative Analysis
* **Recall Performance**: KNN ($k=5$) achieved a significantly higher **Recall** on the test set (**81.76%** compared to **78.38%** for Logistic Regression). This means KNN is more capable of detecting actual churners.
* **F1-Score**: Both models achieve similar F1-scores (~78.9%), meaning they provide a solid, balanced classification boundary.
* **Generalization**: Neither model shows signs of overfitting, as validation and test set performance remains highly consistent with training performance.

---

## 6. Challenges Encountered and Resolutions

* **Challenge 1: Scikit-Learn Restrictions**
  * *Resolution*: All operations (sigmoid activation, cost functions, gradient descent updates, Euclidean distance matrices, split allocations, and validation metrics) were written using vectorized NumPy operations, avoiding standard external libraries.
* **Challenge 2: Sigmoid Numerical Instability**
  * *Resolution*: Large inputs into the exponent function caused float overflow/underflow. We resolved this by clipping values in the linear combination `z` between `-500` and `500` before passing them to the sigmoid function.
* **Challenge 3: Missing Excel Spreadsheets Reader**
  * *Resolution*: Encountered `ModuleNotFoundError: No module named 'openpyxl'` when pandas tried to read the raw Excel file. Resolved by installing `openpyxl` as a dependency.
* **Challenge 4: Git Path Errors**
  * *Resolution*: Git was freshly installed on the machine and not configured in the active environment. We resolved this by querying the exact path (`C:\Program Files\Git\cmd\git.exe`) and configuring local Git repository configurations to match your GitHub profile.

---

## 7. Plan for Milestone 2

For the final milestone (Milestone 2), we will execute the following plan:
1. **Third Model implementation**: Implement a from-scratch Gaussian Naive Bayes classifier (`src/models/naive_bayes.py`) to serve as a probabilistic baseline.
2. **Ensemble System (`ensemble.py`)**: Build a Hard Voting Ensemble classifier that combines the predictions of Logistic Regression, KNN, and Naive Bayes to boost generalization and robustness.
3. **Web Application Development (`app/`)**:
   * **Backend**: Build a Node/Express API to expose prediction endpoints and link Python scripts to the UI.
   * **Frontend**: Create a React dashboard to display customer retention insights, churn probability analysis, and allow users to run manual churn checks.
4. **Final Presentation & PDF Export**: Compile the final progress report, slide deck, and codebase for project submission.
