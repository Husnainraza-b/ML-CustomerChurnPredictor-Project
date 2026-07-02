# ecommerce-churn-predictions
## The Repository Structure

```text
ecommerce-churn-prediction/
│
├── data/                      # Local data only (Do NOT commit to GitHub)
│   ├── raw/                   # The original Kaggle CSV file
│   └── processed/             # The cleaned, scaled CSV output by your Data Engineer
│
├── notebooks/                 # Jupyter notebooks for exploration
│   └── 01_exploratory_data_analysis.ipynb  # The 6 visualizations and EDA
│
├── src/                       # The core Python codebase
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clean_data.py      # Missing values, outliers
│   │   └── encoders.py        # Custom One-Hot Encoding and Scaling
│   │
│   ├── models/                # From-scratch algorithm implementations
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   ├── naive_bayes.py
│   │   └── ensemble.py        # The Hard Voting logic
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py         # Custom Accuracy, Precision, Recall, F1 functions
│
├── app/                       # The web system for the VIVA demo
│   ├── frontend/              # React dashboard code
│   └── backend/               # Node/Express API to connect UI to Python scripts
│
├── reports/                   # Deliverables for submission
│   ├── milestone_1_report.pdf
│   └── final_presentation.pdf
│
├── .gitignore                 # Files Git should ignore (crucial!)
├── requirements.txt           # Python dependencies (NumPy, Pandas, etc.)
└── README.md                  # Project overview and setup instructions
```

## Workflow

1. Clone the repository.
2. Run `pip install -r requirements.txt` to install the allowed libraries.
3. Pull the latest changes before starting to write specific algorithms.
