"""Well-known public datasets for live agent evals, fetched by evals/fetch_public.py into evals/public/.

Every URL is a public mirror on an allowlisted host (external_data.SEARCH_DOMAINS); Kaggle needs auth, so none
point there. expect mirrors evals/make_golden.py EXPECT: finding is a diagnostics check name that must appear in
the `diagnostics` event (None: nothing required), must_drop lists columns the run_search input must drop, and cv
is the allowed cv scheme (a string, or a tuple of allowed ones). metric is the primary metric the purpose implies.
"""
import os

PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
TIME_CV = ("walk_forward", "purged")

PUBLIC = {
    "telco_churn": {
        "url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "target": "Churn", "task": "classification", "metric": "roc_auc",
        "purpose": "Rank our customers by how likely they are to cancel next month so the retention team calls the "
                   "riskiest first. The team has to explain each call, so readable reasons matter.",
        "expect": {"finding": "id_like", "must_drop": ["customerID"], "cv": "kfold"},
        "source": "IBM Telco Customer Churn sample (IBM/telco-customer-churn-on-icp4d on GitHub)",
        "license": "IBM sample data, Apache-2.0 repository",
    },
    "titanic": {
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "target": "Survived", "task": "classification", "metric": "accuracy",
        "purpose": "Predict which Titanic passengers survived from the passenger list, with a model simple enough "
                   "to explain to a history class.",
        "expect": {"finding": "id_like", "must_drop": ["PassengerId", "Name"], "cv": "kfold"},
        "source": "Titanic passenger manifest, Kaggle train split (datasciencedojo/datasets on GitHub)",
        "license": "public domain historical records",
    },
    "california_housing": {
        "url": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
        "target": "median_house_value", "task": "regression", "metric": "mae",
        "purpose": "Estimate the median house value of a California district from census numbers to sanity check "
                   "listing prices. Keep the typical error in dollars small.",
        "expect": {"finding": "adversarial_drift", "must_drop": [], "cv": "kfold"},
        "source": "California Housing, 1990 US Census (Pace and Barry 1997), ageron/handson-ml2 on GitHub",
        "license": "public US Census data; repository Apache-2.0",
    },
    "bank_marketing": {
        "url": "https://raw.githubusercontent.com/DataCanvasIO/Hypernets/master/hypernets/tabular/datasets/bank-uci.csv.gz",
        "target": "y", "task": "classification", "metric": "roc_auc",
        "purpose": "Choose which clients to phone in the next term deposit campaign. The model scores clients "
                   "before anyone calls them.",
        "expect": {"finding": "imbalance", "must_drop": ["id", "duration"], "cv": "kfold"},
        "source": "UCI Bank Marketing (Moro et al. 2014), 10% bank.csv sample with an id column, "
                  "DataCanvasIO/Hypernets on GitHub",
        "license": "UCI, CC BY 4.0",
    },
    "adult_income": {
        "url": "https://huggingface.co/datasets/scikit-learn/adult-census-income/resolve/main/adult.csv",
        "target": "income", "task": "classification", "metric": "roc_auc",
        "purpose": "Predict whether a person earns more than 50K a year from census answers, to audit a "
                   "benefits eligibility rule.",
        "expect": {"finding": "missing_placeholders", "must_drop": [], "cv": "kfold"},
        "source": "UCI Adult / Census Income (Kohavi 1996), scikit-learn/adult-census-income on Hugging Face",
        "license": "UCI, CC BY 4.0",
    },
    "credit_default": {
        "url": "https://huggingface.co/datasets/scikit-learn/credit-card-clients/resolve/main/UCI_Credit_Card.csv",
        "target": "default.payment.next.month", "task": "classification", "metric": "roc_auc",
        "purpose": "Score credit card clients by their risk of defaulting on next month's payment so we can lower "
                   "limits early.",
        "expect": {"finding": "id_like", "must_drop": ["ID"], "cv": "kfold"},
        "source": "UCI Default of Credit Card Clients, Taiwan 2005 (Yeh and Lien 2009), "
                  "scikit-learn/credit-card-clients on Hugging Face",
        "license": "UCI, CC BY 4.0",
    },
    "bike_sharing_hourly": {
        "url": "https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data/p1ch4/"
               "bike-sharing-dataset/hour.csv",
        "target": "cnt", "task": "regression", "metric": "rmse",
        "purpose": "Forecast how many bikes will be rented each hour so we can plan how many bikes to put out at "
                   "the stations tomorrow.",
        "expect": {"finding": "leakage", "must_drop": ["casual", "registered", "instant"], "cv": TIME_CV},
        "source": "UCI Bike Sharing, Capital Bikeshare 2011-2012 hourly (Fanaee-T and Gama 2013), "
                  "deep-learning-with-pytorch/dlwpt-code on GitHub",
        "license": "UCI, CC BY 4.0",
    },
    "walmart_sales": {
        "url": "https://raw.githubusercontent.com/eosphoros-ai/DB-GPT/main/docker/examples/excel/Walmart_Sales.csv",
        "target": "Weekly_Sales", "task": "regression", "metric": "mae",
        "purpose": "Forecast next week's sales for each of our 45 stores so managers can plan staff and stock.",
        "expect": {"finding": "time_order", "must_drop": [], "cv": TIME_CV},
        "source": "Walmart weekly store sales 2010-2012 (Kaggle Walmart Recruiting derivative), "
                  "eosphoros-ai/DB-GPT examples on GitHub",
        "license": "Kaggle competition data, research use; repository MIT",
    },
    "pima_diabetes": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "target": "Outcome", "task": "classification", "metric": "roc_auc",
        "purpose": "Screen clinic patients for diabetes so nurses know who needs a follow-up blood test. Missing a "
                   "real case is worse than an extra test.",
        "expect": {"finding": None, "must_drop": [], "cv": "kfold"},
        "source": "Pima Indians Diabetes (NIDDK, Smith et al. 1988), plotly/datasets on GitHub",
        "license": "public research data (UCI, later Kaggle CC0); repository MIT",
    },
    "card_fraud_sample": {
        "url": "https://raw.githubusercontent.com/Classiq/classiq-library/main/applications/finance/resources/"
               "creditcard.csv",
        "target": "Class", "task": "classification", "metric": "roc_auc",
        "purpose": "Catch fraudulent card transactions. Fraud is rare, so we care about ranking the riskiest "
                   "transactions at the top for review.",
        "expect": {"finding": "leakage", "must_drop": ["Time"], "cv": "kfold"},
        "source": "ULB Credit Card Fraud (Dal Pozzolo et al. 2015), 948-row sample in Classiq/classiq-library",
        "license": "ULB, Database Contents License (DbCL) v1.0; repository Apache-2.0",
    },
}
