# 📱 Telecom Customer Churn Predictor

**🚀 Project Overview**
Predicting which customers are likely to leave a telecom company (churn) to help improve retention and revenue. This project uses machine learning to identify key churn drivers and build predictive models.

**📊 Dataset**

* Source: [Kaggle](https://www.kaggle.com/datasets/mubeenshehzadi/customer-churn-dataset?select=Telco_Cusomer_Churn.csv)
* 7,043 customers, 21 features (demographics, services, billing, payment methods)
* Target: `Churn` (Yes/No)
* Observed imbalance: 73.5% retained vs. 26.5% churned

**🔍 Key Steps**

1. **Data Cleaning & Preparation**

   * Converted `TotalCharges` from object → float
   * Handled missing values and checked for skewness
   * Addressed class imbalance with SMOTE

2. **Exploration & Visualization**

   * Correlation analysis and feature distributions
   * Visualized churn patterns for categorical and numerical features

3. **Modeling & Evaluation**

   * Baseline models: Gradient Boosting, LightGBM, CatBoost, MLPClassifier
   * Hyperparameter tuning (GridSearchCV)
   * Performance metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC
   * Stratified 5-Fold Cross-Validation for robustness

4. **Feature Importance**

   * Top predictors: Tenure, Payment Method, Monthly Charges, Contract Type

5. **Deployment**

   * Final model: **CatBoost**
   * Saved pipeline for Streamlit or web deployment (`catboost_pipeline.pkl`)

**💡 Key Insights**

* Early-tenure customers and month-to-month contracts have the highest churn risk
* High monthly charges and electronic check payments increase churn probability
* Class imbalance affects predictive performance; balancing improves sensitivity

**📈 Key Metrics at a Glance**

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| **CatBoost**        | 0.776    | 0.568     | 0.650  | 0.606    | 0.841   |
| Gradient Boosting   | 0.769    | 0.554     | 0.660  | 0.602    | 0.840   |
| LightGBM            | 0.779    | 0.577     | 0.634  | 0.604    | 0.835   |
| Random Forest       | 0.765    | 0.546     | 0.684  | 0.607    | 0.828   |
| Logistic Regression | 0.743    | 0.511     | 0.765  | 0.612    | 0.838   |

**🛠 Recommendations & Future Work**

* Target high-risk customers with retention campaigns
* Collect additional usage and behavioral data for improved predictions
* Explore ensemble models to boost performance
* Deploy model for real-time churn monitoring

**🎯 Outcome**
The **CatBoost churn predictor** provides telecom operators with actionable insights to target at-risk customers, improve loyalty, and reduce churn. With deployment-ready pipelines, this solution enables **real-time churn monitoring** and supports data-driven decision-making for sustainable growth.

**🚀 Live Demo**
👉 https://telcocustomerchurnpredictor.streamlit.app/
