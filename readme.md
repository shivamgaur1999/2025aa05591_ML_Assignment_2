# ML Assignment 2: Customer Personality Analysis

## Problem Statement

Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors, and concerns of different types of customers.

The objective is to:

- Build and compare six different classification algorithms on customer response data.
- Evaluate each model using multiple performance metrics.
- Identify the most effective model for predicting customer responses.
- Deploy an interactive web application for real-time prediction.

By accurately predicting customer responses, businesses can implement targeted marketing strategies, enhance customer satisfaction, and reduce marketing costs.

## Dataset Description

**Dataset**: Customer Personality Analysis

**Source**: [Kaggle](https://www.kaggle.com/imakash3011/customer-personality-analysis)

**Total Instances**: 2,240 customer records

**Total Features**: 28 (after removing ID and Dt_Customer)

**Features after Preprocessing**: 36 (after one-hot encoding categorical variables)

**Target Variable**: Response (Binary: 1/0)

### Feature Categories

1. **Demographic Information**:
   - **Year_Birth**: Year of birth
   - **Education**: Education level
   - **Marital_Status**: Marital status
   - **Income**: Yearly household income
   - **Kidhome**: Number of children in the household
   - **Teenhome**: Number of teenagers in the household

2. **Purchasing Behavior**:
   - **MntWines**: Amount spent on wine in the last 2 years
   - **MntFruits**: Amount spent on fruits in the last 2 years
   - **MntMeatProducts**: Amount spent on meat in the last 2 years
   - **MntFishProducts**: Amount spent on fish in the last 2 years
   - **MntSweetProducts**: Amount spent on sweets in the last 2 years
   - **MntGoldProds**: Amount spent on gold in the last 2 years

3. **Campaign Responses**:
   - **AcceptedCmp1**: Accepted offer in the 1st campaign
   - **AcceptedCmp2**: Accepted offer in the 2nd campaign
   - **AcceptedCmp3**: Accepted offer in the 3rd campaign
   - **AcceptedCmp4**: Accepted offer in the 4th campaign
   - **AcceptedCmp5**: Accepted offer in the 5th campaign

### Data Preprocessing

The following preprocessing steps were applied:

- Removed ID and Dt_Customer columns (not predictive features).
- Converted Income to numeric and handled missing values.
- Encoded binary categorical variables (Complain, AcceptedCmp1-5) to 0/1.
- One-hot encoded multi-category variables (Education, Marital_Status).
- Converted target variable Response to binary (0: No, 1: Yes).
- Applied Standard Scaling to features for models requiring normalized data (Logistic Regression, kNN).

## Models Used

Six classification models were implemented and evaluated on the dataset. All models were trained using a 70-30 train-test split with stratification to maintain class distribution.

### Comparison Table

| ML Model Name       | Accuracy | AUC     | Precision | Recall | F1      | MCC     |
|---------------------|----------|---------|-----------|--------|---------|---------|
| Logistic Regression | 0.885417 | 0.873191| 0.725490  | 0.37   | 0.490066| 0.464351|
| Decision Tree       | 0.875000 | 0.660122| 0.629032  | 0.39   | 0.481481| 0.430175|
| kNN                 | 0.860119 | 0.782002| 0.565217  | 0.26   | 0.356164| 0.317163|
| Naive Bayes         | 0.781250 | 0.738173| 0.340136  | 0.50   | 0.404858| 0.284463|
| Random Forest       | 0.888393 | 0.870245| 0.857143  | 0.30   | 0.444444| 0.466523|
| XGBoost             | 0.886905 | 0.871372| 0.714286  | 0.40   | 0.512821| 0.479059|

### Model Performance Observations

| ML Model Name       | Observation about Model Performance                                                              |
|---------------------|--------------------------------------------------------------------------------------------------|
| Logistic Regression | High accuracy and AUC, indicating good overall performance. Precision is high, but recall is lower. Balanced F1 score and MCC. |
| Decision Tree       | Moderate accuracy and AUC. Precision and recall are relatively balanced. Decent F1 score and MCC. |
| kNN                 | Lower accuracy and AUC compared to other models. Struggles with precision and recall, resulting in lower F1 score and MCC. |
| Naive Bayes         | Lowest accuracy and AUC. High recall but low precision, indicating many false positives. Lower F1 score and MCC. |
| Random Forest       | Highest accuracy and high AUC. Highest precision, but lower recall. Balanced F1 score and MCC. |
| XGBoost             | High accuracy and AUC. Balanced precision and recall, resulting in the highest F1 score and MCC. |

## Key Findings

**Best Models by Metric**:
- **Accuracy**: Random Forest (0.888393)
- **AUC (Discriminative Power)**: Logistic Regression (0.873191)
- **Precision**: Random Forest (0.857143)
- **Recall**: Naive Bayes (0.50)
- **F1 Score**: XGBoost (0.512821)
- **MCC**: XGBoost (0.479059)

**Insights**:
- Logistic Regression and Random Forest show strong overall performance, with Random Forest achieving the highest accuracy.
- XGBoost provides a good balance between precision and recall, achieving the highest F1 score and MCC.
- Naive Bayes excels in recall but suffers from low precision, indicating many false positives.

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/ML_Assignment_2.git
    cd ML_Assignment_2
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the Models (Optional, Pre-trained Models Included)**:
    ```bash
    python train.py
    ```

4. **Run the Streamlit App Locally**:
    ```bash
    streamlit run app.py
    ```

5. **Access the App**: Open your browser and navigate to `http://localhost:8501`.

## Streamlit Application Features

The interactive web application is organized into 3 tabs for comprehensive analysis:

### Tab 1: 📊 Dataset Description
- Overview Metrics: Total customers, features count, response rate, and data quality
- Problem Statement: Business context and objectives
- Feature Categories: Interactive exploration of demographics, purchasing behavior, and campaign responses
- Visualizations: Distribution charts for education, marital status, purchasing behavior, and response
- Data Sample: Preview of first 10 customer records
- Statistical Summary: Numerical and categorical feature statistics
- Download Options: Export full dataset as CSV

### Tab 2: 📈 Training Analysis
- Performance Overview: Best model identification and accuracy metrics
- Results Table: Complete comparison of all 6 models across 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Visual Comparison:
  - Bar Charts: Individual metric performance for each model
  - Heatmap: Model vs metric performance matrix
  - Confusion Matrix: Display confusion matrix for selected model
- Model Rankings: Sort models by any metric
- Training Configuration: Dataset split details and preprocessing steps

### Tab 3: 🎯 Try It Out
- Model Selection: Choose from 6 trained models
- Two Input Methods:
  - 📁 Upload CSV File: Upload test dataset or training dataset for batch predictions
  - ✏️ Manual Input: Interactive form with customer attributes
- Batch Predictions: Automatic preprocessing and feature alignment
- Downloadable Prediction Results: Download predictions as CSV
- Performance Evaluation: Confusion matrix and classification report for uploaded data
- Real-time Single Customer Prediction: Probability distribution visualization

## Conclusion

This project demonstrates the use of multiple classification models to predict customer responses to marketing campaigns. By accurately predicting customer responses, businesses can implement targeted marketing strategies, enhance customer satisfaction, and reduce marketing costs.

## Acknowledgements

The dataset for this project is provided by Dr. Omar Romero-Hernandez.

## Author

**Name:** Shivam Gaur

**Roll Number:** 2025AA05591

**Program:** M.Tech (AI/ML)

---

## View Live App