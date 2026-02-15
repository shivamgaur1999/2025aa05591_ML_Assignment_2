import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Set the page configuration
st.set_page_config(
    page_title="Customer Personality Analysis",
    page_icon="📊",
    layout="wide"
)

# Title of the app
st.title("📊 Customer Personality Analysis System")

@st.cache_resource
def load_model(model_name):
    """Load the selected model"""
    model_files = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Decision Tree": "models/decision_tree.pkl",
        "kNN": "models/knn.pkl",
        "Naive Bayes": "models/naive_bayes.pkl",
        "Random Forest": "models/random_forest.pkl",
        "XGBoost": "models/xgboost.pkl"
    }
    with open(model_files[model_name], 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    """Load the feature scaler"""
    with open('models/scaler.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_names():
    """Load feature names"""
    with open('models/feature_names.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    """Load the full dataset"""
    return pd.read_csv('data/dataset.csv')

@st.cache_data
def load_results():
    """Load model comparison results"""
    return pd.read_csv('models/model_comparison_results.csv')

def preprocess_data(df):
    """Preprocess the data to match training format"""
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    if 'Dt_Customer' in df.columns:
        df = df.drop('Dt_Customer', axis=1)

    if 'Income' in df.columns:
        df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
        df['Income'].fillna(df['Income'].median(), inplace=True)

    binary_cols = ['Complain', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({1: 1, 0: 0})

    if 'Response' in df.columns:
        df['Response'] = df['Response'].map({1: 1, 0: 0})

    categorical_cols = ['Education', 'Marital_Status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded

# Define the tabs
tab1, tab2, tab3 = st.tabs(["📊 Dataset Description", "📈 Training Analysis", "🎯 Try It Out"])

with tab1:
    try:
        df = load_data()

        st.header("📋 Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Customers", f"{len(df):,}")

        with col2:
            st.metric("Features", df.shape[1] - 1)

        with col3:
            response_pct = (df['Response'] == 1).sum() / len(df) * 100
            st.metric("Response Rate", f"{response_pct:.1f}%")

        with col4:
            st.metric("Data Quality", "100%" if df.isnull().sum().sum() == 0 else "Has Missing")

        st.divider()

        st.header("🎯 Problem Statement")
        st.markdown("""
        **Objective:** Predict which customers are likely to respond to the last marketing campaign based on their
        demographic information, purchasing behavior, and previous campaign responses.

        **Business Impact:**
        - Targeted marketing increases campaign efficiency
        - Personalized offers can enhance customer satisfaction
        - Reducing marketing costs by focusing efforts on likely responders
        """)

        st.divider()

        st.header("🏷️ Feature Categories")

        subtab1, subtab2, subtab3, subtab4 = st.tabs(["Demographic", "Purchasing Behavior", "Campaign Responses", "Target"])

        with subtab1:
            st.subheader("Demographic Information")
            st.markdown("""
            - **Year_Birth**: Year of birth
            - **Education**: Education level
            - **Marital_Status**: Marital status
            - **Income**: Yearly household income
            - **Kidhome**: Number of children in the household
            - **Teenhome**: Number of teenagers in the household
            """)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                df['Education'].value_counts().plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
                ax.set_title('Education Level Distribution')
                ax.set_xlabel('Education')
                ax.set_ylabel('Count')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                df['Marital_Status'].value_counts().plot(kind='bar', ax=ax, color=['#2ca02c', '#d62728'])
                ax.set_title('Marital Status Distribution')
                ax.set_xlabel('Marital Status')
                ax.set_ylabel('Count')
                st.pyplot(fig)

        with subtab2:
            st.subheader("Purchasing Behavior")
            st.markdown("""
            - **MntWines**: Amount spent on wine in last 2 years
            - **MntFruits**: Amount spent on fruits in last 2 years
            - **MntMeatProducts**: Amount spent on meat in last 2 years
            - **MntFishProducts**: Amount spent on fish in last 2 years
            - **MntSweetProducts**: Amount spent on sweets in last 2 years
            - **MntGoldProds**: Amount spent on gold in last 2 years
            """)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df['MntWines'], bins=30, color='skyblue', edgecolor='black')
                ax.set_title('Wine Spending Distribution')
                ax.set_xlabel('Amount')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                df['MntFruits'].value_counts().plot(kind='bar', ax=ax, color=['#9467bd', '#8c564b', '#e377c2'])
                ax.set_title('Fruit Spending Distribution')
                ax.set_xlabel('Amount')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)

        with subtab3:
            st.subheader("Campaign Responses")
            st.markdown("""
            - **AcceptedCmp1**: Accepted offer in 1st campaign
            - **AcceptedCmp2**: Accepted offer in 2nd campaign
            - **AcceptedCmp3**: Accepted offer in 3rd campaign
            - **AcceptedCmp4**: Accepted offer in 4th campaign
            - **AcceptedCmp5**: Accepted offer in 5th campaign
            """)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum().plot(kind='bar', ax=ax, color=['#17becf', '#bcbd22', '#7f7f7f', '#ff7f0e', '#1f77b4'])
                ax.set_title('Campaign Acceptance')
                ax.set_xlabel('Campaign')
                ax.set_ylabel('Count')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Campaign Response Correlation')
                st.pyplot(fig)

        with subtab4:
            st.subheader("Target Variable")
            st.markdown("""
            - **Response**: Whether the customer accepted the offer in the last campaign

            **Class Distribution:**
            """)

            response_counts = df['Response'].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("No Response", f"{response_counts.get(0, 0):,} ({response_counts.get(0, 0)/len(df)*100:.1f}%)")
                st.metric("Response", f"{response_counts.get(1, 0):,} ({response_counts.get(1, 0)/len(df)*100:.1f}%)")

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#2ca02c', '#d62728']
                response_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('Response Distribution')
                ax.set_ylabel('')
                st.pyplot(fig)

        st.divider()

        st.header("📄 Data Sample")
        st.dataframe(df.head(10), use_container_width=True)

        st.divider()

        st.header("📊 Statistical Summary")

        summary_type = st.radio("Select Summary Type:", ["Numerical Features", "Categorical Features"], index=0)

        if summary_type == "Numerical Features":
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        else:
            categorical_cols = df.select_dtypes(include=['object']).columns
            st.dataframe(df[categorical_cols].describe(), use_container_width=True)

        st.divider()

        st.header("📥 Download Dataset")

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset as CSV",
            data=csv,
            file_name="customer_personality_analysis.csv",
            mime="text/csv"
        )

        test_df = pd.read_csv('data/test.csv')
        st.download_button(
            label="Download Test Dataset as CSV",
            data=test_df.to_csv(index=False),
            file_name="test.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("❌ Dataset file not found! Please ensure 'data/dataset.csv' exists.")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")

with tab2:
    try:
        results_df = load_results()

        st.header("📊 Performance Overview")

        col1, col2, col3, col4 = st.columns(4)

        best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        best_accuracy = results_df['Accuracy'].max()

        with col1:
            st.metric("Best Model", best_model)

        with col2:
            st.metric("Best Accuracy", f"{best_accuracy:.2%}")

        with col3:
            st.metric("Best AUC", f"{results_df['AUC'].max():.4f}")

        with col4:
            st.metric("Models Trained", len(results_df))

        st.divider()

        st.header("📋 Complete Results Table")

        st.dataframe(
            results_df.style.highlight_max(
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                color='lightgreen'
            ),
            use_container_width=True
        )

        st.divider()

        st.header("📊 Visual Comparison")

        subtab1, subtab2 = st.tabs(["Bar Charts", "Heatmap"])

        with subtab1:
            st.subheader("Metric Comparison - Bar Charts")

            metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

            for i in range(0, 6, 2):
                col1, col2 = st.columns(2)

                with col1:
                    metric = metrics[i]
                    fig, ax = plt.subplots(figsize=(8, 5))
                    results_df.plot(
                        x='Model',
                        y=metric,
                        kind='bar',
                        ax=ax,
                        legend=False,
                        color='steelblue'
                    )
                    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
                    ax.set_xlabel('')
                    ax.set_ylabel(metric)
                    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
                    ax.grid(axis='y', alpha=0.3)

                    best_idx = results_df[metric].idxmax()
                    ax.patches[best_idx].set_color('green')
                    ax.patches[best_idx].set_alpha(0.7)

                    st.pyplot(fig)

                if i + 1 < len(metrics):
                    with col2:
                        metric = metrics[i + 1]
                        fig, ax = plt.subplots(figsize=(8, 5))
                        results_df.plot(
                            x='Model',
                            y=metric,
                            kind='bar',
                            ax=ax,
                            legend=False,
                            color='steelblue'
                        )
                        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
                        ax.set_xlabel('')
                        ax.set_ylabel(metric)
                        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
                        ax.grid(axis='y', alpha=0.3)

                        best_idx = results_df[metric].idxmax()
                        ax.patches[best_idx].set_color('green')
                        ax.patches[best_idx].set_alpha(0.7)

                        st.pyplot(fig)

        with subtab2:
            st.subheader("Heatmap - Model vs Metric Performance")

            fig, ax = plt.subplots(figsize=(10, 6))

            heatmap_data = results_df.set_index('Model')[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.4f',
                cmap='RdYlGn',
                center=0.7,
                ax=ax,
                cbar_kws={'label': 'Score'}
            )

            ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
            st.pyplot(fig)

        # Add confusion matrix below heatmap
        st.divider()
        st.header("🔍 Confusion Matrix")

        # Select model for confusion matrix
        model_choice = st.selectbox(
            "Select Model for Confusion Matrix:",
            results_df['Model'].unique()
        )

        # Load the model and data
        model = load_model(model_choice)
        df = load_data()
        df_processed = preprocess_data(df)
        X = df_processed.drop('Response', axis=1)
        y = df_processed['Response']

        # Scale the data if necessary
        if model_choice in ["Logistic Regression", "kNN"]:
            scaler = load_scaler()
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X)

        # Calculate confusion matrix
        cm = confusion_matrix(y, predictions)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Response', 'Response'],
            yticklabels=['No Response', 'Response'],
            ax=ax
        )
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'Confusion Matrix - {model_choice}')
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("❌ Results file not found! Please ensure models have been trained.")
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")

with tab3:
    st.header("⚙️ Configuration")
    col1, col2 = st.columns(2)

    model_choice = col1.selectbox(
        "Select Model:",
        [
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ],
        help="Choose which model to use for predictions"
    )

    input_method = col2.radio(
        "How would you like to provide data?",
        ["📁 Upload CSV File", "✏️ Manual Input"],
        index=0
    )

    st.divider()

    if input_method == "📁 Upload CSV File":
        st.header("📁 Upload Test Dataset")

        st.info("""
        Upload a CSV file containing customer data in the same format as the training data.
        The file should include all required features (without ID and Response columns).
        """)

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                st.success(f"✅ File uploaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")

                with st.expander("📋 View Data Preview"):
                    st.dataframe(df.head(10), use_container_width=True)

                st.divider()

                with st.spinner("Preprocessing data..."):
                    has_labels = 'Response' in df.columns
                    if has_labels:
                        y_true = df['Response']

                    df_processed = preprocess_data(df.copy())

                    if 'Response' in df_processed.columns:
                        X = df_processed.drop('Response', axis=1)
                    else:
                        X = df_processed

                    feature_names = load_feature_names()
                    for col in feature_names:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_names]

                model = load_model(model_choice)

                with st.spinner(f"Making predictions with {model_choice}..."):
                    if model_choice in ["Logistic Regression", "kNN"]:
                        scaler = load_scaler()
                        X_scaled = scaler.transform(X)
                        predictions = model.predict(X_scaled)
                        predictions_proba = model.predict_proba(X_scaled)
                    else:
                        predictions = model.predict(X)
                        predictions_proba = model.predict_proba(X)

                st.header("🎯 Prediction Results")

                col1, col2 = st.columns(2)

                response_count = np.sum(predictions == 1)
                no_response_count = np.sum(predictions == 0)

                with col1:
                    st.metric(
                        "Predicted Responders",
                        response_count,
                        delta=f"{response_count/len(predictions)*100:.1f}%",
                        delta_color="inverse"
                    )

                with col2:
                    st.metric(
                        "Predicted Non-Responders",
                        no_response_count,
                        delta=f"{no_response_count/len(predictions)*100:.1f}%"
                    )

                st.divider()

                st.subheader("📊 Detailed Predictions")

                results_df = pd.DataFrame({
                    'Customer Index': range(len(predictions)),
                    'Prediction': ['Will Respond' if p == 1 else 'Will Not Respond' for p in predictions],
                    'Response Probability': [f"{p[1]:.2%}" for p in predictions_proba],
                    'Confidence': [f"{max(p):.2%}" for p in predictions_proba]
                })

                st.dataframe(results_df, use_container_width=True)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{model_choice.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )

                if has_labels:
                    st.divider()
                    st.header("📈 Model Evaluation")

                    st.info("True labels detected in uploaded data. Showing performance metrics.")

                    accuracy = accuracy_score(y_true, predictions)
                    precision = precision_score(y_true, predictions, zero_division=0)
                    recall = recall_score(y_true, predictions, zero_division=0)
                    f1 = f1_score(y_true, predictions, zero_division=0)

                    try:
                        auc = roc_auc_score(y_true, predictions_proba[:, 1])
                    except:
                        auc = None

                    try:
                        mcc = matthews_corrcoef(y_true, predictions)
                    except:
                        mcc = None

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2%}")
                        st.metric("Precision", f"{precision:.4f}")

                    with col2:
                        st.metric("Recall", f"{recall:.4f}")
                        st.metric("F1 Score", f"{f1:.4f}")

                    with col3:
                        if auc is not None:
                            st.metric("AUC", f"{auc:.4f}")
                        if mcc is not None:
                            st.metric("MCC", f"{mcc:.4f}")

                    st.divider()

                    st.subheader("📊 Confusion Matrix")

                    cm = confusion_matrix(y_true, predictions)

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['No Response', 'Response'],
                            yticklabels=['No Response', 'Response'],
                            ax=ax
                        )
                        ax.set_ylabel('True Label')
                        ax.set_xlabel('Predicted Label')
                        ax.set_title(f'Confusion Matrix - {model_choice}')
                        st.pyplot(fig)

                    with col2:
                        st.markdown("### Classification Report")
                        report = classification_report(
                            y_true, predictions,
                            target_names=['No Response', 'Response'],
                            output_dict=True
                        )
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

        else:
            st.info("👆 Upload a CSV file to get started")

    else:
        st.header("✏️ Manual Input")

        st.info("Enter customer information below to predict response probability for a single customer.")

        with st.form("manual_input_form"):
            st.subheader("Demographics")

            col1, col2 = st.columns(2)

            with col1:
                year_birth = st.number_input("Year of Birth", min_value=1900, max_value=2023, value=1980)
                education = st.selectbox("Education Level", ["Graduation", "PhD", "Master", "Basic", "2n Cycle"])

            with col2:
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Together", "Divorced", "Widow", "Alone", "Absurd", "YOLO"])
                income = st.number_input("Yearly Income ($)", min_value=0, value=50000)

            st.divider()
            st.subheader("Purchasing Behavior")

            col1, col2, col3 = st.columns(3)

            with col1:
                mnt_wines = st.number_input("Amount Spent on Wines ($)", min_value=0, value=100)
                mnt_fruits = st.number_input("Amount Spent on Fruits ($)", min_value=0, value=50)

            with col2:
                mnt_meat_products = st.number_input("Amount Spent on Meat Products ($)", min_value=0, value=200)
                mnt_fish_products = st.number_input("Amount Spent on Fish Products ($)", min_value=0, value=50)

            with col3:
                mnt_sweet_products = st.number_input("Amount Spent on Sweet Products ($)", min_value=0, value=30)
                mnt_gold_prods = st.number_input("Amount Spent on Gold Products ($)", min_value=0, value=100)

            st.divider()
            st.subheader("Campaign Responses")

            col1, col2, col3 = st.columns(3)

            with col1:
                accepted_cmp1 = st.selectbox("Accepted Campaign 1", ["No", "Yes"])
                accepted_cmp2 = st.selectbox("Accepted Campaign 2", ["No", "Yes"])

            with col2:
                accepted_cmp3 = st.selectbox("Accepted Campaign 3", ["No", "Yes"])
                accepted_cmp4 = st.selectbox("Accepted Campaign 4", ["No", "Yes"])

            with col3:
                accepted_cmp5 = st.selectbox("Accepted Campaign 5", ["No", "Yes"])

            st.divider()

            submit_button = st.form_submit_button("🎯 Predict Response")

        if submit_button:
            input_data = {
                'Year_Birth': year_birth,
                'Education': education,
                'Marital_Status': marital_status,
                'Income': income,
                'MntWines': mnt_wines,
                'MntFruits': mnt_fruits,
                'MntMeatProducts': mnt_meat_products,
                'MntFishProducts': mnt_fish_products,
                'MntSweetProducts': mnt_sweet_products,
                'MntGoldProds': mnt_gold_prods,
                'AcceptedCmp1': 1 if accepted_cmp1 == "Yes" else 0,
                'AcceptedCmp2': 1 if accepted_cmp2 == "Yes" else 0,
                'AcceptedCmp3': 1 if accepted_cmp3 == "Yes" else 0,
                'AcceptedCmp4': 1 if accepted_cmp4 == "Yes" else 0,
                'AcceptedCmp5': 1 if accepted_cmp5 == "Yes" else 0
            }

            df = pd.DataFrame([input_data])

            df_processed = preprocess_data(df.copy())

            feature_names = load_feature_names()
            for col in feature_names:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            X = df_processed[feature_names]

            model = load_model(model_choice)

            if model_choice in ["Logistic Regression", "kNN"]:
                scaler = load_scaler()
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
                prediction_proba = model.predict_proba(X_scaled)[0]
            else:
                prediction = model.predict(X)[0]
                prediction_proba = model.predict_proba(X)[0]

            st.divider()
            st.header("🎯 Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction == 1:
                    st.error("### ⚠️ WILL RESPOND")
                else:
                    st.success("### ✅ WILL NOT RESPOND")

            with col2:
                st.metric("Response Probability", f"{prediction_proba[1]:.2%}")

            with col3:
                st.metric("Confidence", f"{max(prediction_proba):.2%}")

            st.divider()
            st.subheader("📊 Probability Distribution")

            fig, ax = plt.subplots(figsize=(10, 4))
            categories = ['Will Not Respond', 'Will Respond']
            probabilities = [prediction_proba[0], prediction_proba[1]]
            colors = ['green', 'red']

            bars = ax.barh(categories, probabilities, color=colors, alpha=0.7)
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)

            for bar, prob in zip(bars, probabilities):
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob:.2%}', ha='left', va='center', fontweight='bold')

            st.pyplot(fig)