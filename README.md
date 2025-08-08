# Diabetes Prediction System

## Overview
The Diabetes Prediction System is a machine learning project that predicts whether a person is likely to have diabetes based on key medical parameters.  
It uses patient health data such as glucose levels, BMI, age, and other relevant features to make predictions.

This project demonstrates:
- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Model training and evaluation  
- Deployment-ready prediction pipeline  

## Objective
To build a predictive system that can assist healthcare professionals and individuals in identifying diabetes risk early, enabling timely lifestyle changes or medical intervention.

## Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Gradio  
- **ML Algorithms:** Logistic Regression, Decision Tree, SVM, Random Forest  
- **Dataset:** Pima Indians Diabetes Dataset  
- **Environment:** Jupyter Notebook / Google Colab  

## Dataset Description
The dataset contains medical diagnostic measurements for women of Pima Indian heritage aged 21 and older.

**Features:**
- **Pregnancies:** Number of pregnancies  
- **Glucose:** Plasma glucose concentration  
- **BloodPressure:** Diastolic blood pressure (mm Hg)  
- **SkinThickness:** Triceps skinfold thickness (mm)  
- **Insulin:** 2-Hour serum insulin (mu U/ml)  
- **BMI:** Body mass index  
- **DiabetesPedigreeFunction:** Likelihood of diabetes based on family history  
- **Age:** Age in years  
- **Outcome:** 0 = Non-diabetic, 1 = Diabetic  

## Approach

### Data Preprocessing
- Handled missing/zero values in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI  
- Normalized continuous features using StandardScaler  

### Exploratory Data Analysis (EDA)
- Visualized feature distributions using histograms and boxplots  
- Checked correlation between features using a heatmap  

### Model Training
- Compared Logistic Regression, Decision Tree, SVM, and Random Forest  
- Used train-test split (80-20) for evaluation  

### Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC Curve  
- Achieved highest accuracy with **Random Forest (80.2%)**  

### Deployment
- Created a Gradio interface for real-time predictions  

## Results
- **Best Model:** Random Forest Classifier  
- **Accuracy:** 80.2%  

## Future Improvements
- Deploy as a Flask Web App or Streamlit Dashboard  
- Add more features such as physical activity, diet patterns, and lifestyle habits  
- Train with larger datasets for improved accuracy  

## License
This project is open-source and available under the MIT License.
