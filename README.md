# 🏦 Loan Approval Prediction

This project predicts **loan approval status** based on applicant demographic, financial, and credit history information using **machine learning classification models**.  
I implemented and compared **Logistic Regression**, **Random Forest**, and **Decision Tree** classifiers, with results visualized using confusion matrices.  

The dataset used is the [Kaggle Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) (provided in this repository).

---

## 📊 Features Used

The dataset contains a mix of **categorical** and **numerical** features, including:

- Applicant income  
- Co-applicant income  
- Loan amount and term  
- Credit history  
- Gender  
- Marital status  
- Education level  
- Employment status  
- Property area  

All **categorical features** were **one-hot encoded**, and **numerical features** were **scaled** using a **Scikit-learn Pipeline** and **ColumnTransformer**.

---

## 🧪 Models & Methods

- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)
- **Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`)
- **Pipeline** and **ColumnTransformer** for preprocessing
- **Classification Report**, **Confusion Matrix**, and metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
- **Matplotlib** & **Seaborn** for visualization

---

## 📈 Results

- **Logistic Regression Accuracy:** `90.75%`
- **Random Forest Accuracy:** `97.66%`
- **Decision Tree Accuracy:** `97.54%`

**Key Insight:**  
Random Forest and Decision Tree had almost identical performance. Decision Tree alone is a reliable choice, making Random Forest unnecessary for this dataset.  

Confusion matrices for all three models provide a clear view of classification performance.

---

## 🚀 How to Use

1. **Clone or download this repository**  
2. **Open the provided Google Colab notebook**  
3. **Upload the dataset**:
   - Download the dataset file (included in this repo).
   - In Colab, upload it using:
     ```python
     from google.colab import files
     files.upload()
     ```
4. Run the notebook to perform preprocessing, train models, and view results.

---

## 📁 Dataset

- **Source:** [Kaggle Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Target Variable:** Loan approval status (`Approved` or `Not Approved`)
- Dataset is provided in this repository.

---

## 🧠 Motivation

Automating loan approval prediction can:

- Speed up banking processes  
- Reduce manual evaluation errors  
- Provide fair and consistent decision-making  
- Help in risk assessment for financial institutions

---

## 📌 Requirements

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  

---

## 📜 License

This project is for educational and research purposes only. Dataset license details can be found at the source.

---

## 👤 Author

Muhammad Saad  
GitHub: [@muhammadsaad021](https://github.com/muhammadsaad021)
