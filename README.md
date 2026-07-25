# Credit-Approval-app
<img width="1840" height="840" alt="Screenshot 2026-07-25 123955" src="https://github.com/user-attachments/assets/0ec39d0e-21ac-4ff8-8e3c-dfed9006011e" />
<img width="1908" height="818" alt="Screenshot 2026-07-25 124005" src="https://github.com/user-attachments/assets/84abce8f-d897-489b-b74a-aadcf2e0a09b" />
<img width="1472" height="887" alt="Screenshot 2026-07-25 124128" src="https://github.com/user-attachments/assets/71e2a319-b7e3-4436-8ef6-72a429f4ac08" />

---
The dataset used in this project contains 51,000+ anonymized banking records.
It includes internal bank performance data merged with CIBIL credit bureau history.

**Data Source:** [Kaggle - CIBIL and Bank of Baroda Credit Data](https://www.kaggle.com/datasets/sudhirkumarjoon/cibil-and-bank-of-baroda-credit-data)

### 📖 Overview
This is a Machine Learning web application designed to automate the credit approval process. It helps banks and financial institutions assess the risk level of loan applicants in real-time.

By analyzing key financial factors—such as CIBIL score, recent delinquencies, and credit history—the AI model predicts the likelihood of default and categorizes applicants into risk buckets (**P1** to **P4**).

### ✨ Key Features
* **Real-Time Prediction:** Instant credit decision (Approved/Rejected) based on user input.
* **Risk Classification:** Categorizes users into 4 risk levels:
    * **P1:** Low Risk (High Approval Chance)
    * **P2:** Medium Risk
    * **P3:** High Risk
    * **P4:** Very High Risk (likely Reject)
* **Confidence Score:** Displays the probability percentage of the prediction.
* **Interactive Interface:** Built with **Streamlit** for a seamless user experience.

### 🛠️ Tech Stack
* **Frontend:** Streamlit (Python Web Framework)
* **Backend:** Python
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Processing:** Pandas, NumPy

### 📂 Project Structure

/credit-approval-app
│
├── app.py                    # Main Streamlit application
├── credit_model_simple.pkl   # Pre-trained ML model (Random Forest)
├── requirements.txt          # List of dependencies
|--Retrain.py #for upgrading our model path
README.md                 # Project documentation


