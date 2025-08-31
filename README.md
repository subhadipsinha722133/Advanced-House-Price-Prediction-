# 🏠 Advanced House Price Prediction  

## 📌 Problem Statement  
The objective of this project is to predict the **selling price of a house** based on multiple features (parameters) such as size, location, quality, and other housing attributes.  

---
# 🔗Live Demo
🏠 [Demo Link](https://bvfcv9csevzqbhhq483vua.streamlit.app/)

Video:-

<img width="880" height="450" src="https://github.com/subhadipsinha722133/Advanced-House-Price-Prediction-/blob/main/demo.gif" alt="Python project demo">


---
## 🚀 Approach  

1. **Data Preprocessing**  
   - Handled missing values by creating a separate **"Missing" category** instead of dropping them.  
   - Used **Target Guided Label Encoding** for categorical variables to convert them into numerical format.  

2. **Feature Engineering & Selection**  
   - Polynomial features created for better model learning.  
   - Applied **SelectFromModel** technique and **Lasso Regression** to identify the most important features.  

3. **Model Building**  
   - Implemented **Polynomial Linear Regression** to predict house prices.  

4. **Model Evaluation**  
   - Achieved an **R² score of 91.2%**, meaning the model explains ~91% of the variance in the dataset.  

---

## 📊 Model Performance  
- **R² Score:** 0.912 (91.2%)  
- Indicates strong predictive power and reliable performance.  

---

## 🛠️ Tech Stack  

- **Python**  
- **Streamlit** – For deployment & interactive web app  
- **Scikit-learn** – For machine learning models  
- **Pandas, NumPy** – Data manipulation  
- **Matplotlib, Seaborn** – Visualization  
- **Pickle** – Model serialization
   
Advanced-House-Price-Prediction/  <br>
│-- data/                  # Dataset (if included)  <br>
│-- notebooks/             # Jupyter notebooks for EDA & model building  <br>
│-- app.py                 # Streamlit web app   <br>
│-- model.pkl              # Trained model   <br>
│-- requirements.txt       # Dependencies   <br>
│-- README.md              # Project documentation  <br>

---


## ✨ Features in Web App

- 📊 Data visualization (correlations, missing value impact, etc.)

- 🏠 Predict house price based on user input

- 🔥 Simple and interactive UI

# 👨‍💻 Author

- Subhadip Sinha <br>
💡 AIML Engineer 

--- 

## 🌐 Deployment  
- The model is deployed **locally using Streamlit**.  
- Run the following command to launch the app:  

```bash
streamlit run app.py



