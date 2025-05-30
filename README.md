# 💰 Salary Predictor & Data Explorer

This project is a Streamlit web application that predicts salaries based on user-provided features and allows exploration of the underlying salary dataset.

## 🚀 Live Demo

https://salary-predictor-rf.streamlit.app

## ✨ Features

*   **Salary Prediction:**
    *   Input features: Age, Gender, Education Level, Job Title Category, and Years of Experience.
    *   Uses a pre-trained Random Forest Regressor model.
    *   Provides a point salary prediction.
    *   Displays a 90% confidence interval for the prediction based on individual tree estimates from the Random Forest.
*   **Data Explorer Tab:**
    *   Interactive visualizations of the salary dataset using Plotly Express.
    *   Charts include:
        *   Average Salary by Education Level (Bar Chart)
        *   Overall Salary Distribution (Histogram & Box Plot)
        *   Average Salary by Gender (Bar Chart)
        *   Salary vs. Years of Experience (Scatter Plot with Trendline)
        *   Salary Distribution by Education Level (Box Plot)

## 🛠️ Technologies Used

*   **Python:** Core programming language.
*   **Streamlit:** For building the interactive web application.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Scikit-learn:** For the machine learning model (Random Forest) and preprocessing.
*   **Joblib:** For saving and loading the trained model and preprocessor.
*   **Plotly Express:** For creating interactive charts in the Data Explorer.


## 📈 Model Performance

The Random Forest Regressor model achieved the following performance on a held-out test set:

*   **R² Score:** 0.9717
*   **Mean Absolute Error (MAE):** $3,327.95

## ⚙️ Setup and Local Execution

1.  **Clone the repository (optional, if you're sharing this):**
    ```bash
    git clone https://github.com/pyandcpp-coder/Salary-Predictor.git
    cd Salary-Predictor
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

## 📊 Data Cleaning and Preprocessing

The `Salary_Data.csv` dataset underwent the following preprocessing steps before model training:

1.  **Handling Missing Values:**
    *   Categorical features (`Gender`, `Education Level`, `Job Title`): Imputed with the mode.
    *   Numerical features (`Age`, `Years of Experience`, `Salary`): Imputed with the median.
2.  **Standardizing Categorical Values:**
    *   `Education Level`: Mapped various representations (e.g., "Bachelor's", "Bachelor's Degree") to standardized forms (e.g., "Bachelors").
    *   `Job Title`: Grouped infrequent job titles (appearing less than 10 times) into an 'Other_Job_Title' category to reduce dimensionality.
3.  **Feature Engineering:**
    *   The processed `Job Title Processed` and mapped `Education Level` were used as features.
4.  **Encoding and Scaling (Handled by `preprocessor.joblib`):**
    *   Categorical features (`Gender`, `Education Level`, `Job Title Processed`): One-Hot Encoded.
    *   Numerical features (`Age`, `Years of Experience`): Standard Scaled.

## 🔮 Future Enhancements (Ideas)

*   Incorporate more features if available (e.g., Industry, Location, Company Size).
*   Experiment with other regression models or deep learning.
*   Add more interactive "what-if" scenarios for predictions.
*   Allow users to upload their own (anonymized) data for analysis.

## 🙏 Acknowledgements

*   The dataset used is `Salary_Data.csv` (Kaggle).
*   Thanks to the Streamlit, Pandas, Scikit-learn, and Plotly communities for their excellent libraries.

## 👨‍💻 Developer

*   **Yash Tiwari**
