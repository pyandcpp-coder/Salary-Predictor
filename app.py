import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px 

st.set_page_config(
    page_title='Salary Predictor',
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)


try:
    model = joblib.load('random_forest_tuned_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'random_forest_tuned_model.joblib' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

try:
    preprocessor = joblib.load('preprocessor.joblib')
except FileNotFoundError:
    st.error("Preprocessor file not found. Please make sure 'preprocessor.joblib' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the preprocessor: {e}")
    st.stop()

try:
    df_original_for_dropdowns = pd.read_csv('Salary_Data.csv')
except FileNotFoundError:
    st.error("Salary_Data.csv not found. This file is needed for populating input options.")
    st.stop()
except Exception as e:
    st.error(f"Error loading Salary_Data.csv: {e}")
    st.stop()


df_cleaned_for_plotting = df_original_for_dropdowns.copy()


for col in ['Gender', 'Education Level', 'Job Title', 'Salary']:
    if df_cleaned_for_plotting[col].isnull().any():
        fill_value = df_cleaned_for_plotting[col].mode()[0] if df_cleaned_for_plotting[col].dtype == 'object' else df_cleaned_for_plotting[col].median()
        df_cleaned_for_plotting[col].fillna(fill_value, inplace=True)
for col in ['Age', 'Years of Experience']:
     if df_cleaned_for_plotting[col].isnull().any():
        df_cleaned_for_plotting[col].fillna(df_cleaned_for_plotting[col].median(), inplace=True)


education_mapping_plotting = {
    "Bachelor's": "Bachelors", "Master's": "Masters", "PhD": "PhD",
    "Bachelor's Degree": "Bachelors", "Master's Degree": "Masters",
    "phD": "PhD", "High School": "HighSchool"
}
df_cleaned_for_plotting['Education Level'] = df_cleaned_for_plotting['Education Level'].map(education_mapping_plotting).fillna(df_cleaned_for_plotting['Education Level'])

N_plotting = 10 
job_title_counts_plotting = df_cleaned_for_plotting['Job Title'].value_counts()
rare_job_titles_plotting = job_title_counts_plotting[job_title_counts_plotting < N_plotting].index
df_cleaned_for_plotting['Job Title Processed'] = df_cleaned_for_plotting['Job Title'].replace(rare_job_titles_plotting, 'Other_Job_Title')


df_cleaned_for_plotting['Salary'] = pd.to_numeric(df_cleaned_for_plotting['Salary'], errors='coerce').fillna(df_cleaned_for_plotting['Salary'].median())
df_cleaned_for_plotting['Age'] = pd.to_numeric(df_cleaned_for_plotting['Age'], errors='coerce').fillna(df_cleaned_for_plotting['Age'].median())
df_cleaned_for_plotting['Years of Experience'] = pd.to_numeric(df_cleaned_for_plotting['Years of Experience'], errors='coerce').fillna(df_cleaned_for_plotting['Years of Experience'].median())



@st.cache_data
def get_dropdown_options(df_raw):
    df = df_raw.copy()
    for col in ['Gender', 'Education Level', 'Job Title']:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    for col in ['Age', 'Years of Experience']:
         if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    education_mapping = {
        "Bachelor's": "Bachelors", "Master's": "Masters", "PhD": "PhD",
        "Bachelor's Degree": "Bachelors", "Master's Degree": "Masters",
        "phD": "PhD", "High School": "HighSchool"
    }
    df['Education Level Mapped'] = df['Education Level'].map(education_mapping).fillna(df['Education Level'])
    N = 10
    job_title_counts = df['Job Title'].value_counts()
    rare_job_titles = job_title_counts[job_title_counts < N].index
    df['Job Title Processed Mapped'] = df['Job Title'].replace(rare_job_titles, 'Other_Job_Title')
    unique_genders_list = sorted(df['Gender'].unique().tolist())
    unique_education_levels_list = sorted(df['Education Level Mapped'].unique().tolist())
    unique_job_titles_processed_list = sorted(df['Job Title Processed Mapped'].unique().tolist())
    min_age_val = df['Age'].min()
    max_age_val = df['Age'].max()
    min_exp_val = df['Years of Experience'].min()
    max_exp_val = df['Years of Experience'].max()
    return unique_genders_list, unique_education_levels_list, unique_job_titles_processed_list, min_age_val, max_age_val, min_exp_val, max_exp_val

unique_genders, unique_education_levels, unique_job_titles_processed, min_age, max_age, min_exp, max_exp = get_dropdown_options(df_original_for_dropdowns)


st.title("💰 Salary Insights")


tab1, tab2 = st.tabs(["🧮 Salary Predictor", "📊 Data Explorer"])


with tab1:
    st.header("Salary Predictor")
    st.markdown("""
    Welcome! This app predicts your potential salary based on your profile.
    Please provide the following details:
    """)

    with st.form("salary_input_form"):
        st.subheader("Your Profile Details:")
        col1, col2 = st.columns(2)
        with col1:
            age_input = st.number_input("Age", min_value=int(min_age), max_value=int(max_age),
                                        value=30, step=1)
            gender_input = st.selectbox("Gender", options=unique_genders, index=unique_genders.index("Male") if "Male" in unique_genders else 0)
            experience_input = st.number_input("Years of Experience", min_value=int(min_exp), max_value=int(max_age - 15), value=5, step=1)
        with col2:
            education_input = st.selectbox("Education Level", options=unique_education_levels, index=unique_education_levels.index('Bachelors') if 'Bachelors' in unique_education_levels else 0)

        job_title_input = st.selectbox("Job Title Category",
                                       options=unique_job_titles_processed,
                                       help="Select the job category that best fits. 'Other_Job_Title' for less common roles.",
                                       index=unique_job_titles_processed.index('Software Engineer') if 'Software Engineer' in unique_job_titles_processed else 0)
        submit_button = st.form_submit_button(label="✨ Predict Salary!")

    if submit_button:
        if model is None or preprocessor is None:
            st.error("Model or preprocessor not loaded. Cannot make prediction.")
        else:
            input_data = pd.DataFrame({
                'Age': [age_input], 'Gender': [gender_input], 'Education Level': [education_input],
                'Job Title Processed': [job_title_input], 'Years of Experience': [experience_input]
            })
            st.subheader("Your Input:")
            display_input_df = input_data.copy()
            display_input_df.columns = ["Age", "Gender", "Education", "Job Category", "Experience (Yrs)"]
            st.dataframe(display_input_df, hide_index=True)
            st.divider() 

            try:
                with st.spinner("Calculating your salary prediction... ⏳"):
                    input_processed = preprocessor.transform(input_data)
                    prediction = model.predict(input_processed)
                    predicted_salary = int(prediction[0])
                    individual_tree_predictions = np.array([tree.predict(input_processed)[0] for tree in model.estimators_])
                    lower_bound_percentile = np.percentile(individual_tree_predictions, 5)
                    upper_bound_percentile = np.percentile(individual_tree_predictions, 95)
                    median_prediction_trees = np.median(individual_tree_predictions)

                st.subheader("🎉 Your Predicted Salary:")
                st.markdown(f"<h2 style='text-align: center; color: green;'>${predicted_salary:,.0f} per year</h2>", unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("More Robust Estimated Salary Range (90% Prediction Interval):")
                st.markdown(f"""
                Based on the variation among individual decision trees in the model,
                your salary is estimated to fall within:
                <h3 style='text-align: center; color: darkblue;'>${max(0, lower_bound_percentile):,.0f}  -  ${upper_bound_percentile:,.0f}</h3>
                The median prediction from these trees is: ${median_prediction_trees:,.0f}
                """, unsafe_allow_html=True)
                st.caption("This interval represents a range where 90% of the model's individual tree predictions fall, offering a more nuanced estimate than a single point.")
            except ValueError as ve:
                st.error(f"A ValueError occurred during prediction: {ve}")
                st.error("This often happens if an unexpected category is passed to the OneHotEncoder. Please check selected options.")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
                st.error("Please ensure your preprocessor and model are compatible with the input provided.")


with tab2:
    st.header("Explore Salary Data Trends")
    st.markdown("Visualize patterns and distributions from the dataset used for training the model.")
    if 'df_cleaned_for_plotting' not in locals() or df_cleaned_for_plotting.empty:
        st.warning("Cleaned data for plotting is not available or empty. Please check the data loading and cleaning steps.")
    else:
        st.subheader("Average Salary by Education Level")
        avg_salary_edu = df_cleaned_for_plotting.groupby('Education Level')['Salary'].mean().round(0).sort_values(ascending=False).reset_index()
        fig_edu = px.bar(avg_salary_edu, x='Education Level', y='Salary', text='Salary',
                         title='Average Salary by Education Level', color='Education Level',
                         labels={'Salary': 'Average Salary ($)', 'Education Level': 'Education Level'})
        fig_edu.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_edu.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_tickangle=-45)
        st.plotly_chart(fig_edu, use_container_width=True)
        st.divider()

        st.subheader("Overall Salary Distribution")
        fig_hist = px.histogram(df_cleaned_for_plotting, x='Salary', nbins=50,
                                title='Distribution of Salaries', labels={'Salary': 'Salary ($)'},
                                marginal="box")
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.divider()

        st.subheader("Average Salary by Gender")
        avg_salary_gender = df_cleaned_for_plotting.groupby('Gender')['Salary'].mean().round(0).sort_values(ascending=False).reset_index()
        fig_gender = px.bar(avg_salary_gender, x='Gender', y='Salary', text='Salary',
                            title='Average Salary by Gender', color='Gender',
                            labels={'Salary': 'Average Salary ($)', 'Gender': 'Gender'})
        fig_gender.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_gender.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_gender, use_container_width=True)
        st.divider()

        st.subheader("Salary vs. Years of Experience")
        fig_scatter_exp = px.scatter(df_cleaned_for_plotting, x='Years of Experience', y='Salary',
                                     title='Salary vs. Years of Experience',
                                     labels={'Years of Experience': 'Years of Experience', 'Salary': 'Salary ($)'},
                                     trendline="ols", color='Education Level',
                                     hover_data=['Age', 'Job Title Processed'])
        st.plotly_chart(fig_scatter_exp, use_container_width=True)
        st.divider()

        st.subheader("Salary Distribution by Education Level (Box Plot)")
        fig_box_edu = px.box(df_cleaned_for_plotting, x='Education Level', y='Salary',
                             color='Education Level', title='Salary Distribution by Education Level',
                             labels={'Salary': 'Salary ($)', 'Education Level': 'Education Level'},
                             points="outliers")
        fig_box_edu.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_box_edu, use_container_width=True)



st.sidebar.header("About This App")
st.sidebar.info(
    "This application uses a **Random Forest Regressor** model to predict salaries "
    "and allows exploration of the underlying dataset's trends."
)
st.sidebar.markdown("---")
st.sidebar.header("Model Performance")
mae_from_training = 3327.95
st.sidebar.markdown(f"""
- **R² Score:** 0.9717
- **Mean Absolute Error (MAE):** ${mae_from_training:,.2f}
""")
st.sidebar.markdown("*(Performance metrics based on the model's evaluation on a held-out test set.)*")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by: Yash Tiwari")