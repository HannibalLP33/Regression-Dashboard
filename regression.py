###IMPORT DEPENDENCIES###
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from itertools import combinations
import os
import math
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

###PAGE CONFIGURATIONS
st.set_page_config(page_title="Regression Dashboard")
plt.style.use('dark_background')

###FUNCTIONS###
PROJECT_ROOT_DIR = "."
FOLDER = "figures"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, FOLDER)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_2D_data(features,response):
    plt.rcParams.update({"font.size": 15})
    feature_len = len(features.columns)
    if feature_len == 1:
        fig, ax = plt.subplots(figsize=(14,14))
        for feature in features.columns:
            ax.set(xlabel = str(feature), ylabel = response.name)
            ax.scatter(response, features[feature], c = response, cmap = "jet")            
    else:
        if feature_len%2 == 0:
            start_location = int(((feature_len*100)/2)+21)
            row = int(str(start_location)[0])
            column = int(str(start_location)[1])
            index = [x+1 for x in range(feature_len)]
        else:
            start_location = int((((feature_len+1)*100)/2)+21)
            row = int(str(start_location)[0])
            column = int(str(start_location)[1])
            index = [x+1 for x in range(feature_len)]
        fig = plt.figure(figsize=(14,14))
        for i, feature in enumerate(features.columns):
            ax = plt.subplot(row, column, index[i])
            ax.set(xlabel = str(feature), ylabel = response.name)
            ax.scatter(response, features[feature], c = response, cmap = "jet")
    plt.tight_layout()
    return fig
    
def plot_3D_data(features, response):
    plt.rcParams.update({"font.size":15})
    axis_comb = list(combinations(features.columns.tolist(),2))
    comb_len = len(axis_comb)
    if comb_len == 1:
        fig = plt.figure(figsize = [14,14])
        ax = fig.add_subplot(projection = "3d")
        for x, y in axis_comb:
            ax.set(xlabel = str(x), ylabel = str(y), zlabel = response.name)
            ax.scatter3D(features[x], features[y], response, c = response, cmap = "jet")
    else:
        fig = plt.figure(figsize = [25,25])
        if comb_len%2 == 0:
            start_location = int(((comb_len*100)/2)+21)
            row = int(str(start_location)[0])
            column = int(str(start_location)[1])
            index = [x+1 for x in range(comb_len)]
        else: 
            start_location = int((((comb_len+1)*100)/2)+21)
            row = int(str(start_location)[0])
            column = int(str(start_location)[1])
            index = [x+1 for x in range(comb_len)]
       
        for i, (x, y) in enumerate(axis_comb):
            ax = fig.add_subplot(row, column, index[i],projection = "3d")
            ax.set(xlabel = str(x), ylabel = str(y), zlabel = response.name)
            ax.scatter3D(features[x], features[y], response, c = response, cmap = "jet")
    plt.tight_layout()
    return fig

def data_summary(df, x_train, x_val, x_test):
    summary.subheader("Summary")
    
    num_rows = len(df)
    num_col = len(df.columns)
    num_na = df.isna().sum()
    df_dtypes = df.dtypes
    cormatrix = df.corr().round(3)
    description = df.describe()
    
    dataset_col1, dataset_col2 = summary.columns(2)
   
    dataset_col1.write(f"Full Dataset Size: {len(df)}")
    dataset_col2.write(f"Training Set Size: {len(x_train)}")
    dataset_col2.write(f"Validation Set Size: {len(x_val)}")
    dataset_col2.write(f"Test Set Size: {len(x_test)}") 
    dataset_col1.write(f"Number of Rows: {num_rows}")
    dataset_col1.write(f"Number of Columns: {num_col}")
    dataset_col1.write("Number of Missing Values:")
    dataset_col1.dataframe(num_na)
    dataset_col2.write("Datatypes:")
    dataset_col2.dataframe(df_dtypes.astype(str))
    summary.write("Correlation Matrix:")
    summary.dataframe(cormatrix)
    summary.write("Measures of Central Tendency")
    summary.dataframe(description)    

def continous_evaluations(true_values, predicted):
    col1, col2, col3, col4 = build.columns(4)
    col1.write(f"MAE: {mean_absolute_error(true_values, predicted).round(4)}")
    col2.write(f"MSE: {mean_squared_error(true_values, predicted).round(4)}")
    col3.write(f"RMSE {round(math.sqrt(mean_squared_error(true_values, predicted)),4)}")
    col4.write(f"MAPE: {mean_absolute_percentage_error(true_values, predicted).round(4)}")

def linear_reg(model_type, features, response, alpha, l1, rand_state, rounding):
    if model_type == "Ridge":
        model = Ridge(alpha = alpha, random_state = rand_state)
        model.fit(features, response)
    
    elif model_type == "Lasso":
        model = Lasso(alpha = alpha,random_state = rand_state)
        model.fit(features, response)
   
    elif model_type == "Elastic Net":
        model = ElasticNet(alpha = alpha, l1_ratio = l1,random_state = rand_state)
        model.fit(features, response)

    else:
        model = LinearRegression()
        model.fit(features, response)
    formula_coef = np.c_[model.feature_names_in_.reshape(-1,1), model.coef_.reshape(-1,1)]
    formula = ""
    for i, (label, coef) in enumerate(formula_coef):
        if i == 0:
            string = f"{round(coef,rounding)}{label}"
            formula += string
        else:
            string = f"{np.where((np.sign(round(coef,rounding)) >= 0), '+', '')}{round(coef,rounding)}{label}"
            formula += string
    formula += f"{np.where((np.sign(model.intercept_) >= 0), '+','')} {model.intercept_.round(rounding)}"
    
    return model, formula

###SIDE BAR###
try:
    filename = st.sidebar.file_uploader("BEGIN BY UPLOADING A DATA FILE")
    df = pd.read_csv(filename)
    st.sidebar.title("Model")
    regression = st.sidebar.selectbox("Pick a Regression Model:", ["Linear Regression", "Polynomial Regression"], 0)

    st.sidebar.title("Model Parameters") 
    y_variable = st.sidebar.selectbox("Dependent Variable:", df.columns.tolist())
    remaining_x_columns = [x_column for x_column in df.columns.tolist() if x_column!= y_variable]
    x_variable = st.sidebar.multiselect("Independent Variable(s):", remaining_x_columns, remaining_x_columns)
    rounding = st.sidebar.slider(f"Number of decimal places in equation",0,5, value = 2)
    train_size = st.sidebar.slider("Set Training Set Size (%):", 0, 100, value = 60)


    X = df.loc[:,x_variable]
    y = df.loc[:,y_variable]
    if train_size == 100:
        x_train = X
        y_train = y
        x_val = []
        y_val = []
        x_test = []
        y_test = []
        val_size = 0
        test_size = 0
    else:
        val_size = st.sidebar.slider("(Optional) Set Validation Set Size (%):", 0, (100-train_size), value = 0)
        test_size = 100-train_size-val_size
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1-(train_size/100), random_state = 42)
        
    if val_size != 0:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = val_size/100, random_state=42)
    else:
        x_val = []

    
    summary, visuals, build = st.tabs(["Data Summary", "Visualization", "Build Model"])

    ###MODELS###
    if regression == "Linear Regression":
        ###Side Bar###
        model_type = st.sidebar.radio("Select a model type",["Normal", "Ridge", "Lasso", "Elastic Net"])
        if model_type == "Ridge" or model_type =="Lasso":
            alpha_value = st.sidebar.slider("Alpha Term:", 1.0, 10.0, value = 1.0)
            l1_term = None
        elif model_type == "Elastic Net":
            alpha_value = st.sidebar.slider("Alpha Term:", 1.0, 10.0, value = 1.0)
            l1_term = st.sidebar.slider("L1 Term:", 0.0, 1.0, value = 0.5)
        else:
            alpha_value = None
            l1_term = None
        rand_state = st.sidebar.slider("Random State:", 0, 100, value = 7)
         
        
        ###TAB 1###
        summary.title("Linear Regression Model")
        data_summary(df, x_train, x_val, x_test)
        
        ###TAB 2###
        visuals.subheader("Full Dataset Visualization")
        if len(X.columns)>0:
            visuals.subheader("2D Scatter Plot(s)")
            fig1 = plot_2D_data(X,y)
            visuals.pyplot(fig1)
            if len(X.columns)>1:
                visuals.subheader("3D Scatter Plot(s)")
                fig2 = plot_3D_data(X,y)
                visuals.pyplot(fig2)
            else:
                visuals.write("Select two or more independent variables to generate a 3D plot")
        else:
            visuals.write("Select at least one indpendent variable to generate a 2D plot")
        
        ###Tab3###
        model, formula = linear_reg(model_type = model_type, features = x_train, response = y_train, alpha = alpha_value, l1 = l1_term, rand_state= rand_state, rounding = rounding)
        build.subheader("Current Linear Model Formula")
        build.latex(f"Formula = {formula}")
        
        
        training_predictions = model.predict(x_train)
        build.subheader("Model Evaluation with Training Data")
        continous_evaluations(y_train, training_predictions)
        
        
        if (val_size != 0):
            val_predictions = model.predict(x_val)
            build.subheader("Model Evaluation with Validation Data")
            continous_evaluations(y_val, val_predictions)
            
        if (test_size != 0):
            if build.button("Use Current Model on Test Set"):
                test_predictions = model.predict(x_test)
                build.subheader("Model Evaluation with Test Data")
                continous_evaluations(y_test, test_predictions)
                    
            
    if regression == "Polynomial Regression":
        summary.title("Polynomial Regression Model")
        summary.image("polynomialdegrees.jpeg", use_column_width=True)
        ###TAB 1###
        data_summary(df, x_train, x_val, x_test)
        
        ###TAB 2###
        visuals.subheader("Full Dataset Visualization")
        if len(X.columns)>0:
            visuals.subheader("2D Scatter Plot(s)")
            fig1 = plot_2D_data(X,y)
            visuals.pyplot(fig1)
            if len(X.columns)>1:
                visuals.subheader("3D Scatter Plot(s)")
                fig2 = plot_3D_data(X,y)
                visuals.pyplot(fig2)
            else:
                visuals.write("Select two or more independent variables to generate a 3D plot")
        else:
            visuals.write("Select at least one indpendent variable to generate a 2D plot")
        ### SIDE BAR ###
        bias_estimate = np.where(st.sidebar.radio("Include Estimate Bias:", ["False", "True"]) == "False", False, True)
        degree = st.sidebar.slider("Choose a Degree:", 1,50)
        
        ### MODEL PARAMETERS ###
        poly_feat = PolynomialFeatures(degree = degree, include_bias = bias_estimate)
        poly_train = poly_feat.fit_transform(x_train)
        model = LinearRegression()
        model.fit(poly_train, y_train)
        
        ### GENERATE FORMULA ###
        formula_coef = np.c_[poly_feat.get_feature_names_out().reshape(-1,1), model.coef_.reshape(-1,1)]
        formula = ""
        for i, (label, coef) in enumerate(formula_coef):
            if i == 0 and label == "1":
                string = f"{round(coef,rounding)}"
                formula += string
            elif i == 0:
                string = f"{round(coef,rounding)}{label}"
                formula += string
            else:
                string = f"{np.where(coef>=0, '+','')}{round(coef,rounding)}{label}"
                formula += string
        formula += f"{np.where(model.intercept_ >= 0, '+', '')}{model.intercept_.round(rounding)}"
        build.subheader("Current Polynomial Regression Model Formula")
        build.latex(f"Formula = {formula}")
        train_predicted = model.predict(poly_train)
        build.subheader("Model Evaluation with Training Data")
        continous_evaluations(y_train, train_predicted)
        if (val_size != 0):
            poly_val = poly_feat.transform(x_val)
            val_predicted = model.predict(poly_val)
            build.subheader("Model Evaluation with Validation Data")
            continous_evaluations(y_val, val_predicted)
        if (test_size != 0):
            if build.button("Use Current Model on Test Set"):
                poly_test = poly_feat.transform(x_test)
                test_predicted = model.predict(poly_test)
                build.subheader("Model Evaluation with Test Data")
                continous_evaluations(y_test, test_predicted)
except Exception as e:
    pass