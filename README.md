# Streamlit Regression Dashboard

Regression-Dashboard uses the Streamlit library to create a dashboard to visualize linear and polynomial regression analysis using the Scikit-Learn library.  To begin, upload a dataset
and set your parameters.  Under the "Data Summary" tab, an initial exploratory data analysis is displayed.  The "Visualization" tab displays a 
two-dimensional and three-dimensional scatter plot of the dependent vs independent variable(s).  Lastly, the "Build Model" tab performs the regression 
selected using the established parameters set in the sidebar.  Note the training set is preset at 60% and the remaining 40% is used as the test set.
These values can be changed and the test set will be the remaining percentage after the training and validation set size is set.  To run the application 
run "streamlit run regression.py".  
