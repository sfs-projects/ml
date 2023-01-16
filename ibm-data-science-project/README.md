# SpaceX Falcon 9 Landing Prediction using Machine Learning

The main goal of this project is to predict whether the Falcon 9 first stage will land successfully using machine learning techniques. 

## Project Description:
The project includes the following steps:

-Data Collection: Data was collected from the SpaceX API and a Wikipedia page using web scraping.
-Data Wrangling and Preprocessing: Data was cleaned and exported to CSV for analysis.
-Exploratory Data Analysis: SQL and visualization libraries such as Folium, Seaborn, and Plotly Dash were used to analyze the data.
-Machine Learning Prediction: Several models (Logistic Regression, SVM, Decision Tree, and KNN) were trained and evaluated using grid search.

## Table Of Contents:
The project includes:

-Main Project Jupyter Notebook file: ibm_data_science_project.ipynb
-Plotly Dash app: anexes/plotly_dash_spacex_app.py
-Folium Map: anexes/folium_map_jupyter.ipynb
-Powerpoint Presentation: presentation/ibm_data_science_presentation.pptx

## Conclusion
During our investigation, the results of our analysis indicate that there are some features of rocket launches  have a correlation with the success or failure launches. We conclude that the KNN algorithm was the best choice for this problem.

## Dependencies
This project requires the following libraries:

pandas
numpy
matplotlib
seaborn
plotly
dash
sklearn
folium

## Limitations
The dataset used in this project was relatively small with only 93 observations and 83 features. This may limit the generalizability of our findings and increase the risk of overfitting the model. To mitigate the effects of this small dataset, we used Cross-Validation technique.

## Conclusion
The results of our analysis indicate that there are some features of rocket launches that have a correlation with the success or failure of launches. The KNN algorithm was found to be the best choice for this problem. However, more data would be beneficial in increasing the robustness and reliability of the results.
