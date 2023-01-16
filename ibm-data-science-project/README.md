# SpaceX Falcon 9 Landing Prediction using Machine Learning

The main goal of this project is to predict whether the Falcon 9 first stage will land successfully using machine learning techniques. SpaceX prides itself in being able to reuse the first stage of a rocket launch, which is a significant cost savings. If we can determine if the first stage will land, we can determine the cost of a launch. This information can be used if an alternate company wants to bid against SpaceX for a rocket launch.

## Project Description:
The methodolgy followed will include:

- Data Collection
- Data Wrangling and Preprocessing
- Exploratory Data Analysis
- Data Visualization
- Machine Learning Prediction

## Table Of Contents:
The project includes:

- Main Project Jupyter Notebook file: ibm_data_science_project.ipynb
- Plotly Dash app: plotly_dash_spacex_app.py
- Folium Map: folium_map_jupyter.ipynb
- Powerpoint Presentation: ibm_data_science_presentation.pptx
- PDF (same as PPT): ibm_data_science_presentation.pdf

1. Space-X Data Collection API
The API used here is public.
The API provides data about types of rocket launches done by SpaceX
The data is cleaned and exported to CSV for analysis

2. Data Collection with Web Scraping
We perform web scraping to collect Falcon 9 historical launch records from a Wikipedia page.
The data is cleaned and exported to CSV for analysis
Snippet of the wikipedia page:

3. Exploratory Data Analysis
We perform some Exploratory Data Analysis (EDA) to find some patterns in the data and determine what would be the label for training supervised models.
In the data set, there are several different cases where the booster did not land successfully. Sometimes a landing was attempted but failed due to an accident; for example, True Ocean means the mission outcome was successfully landed to a specific region of the ocean while False Ocean means the mission outcome was unsuccessfully landed to a specific region of the ocean. True RTLS means the mission outcome was successfully landed to a ground pad False RTLS means the mission outcome was unsuccessfully landed to a ground pad. 
True ASDS means the mission outcome was successfully landed on a drone ship False ASDS means the mission outcome was unsuccessfully landed on a drone ship.
Successful landing example:
Unsuccessful landing example:

4. Exploratory Data Analysis with SQL
We use SQL to query the database and answer several questions about the data such as:
-The names of the unique launch sites in the space mission
-The total payload mass carried by boosters launched by NASA (CRS)
-The average payload mass carried by booster version F9 v1.1
-Some of the SQL statements or functions used include SELECT, DISTINCT, AS, FROM, WHERE, LIMIT, LIKE, SUM(), AVG(), MIN(), BETWEEN, COUNT(), and YEAR().

5. Exploratory Data Analysis with Data Visualization and Plotly Dash
We use Python's Matplotlib and Seaborn libraries to visualize the relationships that exist within the dataset.
The "One-Hot Encoding" technique is used to create binary category variables as part of the Feature Engineering
Visualizing the success rate in each orbit:
-Class 1 = Success
-Class 0 = Failure

6. Interactive Map Analytics with Folium
-In this notebook we perform the following:
-Mark all launch sites on a map
-Mark the success/failed launches for each site on the map
-Calculate the distances between a launch site and the coastlines or other important landmarks

## Conclusion
During our investigation, the results of our analysis indicate that there are some features of rocket launches  have a correlation with the success or failure launches. We conclude that the KNN algorithm was the best choice for this problem.

### Dependencies
pandas
numpy
matplotlib
seaborn
plotly
sklearn
sqlite3
folium
