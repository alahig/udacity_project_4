# Udacity capstone project Nano degree data scientist


This project contains the code written to solve a Udacity problem.

This is the Udacity Data Scientist Nanodegree Capstone project. The aim is to demonstrate the concepts learned in the programme. To do this, I have set up a project that includes

- Collecting the data (using APIs)
- Cleaning and storing the data in a database
- Writing modular, documented code 
- Analysing different learning algorithms
- Drawing conclusions and communicating them
  
Because of my background, I decided to work on an economic problem, namely forecasting inflation. Inflation measures the general increase in the price of goods and services over time. Inflation affects the purchasing power of consumers and is an important economic variable. Inflation forecasting is particularly relevant in the current context of high inflation rates following the COVID pandemic.  The current discussion is whether and how fast inflation will return to pre-COVID levels. 

The aim of this project is to analyse different inflation forecasting methods and to compare their accuracy and reliability. The project will focus on the US case. Macroeconomic variables will be retrieved from the Federal Reserve Economic Data API.


To run the notebook, you will need the following:

- Python 3.6 or higher
- Jupyter Notebook
- Numpy
- Pandas
- Matplotlib
- Seaborn

## Installation

To install the required packages, you can use pip or conda. For example, with pip you can run:

bash
./setup.sh


### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `!python udacity_inflation_forecast/udacity_inflation_forecast/data/process_data.py udacity_inflation_forecast/udacity_inflation_forecast/data`
      The ETL pipeline relies on an API key which is not part of the code. 
      You will need to store the key into the .env file by running the command which I gave on the submission comment for the project. 
      

    - The analysis that was used to create the report is here:
        
        `udacity_inflation_forecast/udacity_inflation_forecast/notebooks/analysis_report.ipynb`
        

### Code basis

The code is organized as follows:
- The data folder contains the script The ETL script. The script downloads data from the internet from various sources, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.
- The models folder contains the script to train the model. The different models tested are stored in models.py.
The model is a linear regression that uses cross validation to select the relevant variables. 


- README.md

