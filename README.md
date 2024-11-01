
# CMPS 3500 - Deep Learning and Neural Networks Project

## Project Overview

This project is part of CMPS 3500 - Introduction to Deep Learning and Neural Networks at CSUB. The goal is to develop an intelligent system that can classify individuals into different credit score categories using a neural network model. The project requires reading, processing, and analyzing credit data, followed by training, testing, and tuning a neural network model to achieve optimal predictions.

### Repository Contents
- `ClassProjectGroup3.py`: Main Python file containing code to load data, process data, train and test the neural network model.
- `credit_score_data.csv`: Data file containing anonymized customer data with 30 features and 80,000 records. Used for training and testing the model.
- `predictionsProject1.csv`: Output file (to be generated) containing predictions with two columns: ID and Credit_Score.

---

## Project Requirements

### Data Processing
1. **Data Loading**: Read data from `credit_score_data.csv`, handling possible missing or invalid values.
2. **Data Cleaning**: Preprocess data to handle missing values, normalize or standardize features, and transform any necessary columns.

### Model Implementation
1. **Model Selection**: Construct a neural network model suitable for credit score classification.
2. **Model Training**: Train the neural network on the processed dataset.
3. **Model Tuning**: Optimize the model by testing different architectures and hyperparameters to minimize Root Mean Square Error (RMSE).

### Model Grading
- **Evaluation**: Evaluate the model using a reserved testing set and compare performance metrics (RMSE).
- **Interface**: Implement a user-friendly interface that allows multiple testing rounds and easy navigation through options.

---

## Run Description

The program consists of a menu with the following options:
1. **Load Data**: Load and preprocess the credit score data.
2. **Clean Data**: Clean and process the data further, if necessary.
3. **Train Model**: Train the neural network model and print model details, including hyperparameters and RMSE.
4. **Test Model**: Test the model and generate predictions.

Example output format:
```
Menu:
 1) Load Data
 2) Clean Data
 3) Train Model
 4) Test Model
 5) Exit
Select Option:
```

## Data Set Description

The dataset `credit_score_data.csv` includes the following columns:

| Column Name              | Description                                              |
|--------------------------|----------------------------------------------------------|
| `Customer_ID`            | Unique identifier for a customer                         |
| `Month`                  | Month when data was gathered                             |
| `Name`                   | Full name of the customer                                |
| `Age`                    | Age of the customer                                      |
| `Annual_Income`          | Customer's annual income                                 |
| ...                      | ...                                                      |
| `Credit_Score`           | The assigned credit score                                |

(Refer to the project specification for the full list of features.)

---

## Error Handling and User Interface

- **Error Handling**: Implemented to detect corrupt files, incorrect column counts, and malformed data. Program should print user-friendly error messages and avoid crashes.
- **User Interface**: Menu-driven interface guides the user through loading data, cleaning, training, and testing the model.

---

## Resources

- [How to Use Pylint - “Good Code”](https://pylint.pycqa.org/en/latest/)
- [Python Data Wrangling Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html)



