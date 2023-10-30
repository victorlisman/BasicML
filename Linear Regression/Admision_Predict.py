# Admission_Predict.csv: https://www.kaggle.com/mohansacharya/graduate-admissions
# Linear Regression model with an R2 score of 0.72 and MSE of 0.004


import pandas as pd
import numpy as np
from LR import LR
from sklearn.model_selection import train_test_split

data = pd.read_csv('./Datasets/Adm_pred/Admission_Predict.csv', header=1)

# Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.2)

# Save the training data to a csv file
train.to_csv('./Datasets/Adm_pred/train.csv', index=False)

# Save the testing data to a csv file
test.to_csv('./Datasets/Adm_pred/test.csv', index=False)

# Read the training and testing data
X_train = train.iloc[:, 1:8].values
y_train = train.iloc[:, 8].values

X_test = test.iloc[:, 1:8].values
y_test = test.iloc[:, 8].values

# Make sure the data is np.float64
X_test = np.array(X_test, dtype=np.float64)
y_test = np.array(y_test, dtype=np.float64)

X_train = np.array(X_train, dtype=np.float64)
y_train = np.array(y_train, dtype=np.float64)

    

def main():
    # Train the model
    model = LR(lr=0.00001, epochs=100000)
    model.fit(X_train, y_train)

    # Evaluate the model and print the MSE and R2 score
    mse, r2 = model.evaluate(X_test, y_test)
    print(f'MSE: {mse}, R2: {r2}')

if __name__ == '__main__':
    main()