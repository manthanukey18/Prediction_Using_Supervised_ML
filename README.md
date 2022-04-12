# Prediction_Using_Supervised_ML

Predication-Using-Supervised-ML
import numpy as np import pandas as pd import matplotlib.pyplot as plt import seaborn as sns from sklearn.model_selection import train_test_split from sklearn.linear_model import LinearRegression from sklearn.metrics import mean_absolute_error

data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv') data.head(10)

sns.set_style('whitegrid') sns.scatterplot(y= data['Scores'], x= data['Hours'], color='blue') plt.title('Marks Vs Study Hours',size=20) plt.ylabel('Percentage', size=10) plt.xlabel('Hours of studying', size=10) plt.show()

sns.regplot(x= data['Hours'], y= data['Scores'], color='black') plt.title('Regression Plot',size=22) plt.ylabel('Marks Percentage', size=13) plt.xlabel('Hours Studied', size=13) plt.show() print(data.corr())

Defining X and y from the Data
a = data.iloc[:, :-1].values
b = data.iloc[:, 1].values

Spliting the Data in two
train_a, val_a, train_b, val_b = train_test_split(a, b, random_state = 0)

regression = LinearRegression() regression.fit(train_a, train_b)

pred_b = regression.predict(val_a) prediction = pd.DataFrame({'Hours': [i[0] for i in val_a], 'Predicted Marks': [k for k in pred_b]}) prediction

compare_scores = pd.DataFrame({'Actual Marks': val_b, 'Predicted Marks': pred_b}) compare_scores

plt.scatter(x=val_a, y=val_b, color='blue') plt.plot(val_a, pred_b, color='red') plt.title('Actual vs Predicted', size=22) plt.ylabel('Marks Percentage', size=13) plt.xlabel('Hours Studied', size=13) plt.show()

Calculating the accuracy of model
print('Mean absolute error: ',mean_absolute_error(val_a,pred_b))

hours = [7.00] answer = regression.predict([hours]) print("Score = {}".format(round(answer[0],3)))
