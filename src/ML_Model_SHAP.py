import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


df_stress_Level = pd.read_csv('src/StressLevelDataset.csv')

head = df_stress_Level.head() #Head of DataFrame
print(head)

df_stress_Level.info() #check if balacing dataset needs to be done


null_values = df_stress_Level.isnull().sum() #check for null values
print(null_values)

X = df_stress_Level[['self_esteem', 'mental_health_history',
                     'blood_pressure', 'sleep_quality', 'living_conditions',
                     'social_support','anxiety_level', 'safety']]
y = df_stress_Level['stress_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train) #model fitting 

y_pred = model.predict(X_test) #model prediction
print(y_pred)

from sklearn.metrics import accuracy_score, r2_score

r2 = r2_score(y_test, y_pred) #R2 score
print("R2 Score:", r2) #0.82, strong, correlation, good model fit

#SHAP plots and values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar") #Bar plot (importance of features)

shap.summary_plot(shap_values, X_train) #regular plot (Variability of feature impact visualized)



#Dependence plot: how one or two features affect the model output
shap.dependence_plot("anxiety_level", shap_values, X_train)

'''
Explanation:
The code above reflects a dataset containing various features that
 may contribute to stress levels in students.
 
 The model is a Random Forest Regressor since most variables are numerical and continuous.

 The R2 score of 0.82 indicates a strong correlation between all of the features and the stress level.

 The SHAP plots indicate that anxiety level, self-esteem, and mental health history
 are the most important features in determining stress level.

 The SHAP summary plot shows more specifically how each feature impacts the 
 model's predictions based on each data point.

 SHAP values do a great job of explaining how each factor affects the target variable in a 
 dataset. It is a type of explainable/interpretable MAchine Learning Framework/process that
 is widley used in the industry for production ML models, Explainable AI, and Data Science.

 This is an intro to some of the plots in SHAP for describing feature importance and impact.

 Author: Bhargav Ashok, Virginia Tech '27, Computational Modeling and Data Analytics

 Links/references/tools used will be provided in the README file on github. 

 '''