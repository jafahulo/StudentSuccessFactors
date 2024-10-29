**WGU 2024 1920**

# WGU Data Analytics Capstone Project
## Introduction
Throughout my time at WGU I have learned several skills to become successful. In my personal experience, 
I've seen that pacing myself, allowing time for fun, and setting goals has been hugely influential to my success.

In this capstone project I will analyze a dataset of student performance factors and exam scores found on Kaggle.com 
(Courtesy of [lainguyn123](https://www.kaggle.com/lainguyn123)).

The purpose of this project is to display skills I have learned while earning my bachelors degree. I will apply statistical
analysis to understand correlations between student behavior and their exam scores.

## Hypothesis Statement:
What Is the top contributor to high and low exam scores? Are they the same?

## Dataset information
### Description
This dataset provides a comprehensive overview of various factors affecting student performance in exams. It includes information on study habits, attendance, parental involvement, and other aspects influencing academic success.

### Column Descriptions

| Attribute                  | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| Hours_Studied               | Number of hours spent studying per week.                                    |
| Attendance                  | Percentage of classes attended.                                             |
| Parental_Involvement        | Level of parental involvement in the student's education (Low, Medium, High).|
| Access_to_Resources         | Availability of educational resources (Low, Medium, High).                  |
| Extracurricular_Activities   | Participation in extracurricular activities (Yes, No).                      |
| Sleep_Hours                 | Average number of hours of sleep per night.                                 |
| Previous_Scores             | Scores from previous exams.                                                 |
| Motivation_Level            | Student's level of motivation (Low, Medium, High).                          |
| Internet_Access             | Availability of internet access (Yes, No).                                  |
| Tutoring_Sessions           | Number of tutoring sessions attended per month.                             |
| Family_Income               | Family income level (Low, Medium, High).                                    |
| Teacher_Quality             | Quality of the teachers (Low, Medium, High).                                |
| School_Type                 | Type of school attended (Public, Private).                                  |
| Peer_Influence              | Influence of peers on academic performance (Positive, Neutral, Negative).    |
| Physical_Activity           | Average number of hours of physical activity per week.                      |
| Learning_Disabilities       | Presence of learning disabilities (Yes, No).                                |
| Parental_Education_Level     | Highest education level of parents (High School, College, Postgraduate).    |
| Distance_from_Home          | Distance from home to school (Near, Moderate, Far).                         |
| Gender                      | Gender of the student (Male, Female).                                       |
| Exam_Score                  | Final exam score.                                                          |
> Source: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors


## Analysis Steps
### Exploration

### Cleaning

### Identifying top features


Model Fitting:
```
# Define parameter grid for Gradient Boosting
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6]
}

gbr = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from the grid search
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")
```


## Contributions
The dataset for this project was provided under the CCO 1.0 Universal Public Domain license by lainguyn123 on Kaggle.com
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

