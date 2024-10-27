import os.path

from functions import loadData, exploreData, cleanData, displayCorrHeatMap
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import shap
import pickle
import scipy.stats as stats
import matplotlib.pyplot as plt
import textwrap

df = loadData('data/StudentPerformanceFactors.csv')

# Explore the features of our data
exploreData(df)

# Clean any missing values / duplicate records
df = cleanData(df)

df = pd.get_dummies(df)

# Display correlation analysis
correlationMatrix = df.select_dtypes(exclude=['object']).corr()
displayCorrHeatMap(correlationMatrix)

# grab numerical features
features = df.select_dtypes(exclude=['object']).drop(columns=['Exam_Score'], errors='ignore')

# Create polynomial features (interaction terms)
polyFeatures = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interactionFeatures = polyFeatures.fit_transform(features)

# Convert interaction features into a DataFrame
interactionDf = pd.DataFrame(interactionFeatures, columns=polyFeatures.get_feature_names_out(features.columns))

# Fit a linear regression model to identify interaction contributions to exam_score
X = interactionDf
y = df['Exam_Score']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define parameter grid for Gradient Boosting
# parameterGrid = {
#     'n_estimators': [100, 200, 300, 500],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'max_depth': [3, 4, 5, 6]
# }
#
# gbr = GradientBoostingRegressor(random_state=42)
#
# gridSearch = GridSearchCV(estimator=gbr, param_grid=parameterGrid, cv=3, scoring='r2', n_jobs=-1)
# gridSearch.fit(X_train, y_train)
#
# # Best parameters from the grid search
# best_params = gridSearch.best_params_
# print(f"Best Parameters: {best_params}")

# If the model isn't saved, create it and save it. If it already exists, load it in to save time.
if not os.path.exists('gbr_model.pkl'):
    # Train a new model using the best parameters
    gbr = GradientBoostingRegressor(learning_rate=.01, max_depth=3, n_estimators=500, random_state=42)
    gbr.fit(X_train, y_train)
    pickle.dump(gbr, open('gbr_model.pkl', 'wb'))

else:
    gbr = pickle.load(open('gbr_model.pkl', 'rb'))
# Load the SHAP explainer to identify which features impact predictions
explainer = shap.Explainer(gbr, X_train)

# Calculate SHAP values for the test set
shapValues = explainer(X_test)

# Summary plot to get an overview of feature importance
shap.summary_plot(shapValues, X_test)

# Evaluate the tuned model
yValues = gbr.predict(X_test)
mseRank = mean_squared_error(y_test, yValues)
r2Rank = r2_score(y_test, yValues)
print("--------------- Model scores ---------------")
print(f"Tuned MSE: {mseRank:.2f}")
print(f"Tuned R² Score: {r2Rank:.2f}")

topShapValues = pd.DataFrame(shapValues.values, columns=X_test.columns).abs().mean().sort_values(ascending=False)

print("--------------- Top Shap Values ---------------")
print(topShapValues)

# Conduct Pearson correlation for each feature against Exam_Score
results = {}
for feature in interactionDf.columns:
    correlation, p_value = stats.pearsonr(X_train[feature], y_train)
    results[feature] = {"Correlation": correlation, "P-value": p_value}


# Filter significant features based on p-value and minimum correlation threshold
significanceLevel = 0.05
correlationThreshold = 0.2

# Filter only significant features based on p-value threshold (e.g., p < 0.05)
significantResults = {feature: res for feature, res in results.items() if res["P-value"] < significanceLevel}


# Convert significant results to a DataFrame for easier manipulation
correlation_results = pd.DataFrame(significantResults).T

# Filter for features with a high absolute correlation value
filteredResults = correlation_results[correlation_results['Correlation'].abs() > correlationThreshold]

# Sort the results by correlation value
filteredResults.sort_values(by='Correlation', inplace=True, ascending=False)

# Wrap feature names for better readability
prettyLabels = [textwrap.fill(label, width=30) for label in filteredResults.index]

# Bar plot for correlation values (showing only significant correlations with high correlation)
plt.figure(figsize=(12, 10))
plt.barh(prettyLabels, filteredResults['Correlation'], color='skyblue')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Feature')
plt.title('Statistically Significant and High Correlations with Exam Score (p < 0.05, |correlation| > 0.2)')
plt.gca().invert_yaxis()  # Invert y-axis to have highest correlation on top
plt.tight_layout()
plt.show()
