from functions import loadData, exploreAndCleanData, correlationHeatMap


df = loadData('data/StudentPerformanceFactors.csv')

# explore the features of our data and clean any missing values / duplicate records
# save cleaned dataset
df = exploreAndCleanData(df)


correlationHeatMap(df)

