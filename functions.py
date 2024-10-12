import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Grab our data
def loadData(filePath):
    df = pd.read_csv(filePath)
    return df

def exploreAndCleanData(df):

    nullValueCount = df.isnull().sum()
    nullValueCount = nullValueCount[nullValueCount > 0]
    duplicateCount = df.duplicated().sum()

    df.dropna(inplace=True) # clear null values
    df.drop_duplicates(inplace=True) # clear duplicated values

    print("Data Info:\n------------------------------")
    print(f"Cleaned Null Values: \n{nullValueCount}\n")
    print(f"Cleaned Duplicates: {duplicateCount}")

    print(f"\nDataframe shape:\n{df.shape}")
    print(f"\nDataframe Info: {df.describe()}")
    print(f"\nFeature counts:")
    print("--------------------------------------------")
    for col in df.select_dtypes(exclude=['number']):
        print(f"{df[col].value_counts()}\n")

    return df

def correlationHeatMap(df):
    # correlation analysis
    correlationMatrix = df.select_dtypes(exclude=['object']).corr()

    plt.figure(figsize=(10, 10))
    colors = mcolors.LinearSegmentedColormap.from_list('my_colormap', ["#ff0000", "#FFFFFF", "#ff0000"])
    sns.heatmap(correlationMatrix, annot=True, cmap=colors, linewidths=.5, vmin=-1, vmax=1)
    plt.title("Correlation HeatMap")
    plt.show()

    return correlationMatrix