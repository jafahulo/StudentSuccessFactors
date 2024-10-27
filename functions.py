import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Grab our data
def loadData(filePath):
    """
    This function loads the data from the csv file into a pandas dataframe.
    :param filePath: File path to csv
    :return: dataFrame containing the csv data
    """
    df = pd.read_csv(filePath)
    return df

def exploreData(df):
    """
    This function explores data in the provided dataFrame.
    :param df: Dataframe to print exploratory data about
    :return: None
    """
    nullValueCount = df.isnull().sum()
    nullValueCount = nullValueCount[nullValueCount > 0]
    duplicateCount = df.duplicated().sum()

    print(f"Duplicates: {duplicateCount}")
    print(f"Null Values: \n{nullValueCount}\n")
    print(f"\nDataframe shape:\n{df.shape}")
    print(f"\nDataframe Info: {df.describe()}")
    print(f"\nFeature counts:")
    print("--------------------------------------------")
    for col in df.select_dtypes(exclude=['number']):
        print(f"{df[col].value_counts()}\n")

    return None

def cleanData(df, silent=True):
    """
    This function cleans the data in the provided dataframe.
    :param df: dataFrame to be cleaned
    :param silent: execute quietly
    :return: cleaned dataFrame
    """
    nullValueCount = df.isnull().sum()
    nullValueCount = nullValueCount[nullValueCount > 0]
    duplicateCount = df.duplicated().sum()

    df.dropna(inplace=True) # clear null values
    df.drop_duplicates(inplace=True) # clear duplicated values
    if not silent:
        print("Data Info:\n------------------------------")
        print(f"Cleaned Null Values: \n{nullValueCount}\n")
        print(f"Cleaned Duplicates: {duplicateCount}")

    return df

def displayCorrHeatMap(df):
    """
    This function displays the correlation heatmap of the data in the provided dataframe.
    :param df: Pandas dataframe correlation matrix
    :return: None
    """

    plt.figure(figsize=(50, 50))
    colors = mcolors.LinearSegmentedColormap.from_list('my_colormap', ["#ff0000", "#FFFFFF", "#ff0000"])
    sns.heatmap(df, annot=True, cmap=colors, linewidths=.5, vmin=-1, vmax=1)
    plt.title("Correlation HeatMap")
    plt.show()

    return None
