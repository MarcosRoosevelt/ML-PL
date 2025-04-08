import pandas as pd


def loadDataFrameList():
    df0506_practice = pd.read_csv("football_data_seasons/PremierLeague_0506.csv")
    df0607_practice = pd.read_csv("football_data_seasons/PremierLeague_0607.csv")
    df0708_practice = pd.read_csv("football_data_seasons/PremierLeague_0708.csv")
    df0809_practice = pd.read_csv("football_data_seasons/PremierLeague_0809.csv")
    df0910_practice = pd.read_csv("football_data_seasons/PremierLeague_0910.csv")
    df1011_practice = pd.read_csv("football_data_seasons/PremierLeague_1011.csv")
    df1112_practice = pd.read_csv("football_data_seasons/PremierLeague_1112.csv")
    df1213_practice = pd.read_csv("football_data_seasons/PremierLeague_1213.csv")
    df1314_practice = pd.read_csv("football_data_seasons/PremierLeague_1314.csv")
    df1415_test = pd.read_csv("football_data_seasons/PremierLeague_1415.csv")
    df1516_test = pd.read_csv("football_data_seasons/PremierLeague_1516.csv")

    dataFramesList = [df0506_practice, df0607_practice, df0708_practice, df0809_practice, df0910_practice,
                      df1011_practice
        , df1112_practice, df1213_practice, df1314_practice, df1415_test, df1516_test]

    for i in range(len(dataFramesList)):
        dataFramesList[i] = dataFramesList[i].dropna(subset=["HomeTeam"])

    return dataFramesList
