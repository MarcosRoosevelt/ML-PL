import pandas as pd


def trainTestSplit():
    df = pd.read_csv("all_data.csv")

    bettingStats = ["B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "GBH", "GBD", "GBA", "IWH", "IWD", "IWA", "LBH",
                    "LBD", "LBA", "SBH", "SBD", "SBA", "WHH", "WHD", "WHA", "SJH", "SJD", "SJA", "VCH", "VCD", "VCA",
                    "Bb1X2", "BbMxH", "BbAvH", "BbMxD", "BbAvD", "BbMxA", "BbAvA", "BbOU", "BbMx>2.5", "BbAv>2.5",
                    "BbMx<2.5", "BbAv<2.5", "BbAH", "BbAHh", "BbMxAHH", "BbAvAHH", "BbMxAHA", "BbAvAHA", "BSH", "BSD",
                    "BSA", "PSA", "PSH", "PSD", "PSCA", "PSCD", "PSCH"]
    # Qualitativity Variables
    genDropInfo = ["Div", "Date", "HomeTeam", "AwayTeam", "Referee"]
    # Initial variables
    nanFeatures = ['GoalsKPP', 'HomeGoalsKPP', 'AwayGoalsKPP', 'CornersKPP', 'HomeCornersKPP', 'AwayCornersKPP',
                   'ShotsTargetKPP', 'HomeShotTargetKPP', 'AwayShotTargetKPP', 'Streak', 'HomeStreak', 'AwayStreak',
                   'WeightedStreak', 'HomeWeightedStreak', 'AwayWeightedStreak']
    overfittingData = ["FTHG", "FTAG", "HTHG", "HTAG", "HTR", "MHTGD", "MATGD"]

    df.drop(bettingStats + genDropInfo + overfittingData, axis=1, inplace=True)

    seasons = sorted(df['season'].unique())
    dfs = []

    for season in seasons:
        tempDF = df[(df['season'] == (season))]
        tempDF = tempDF.dropna(subset=nanFeatures)
        dfs.append(tempDF)

    df = pd.concat(dfs)

    Y = df[["FTR"]]
    X = df.drop(columns=["FTR"])

    train_seasons = [f'{year}-{year + 1}' for year in range(2005, 2014)]
    test_seasons = [f'{year}-{year + 1}' for year in range(2014, 2016)]

    train_mask = X['season'].isin(train_seasons)
    test_mask = X['season'].isin(test_seasons)

    XTrain = X[train_mask].copy()
    XTest = X[test_mask].copy()
    YTrain = Y[train_mask].copy()
    YTest = Y[test_mask].copy()

    XTrain = XTrain.drop(columns=['season'])
    XTest = XTest.drop(columns=['season'])

    return XTrain, XTest, YTrain.values.ravel(), YTest.values.ravel()