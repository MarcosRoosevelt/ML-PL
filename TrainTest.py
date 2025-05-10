import pandas as pd
from sklearn.model_selection import train_test_split


def trainTestSplit(test_size=0.2, random_state=42):
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

    df = df.dropna(subset=nanFeatures)

    Y = df["FTR"]
    X = df.drop(columns=["FTR", "season"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y
    )

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

    