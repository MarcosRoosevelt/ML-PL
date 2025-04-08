import pandas as pd
import numpy as np
from loaded_data import loadDataFrameList
from team_Mapping import standardize_team_names

dfs = loadDataFrameList()
processed_dfs = []
k = 5

for df in dfs:
    teamsRatingFifa = pd.read_csv("OverallInformation/TeamsOverallInformationData.csv")
    currentSeason = df["season"].unique()[0]
    print(f'Temporada atual: {currentSeason}')

    teamsRatingFifa4Season = teamsRatingFifa[teamsRatingFifa["Season"] == currentSeason]

    df = standardize_team_names(df, team_columns=['HomeTeam', 'AwayTeam'])
    teamsRatingFifa4Season = standardize_team_names(teamsRatingFifa4Season, team_columns=['Name'])

    df['MHTGD'] = df.apply(lambda row: row['FTHG'] - row['FTAG'], axis=1)
    df['MATGD'] = df.apply(lambda row: row['FTAG'] - row['FTHG'], axis=1)

    # Fifa Rating
    df["OverHome"] = np.nan
    df["OverAway"] = np.nan
    df["AttOverHome"] = np.nan
    df["AttOverAway"] = np.nan
    df["MidOverHome"] = np.nan
    df["MidOverAway"] = np.nan
    df["DefOverHome"] = np.nan
    df["DefOverAway"] = np.nan

    # Goals difference
    df["GoalsDiffHome"] = np.nan
    df["GoalsDiffAway"] = np.nan

    # KPP (k-Past Performances)
    df["HomeGoalsKPP"] = np.nan
    df["AwayGoalsKPP"] = np.nan
    df["HomeCornersKPP"] = np.nan
    df["AwayCornersKPP"] = np.nan
    df["HomeShotTargetKPP"] = np.nan
    df["AwayShotTargetKPP"] = np.nan

    # Streak
    df["HomeStreak"] = np.nan
    df["AwayStreak"] = np.nan
    df["HomeWeightedStreak"] = np.nan
    df["AwayWeightedStreak"] = np.nan

    teamsSeason = sorted(df["HomeTeam"].unique(), key=str.lower)

    for i in range(len(teamsSeason)):
        team = teamsSeason[i]
        print(f'Time analisado: {team}')
        teamGames = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
        overTeam = teamsRatingFifa4Season[teamsRatingFifa4Season["Name"] == team]

        overall = overTeam["Overall"].values[0]
        attack = overTeam["Attack"].values[0]
        midfield = overTeam['Midfield'].values[0]
        defense = overTeam['Defense'].values[0]

        gameIndex = teamGames.index.tolist()

        homeGameIndex = []
        awayGameIndex = []

        # Goals difference
        goalsDiffPerGame = []
        goalsDiffCumulative = []

        # KPP (k-Past Performances)
        goals = []
        corners = []
        shotsTarget = []
        goalsKPP = [np.nan] * k
        cornersKPP = [np.nan] * k
        shotsTargetKPP = [np.nan] * k

        # Streak
        matchPointsList = []
        streaks = [np.nan] * k
        weightedStreaks = [np.nan] * k

        for i, r in teamGames.iterrows():
            if team == r["HomeTeam"]:
                homeGameIndex.append(i)
                goalsDiffPerGame.append(r["MHTGD"])
                goals.append(r["FTHG"])
                corners.append(r["HC"])
                shotsTarget.append(r["HST"])

                if r["FTR"] == "H":
                    matchPointsList.append(3)
                elif r["FTR"] == "A":
                    matchPointsList.append(0)
                elif r["FTR"] == "D":
                    matchPointsList.append(1)

            elif team == r["AwayTeam"]:
                awayGameIndex.append(i)
                goalsDiffPerGame.append(r["MATGD"])
                goals.append(r["FTAG"])
                corners.append(r["AC"])
                shotsTarget.append(r["AST"])

                if r["FTR"] == "A":
                    matchPointsList.append(3)
                elif r["FTR"] == "H":
                    matchPointsList.append(0)
                elif r["FTR"] == "D":
                    matchPointsList.append(1)

        goalsDiffCumulative = [0]
        goalsDiffAfterFirstMatch = [sum(goalsDiffPerGame[:i]) for i in range(1, len(gameIndex))]
        goalsDiffCumulative += goalsDiffAfterFirstMatch

        weights = list(range(1, k + 1))

        for i in range(k, len(gameIndex) + 1):
            goalsKPP.append(round(sum(goals[i - k:i]) / k, 2))
            cornersKPP.append(round(sum(corners[i - k:i]) / k, 2))
            shotsTargetKPP.append(round(sum(shotsTarget[i - k:i]) / k, 2))

            nfStreak = 3 * k
            nfWeightedStreak = 3 * (k * (k + 1) / 2)

            streaks.append(round(sum(matchPointsList[i - k:i]) / (nfStreak), 2))

            weightedPoints = np.array(matchPointsList[i - k:i]) * np.array(weights)
            weightedStreaks.append(round(sum(weightedPoints) / (nfWeightedStreak), 2))

        for m in range(len(gameIndex)):
            if gameIndex[m] in homeGameIndex:
                df.loc[gameIndex[m], "OverHome"] = overall
                df.loc[gameIndex[m], "AttOverHome"] = attack
                df.loc[gameIndex[m], "MidOverHome"] = midfield
                df.loc[gameIndex[m], "DefOverHome"] = defense
                df.loc[gameIndex[m], "GoalsDiffHome"] = goalsDiffCumulative[m]
                df.loc[gameIndex[m], "HomeGoalsKPP"] = goalsKPP[m]
                df.loc[gameIndex[m], "HomeCornersKPP"] = cornersKPP[m]
                df.loc[gameIndex[m], "HomeShotTargetKPP"] = shotsTargetKPP[m]
                df.loc[gameIndex[m], "HomeStreak"] = streaks[m]
                df.loc[gameIndex[m], "HomeWeightedStreak"] = weightedStreaks[m]
            elif gameIndex[m] in awayGameIndex:
                df.loc[gameIndex[m], "OverAway"] = overall
                df.loc[gameIndex[m], "AttOverAway"] = attack
                df.loc[gameIndex[m], "MidOverAway"] = midfield
                df.loc[gameIndex[m], "DefOverAway"] = defense
                df.loc[gameIndex[m], "GoalsDiffAway"] = goalsDiffCumulative[m]
                df.loc[gameIndex[m], "AwayGoalsKPP"] = goalsKPP[m]
                df.loc[gameIndex[m], "AwayCornersKPP"] = cornersKPP[m]
                df.loc[gameIndex[m], "AwayShotTargetKPP"] = shotsTargetKPP[m]
                df.loc[gameIndex[m], "AwayStreak"] = streaks[m]
                df.loc[gameIndex[m], "AwayWeightedStreak"] = weightedStreaks[m]

    # Fifa Rating
    df["DiffOverall"] = df.apply(lambda r: r["OverHome"] - r["OverAway"], axis=1)
    df["DiffAttack"] = df.apply(lambda r: r["AttOverHome"] - r["AttOverAway"], axis=1)
    df["DiffMidfield"] = df.apply(lambda r: r["MidOverHome"] - r["MidOverAway"], axis=1)
    df["DiffDefense"] = df.apply(lambda r: r["DefOverHome"] - r["DefOverAway"], axis=1)

    # Goals difference
    df["GoalsDiff"] = df.apply(lambda r: r["GoalsDiffHome"] - r["GoalsDiffAway"], axis=1)

    # KPP (k-Past Performances)
    df["GoalsKPP"] = df.apply(lambda r: r["HomeGoalsKPP"] - r["AwayGoalsKPP"], axis=1)
    df["CornersKPP"] = df.apply(lambda r: r["HomeCornersKPP"] - r["AwayCornersKPP"], axis=1)
    df["ShotsTargetKPP"] = df.apply(lambda r: r["HomeShotTargetKPP"] - r["AwayShotTargetKPP"], axis=1)

    # Streak
    df["Streak"] = df.apply(lambda r: r["HomeStreak"] - r["AwayStreak"], axis=1)
    df["WeightedStreak"] = df.apply(lambda r: r["HomeWeightedStreak"] - r["AwayWeightedStreak"], axis=1)

    processed_dfs.append(df)

all_data = pd.concat(processed_dfs, ignore_index=True)
all_data.to_csv("./all_data.csv", sep=",", index=False)
print('Dado salvo com sucesso!')