def get_team_mapping():
    team_mapping = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Manchester Utd": "Manchester United",
        "Charlton": "Charlton Athletic",
        "Blackburn": "Blackburn Rovers",
        "Blackburn Rvrs": "Blackburn Rovers",
        "Arsenal": "Arsenal FC",
        "West Brom": "West Bromwich",
        "Newcastle": "Newcastle United",
        "Newcastle Utd": "Newcastle United",
        "Wigan": "Wigan Athletic",
        "West Ham": "West Ham United",
        "Birmingham": "Birmingham City",
        "Bolton": "Bolton Wanderers",
        "Tottenham": "Tottenham Hotspur",
        "Spurs": "Tottenham Hotspur",
        "Reading": "Reading FC",
        "Derby": "Derby County",
        "Stoke": "Stoke City",
        "Hull": "Hull City",
        "Wolves": "Wolverhampton Wanderers",
        "Wolverhampton": "Wolverhampton Wanderers",
        "Swansea": "Swansea City",
        "Norwich": "Norwich City",
        "Cardiff": "Cardiff City",
        "Leicester": "Leicester City",
        "Chelsea": "Chelsea FC",
        "Fulham": "Fulham",
        "Sunderland": "Sunderland",
        "Aston Villa": "Aston Villa",
        "Everton": "Everton",
        "Liverpool": "Liverpool",
        "Portsmouth": "Portsmouth",
        "Middlesbrough": "Middlesbrough",
        "Sheffield United": "Sheffield United",
        "Watford": "Watford",
        "Burnley": "Burnley",
        "Blackpool": "Blackpool",
        "QPR": "QPR",
        "Southampton": "Southampton",
        "Crystal Palace": "Crystal Palace",
        "Bournemouth": "Bournemouth",
        "Huddersfield": "Huddersfield",
        "Brighton": "Brighton"
    }

    return team_mapping


def standardize_team_names(df, team_columns=None):
    standardized_df = df.copy()

    if team_columns is None:
        team_columns = ['HomeTeam', 'AwayTeam']

    mapping = get_team_mapping()

    for column in team_columns:
        if column in standardized_df.columns:
            standardized_df[column] = standardized_df[column].replace(mapping)

    return standardized_df