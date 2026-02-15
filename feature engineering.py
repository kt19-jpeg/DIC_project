import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TEAM_NAME_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
}


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    team_cols = ["batting_team", "bowling_team", "toss_winner", "match_won_by", "superover_winner"]
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


def add_ball_level_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sort_cols = ["match_id", "innings", "over", "ball"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    group_cols = ["match_id", "innings", "batting_team"]

    df["is_boundary"] = df["runs_batter"].isin([4, 6]).astype("int8")
    df["is_six"] = (df["runs_batter"] == 6).astype("int8")
    df["is_four"] = (df["runs_batter"] == 4).astype("int8")
    df["is_dot_ball"] = ((df["runs_total"] == 0) & (df["valid_ball"] == 1)).astype("int8")
    df["is_wicket_ball"] = (
        (df["bowler_wicket"].fillna(0) > 0) | (df["wicket_kind"].notna())
    ).astype("int8")

    # Over phases based on legal over number in T20.
    df["over_phase"] = pd.cut(
        df["over"],
        bins=[-1, 5, 14, 19],
        labels=["powerplay", "middle", "death"],
    ).astype("string")

    df["cum_runs"] = df.groupby(group_cols)["runs_total"].cumsum()
    df["cum_legal_balls"] = df.groupby(group_cols)["valid_ball"].cumsum()
    df["cum_boundaries"] = df.groupby(group_cols)["is_boundary"].cumsum()
    df["cum_wickets"] = df.groupby(group_cols)["is_wicket_ball"].cumsum()

    df["current_run_rate"] = np.where(
        df["cum_legal_balls"] > 0, df["cum_runs"] * 6 / df["cum_legal_balls"], np.nan
    )
    df["balls_remaining"] = 120 - df["cum_legal_balls"]
    df["wickets_in_hand"] = 10 - df["cum_wickets"]
    df["boundary_pct_so_far"] = np.where(
        df["cum_legal_balls"] > 0,
        df["cum_boundaries"] * 100 / df["cum_legal_balls"],
        np.nan,
    )

    # Chase context features for 2nd innings only.
    target = (
        df.groupby(group_cols, as_index=False)["runs_target"]
        .max()
        .rename(columns={"runs_target": "innings_target"})
    )
    df = df.merge(target, on=group_cols, how="left")
    df["runs_required"] = np.where(df["innings"] == 2, df["innings_target"] - df["cum_runs"], np.nan)
    df["required_run_rate"] = np.where(
        (df["innings"] == 2) & (df["balls_remaining"] > 0),
        df["runs_required"] * 6 / df["balls_remaining"],
        np.nan,
    )
    df["pressure_index"] = df["required_run_rate"] - df["current_run_rate"]

    batter_group = ["match_id", "innings", "batter"]
    df["batter_cum_runs"] = df.groupby(batter_group)["runs_batter"].cumsum()
    df["batter_cum_balls"] = df.groupby(batter_group)["balls_faced"].cumsum()
    df["batter_sr_live"] = np.where(
        df["batter_cum_balls"] > 0,
        df["batter_cum_runs"] * 100 / df["batter_cum_balls"],
        np.nan,
    )

    bowler_group = ["match_id", "innings", "bowler"]
    df["bowler_cum_runs_conceded"] = df.groupby(bowler_group)["runs_bowler"].cumsum()
    df["bowler_cum_balls"] = df.groupby(bowler_group)["valid_ball"].cumsum()
    df["bowler_economy_live"] = np.where(
        df["bowler_cum_balls"] > 0,
        df["bowler_cum_runs_conceded"] * 6 / df["bowler_cum_balls"],
        np.nan,
    )

    df["runs_last_12_balls"] = (
        df.groupby(group_cols)["runs_total"]
        .rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=group_cols, drop=True)
    )
    df["wkts_last_12_balls"] = (
        df.groupby(group_cols)["is_wicket_ball"]
        .rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=group_cols, drop=True)
    )

    return df


def build_match_level_features(ball_df: pd.DataFrame) -> pd.DataFrame:
    match_key = ["match_id", "innings", "batting_team", "bowling_team"]
    inning_agg = (
        ball_df.groupby(match_key, as_index=False)
        .agg(
            total_runs=("runs_total", "sum"),
            legal_balls=("valid_ball", "sum"),
            wickets=("is_wicket_ball", "sum"),
            boundaries=("is_boundary", "sum"),
            dot_balls=("is_dot_ball", "sum"),
            sixes=("is_six", "sum"),
            fours=("is_four", "sum"),
            venue=("venue", "first"),
            date=("date", "first"),
            toss_winner=("toss_winner", "first"),
            toss_decision=("toss_decision", "first"),
            match_won_by=("match_won_by", "first"),
        )
    )

    inning_agg["run_rate"] = np.where(
        inning_agg["legal_balls"] > 0, inning_agg["total_runs"] * 6 / inning_agg["legal_balls"], np.nan
    )
    inning_agg["dot_ball_pct"] = np.where(
        inning_agg["legal_balls"] > 0, inning_agg["dot_balls"] * 100 / inning_agg["legal_balls"], np.nan
    )
    inning_agg["boundary_pct"] = np.where(
        inning_agg["legal_balls"] > 0, inning_agg["boundaries"] * 100 / inning_agg["legal_balls"], np.nan
    )
    inning_agg["balls_used"] = inning_agg["legal_balls"]
    inning_agg["balls_unused"] = 120 - inning_agg["balls_used"]

    inning_agg["won_match"] = (inning_agg["batting_team"] == inning_agg["match_won_by"]).astype("int8")
    inning_agg["toss_winner_batted"] = (
        (inning_agg["toss_winner"] == inning_agg["batting_team"])
        & (inning_agg["toss_decision"] == "bat")
    ).astype("int8")
    inning_agg["toss_winner_fielded"] = (
        (inning_agg["toss_winner"] == inning_agg["bowling_team"])
        & (inning_agg["toss_decision"] == "field")
    ).astype("int8")

    # 2nd innings target pressure snapshot.
    targets = (
        ball_df.groupby(["match_id", "innings", "batting_team"], as_index=False)["innings_target"]
        .max()
        .rename(columns={"innings_target": "target"})
    )
    inning_agg = inning_agg.merge(
        targets, on=["match_id", "innings", "batting_team"], how="left"
    )
    inning_agg["required_to_win"] = np.where(
        inning_agg["innings"] == 2, inning_agg["target"], np.nan
    )
    inning_agg["chase_success"] = np.where(
        inning_agg["innings"] == 2, inning_agg["won_match"], np.nan
    )

    return inning_agg


def main():
    parser = argparse.ArgumentParser(
        description="Create ball-level and match-level features for IPL dataset."
    )
    parser.add_argument(
        "--input", default="IPL_cleaned.csv", help="Input cleaned CSV path from preprocessing step"
    )
    parser.add_argument(
        "--ball-output",
        default="IPL_ball_features.csv",
        help="Ball-level feature CSV output path",
    )
    parser.add_argument(
        "--match-output",
        default="IPL_match_features.csv",
        help="Match/innings-level feature CSV output path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    ball_output = Path(args.ball_output)
    match_output = Path(args.match_output)

    df = pd.read_csv(input_path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = normalize_team_names(df)
    ball_features = add_ball_level_features(df)
    match_features = build_match_level_features(ball_features)

    ball_features.to_csv(ball_output, index=False)
    match_features.to_csv(match_output, index=False)

    print(f"Ball-level features written: {ball_output}")
    print(f"Match-level features written: {match_output}")


if __name__ == "__main__":
    main()
