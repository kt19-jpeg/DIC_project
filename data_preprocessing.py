import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLACEHOLDER_VALUES = {
    "",
    "unknown",
    "Unknown",
    "NA",
    "N/A",
    "null",
    "None",
}


def load_data(input_csv: Path) -> pd.DataFrame:
    return pd.read_csv(input_csv, low_memory=False)


def clean_ipl_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove technical index columns often created during export.
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace(list(PLACEHOLDER_VALUES), pd.NA)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "innings",
        "over",
        "ball",
        "ball_no",
        "runs_batter",
        "balls_faced",
        "valid_ball",
        "runs_extras",
        "runs_total",
        "runs_bowler",
        "runs_target",
        "team_runs",
        "team_balls",
        "team_wicket",
        "batter_runs",
        "batter_balls",
        "bowler_wicket",
        "day",
        "month",
        "year",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_like_cols = ["runs_not_boundary", "umpires_call", "striker_out"]
    for col in bool_like_cols:
        if col in df.columns:
            if df[col].dtype == "bool":
                continue
            df[col] = (
                df[col]
                .astype("string")
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )

    df = df.drop_duplicates()

    return df


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else pd.NA


def build_eda_report(df: pd.DataFrame) -> str:
    n_rows, n_cols = df.shape
    n_matches = int(df["match_id"].nunique()) if "match_id" in df.columns else 0

    if {"batting_team", "bowling_team"}.issubset(df.columns):
        n_teams = int(
            pd.unique(pd.concat([df["batting_team"], df["bowling_team"]]).dropna()).size
        )
    else:
        n_teams = 0

    date_min = df["date"].min() if "date" in df.columns else pd.NaT
    date_max = df["date"].max() if "date" in df.columns else pd.NaT

    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    innings = (
        df.groupby(["match_id", "innings", "batting_team"], as_index=False)
        .agg(
            total_runs=("runs_total", "sum"),
            legal_balls=("valid_ball", "sum"),
            wickets=("team_wicket", "max"),
            venue=("venue", first_non_null),
        )
    )
    innings["run_rate"] = np.where(
        innings["legal_balls"] > 0, innings["total_runs"] * 6 / innings["legal_balls"], np.nan
    )

    first_innings_avg = innings.loc[innings["innings"] == 1, "total_runs"].mean()
    second_innings_avg = innings.loc[innings["innings"] == 2, "total_runs"].mean()

    match_meta = (
        df.groupby("match_id", as_index=False)
        .agg(
            match_won_by=("match_won_by", first_non_null),
            toss_winner=("toss_winner", first_non_null),
            toss_decision=("toss_decision", first_non_null),
            year=("year", first_non_null),
        )
    )

    inn1 = innings[innings["innings"] == 1][["match_id", "batting_team"]].rename(
        columns={"batting_team": "team_1st"}
    )
    inn2 = innings[innings["innings"] == 2][["match_id", "batting_team"]].rename(
        columns={"batting_team": "team_2nd"}
    )
    match_view = match_meta.merge(inn1, on="match_id", how="left").merge(
        inn2, on="match_id", how="left"
    )
    match_view["winner_is_1st"] = match_view["match_won_by"] == match_view["team_1st"]
    match_view["winner_is_2nd"] = match_view["match_won_by"] == match_view["team_2nd"]
    decisive = match_view[match_view["winner_is_1st"] | match_view["winner_is_2nd"]]
    defend_pct = decisive["winner_is_1st"].mean() * 100 if len(decisive) else np.nan
    chase_pct = decisive["winner_is_2nd"].mean() * 100 if len(decisive) else np.nan
    toss_win_pct = (
        (match_view["toss_winner"] == match_view["match_won_by"]).mean() * 100
        if len(match_view)
        else np.nan
    )

    top_batters = (
        df.groupby("batter", as_index=False)
        .agg(runs=("runs_batter", "sum"), balls=("balls_faced", "sum"))
        .query("balls > 0")
    )
    top_batters["strike_rate"] = top_batters["runs"] * 100 / top_batters["balls"]
    top_batters = top_batters.sort_values("runs", ascending=False).head(10)

    top_bowlers = (
        df.groupby("bowler", as_index=False)
        .agg(
            wickets=("bowler_wicket", "sum"),
            legal_balls=("valid_ball", "sum"),
            runs_conceded=("runs_bowler", "sum"),
        )
        .query("legal_balls > 0")
    )
    top_bowlers["economy"] = top_bowlers["runs_conceded"] * 6 / top_bowlers["legal_balls"]
    top_bowlers = top_bowlers.sort_values("wickets", ascending=False).head(10)

    venue_scoring = (
        innings[innings["innings"] == 1]
        .groupby("venue", as_index=False)
        .agg(matches=("match_id", "nunique"), avg_first_innings=("total_runs", "mean"))
        .query("matches >= 10")
        .sort_values("avg_first_innings", ascending=False)
        .head(10)
    )

    yearly_scoring = (
        innings.merge(match_meta[["match_id", "year"]], on="match_id", how="left")
        .groupby(["year", "innings"], as_index=False)
        .agg(avg_runs=("total_runs", "mean"))
        .pivot(index="year", columns="innings", values="avg_runs")
        .rename(columns={1: "avg_1st", 2: "avg_2nd"})
        .sort_index()
    )

    lines = []
    lines.append("# IPL Data Cleaning + EDA Report")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append(f"- Rows: **{n_rows:,}**")
    lines.append(f"- Columns: **{n_cols}**")
    lines.append(f"- Unique matches: **{n_matches:,}**")
    lines.append(f"- Unique teams: **{n_teams}**")
    if pd.notna(date_min) and pd.notna(date_max):
        lines.append(f"- Date range: **{date_min.date()}** to **{date_max.date()}**")
    lines.append("")

    lines.append("## Missing Values (Top 12)")
    if len(missing) == 0:
        lines.append("- No missing values found.")
    else:
        for col, pct in missing.head(12).items():
            lines.append(f"- `{col}`: {pct:.2f}%")
    lines.append("")

    lines.append("## Match Dynamics")
    lines.append(f"- Average 1st innings score: **{first_innings_avg:.2f}**")
    lines.append(f"- Average 2nd innings score: **{second_innings_avg:.2f}**")
    lines.append(f"- Toss winner also won: **{toss_win_pct:.2f}%**")
    lines.append(f"- Defend wins (decisive games): **{defend_pct:.2f}%**")
    lines.append(f"- Chase wins (decisive games): **{chase_pct:.2f}%**")
    lines.append("")

    lines.append("## Top Batters By Runs")
    for _, row in top_batters.iterrows():
        lines.append(
            f"- {row['batter']}: {int(row['runs'])} runs, SR {row['strike_rate']:.1f} ({int(row['balls'])} balls)"
        )
    lines.append("")

    lines.append("## Top Bowlers By Wickets")
    for _, row in top_bowlers.iterrows():
        lines.append(
            f"- {row['bowler']}: {int(row['wickets'])} wickets, economy {row['economy']:.2f}"
        )
    lines.append("")

    lines.append("## Highest Scoring Venues (Avg 1st Inns, min 10 matches)")
    for _, row in venue_scoring.iterrows():
        lines.append(
            f"- {row['venue']}: {row['avg_first_innings']:.1f} ({int(row['matches'])} matches)"
        )
    lines.append("")

    lines.append("## Yearly Scoring Trend (Last 10 Years)")
    for year, row in yearly_scoring.tail(10).iterrows():
        avg_1st = row.get("avg_1st", np.nan)
        avg_2nd = row.get("avg_2nd", np.nan)
        lines.append(f"- {int(year)}: 1st inns {avg_1st:.1f}, 2nd inns {avg_2nd:.1f}")
    lines.append("")

    lines.append("## Insights")
    lines.append("- Most null values are context-driven (reviews, dismissals, super-over fields).")
    lines.append("- Recent seasons are materially higher-scoring than earlier IPL seasons.")
    lines.append("- Chasing has a slight edge in decisive results in this dataset.")
    lines.append("- Toss matters, but effect size is moderate rather than dominant.")

    return "\n".join(lines)


def save_eda_tables(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_table = (df.isna().mean() * 100).sort_values(ascending=False).reset_index()
    missing_table.columns = ["column", "missing_pct"]
    missing_table.to_csv(output_dir / "eda_missingness.csv", index=False)

    innings_table = (
        df.groupby(["match_id", "innings", "batting_team"], as_index=False)
        .agg(total_runs=("runs_total", "sum"), legal_balls=("valid_ball", "sum"))
    )
    innings_table["run_rate"] = np.where(
        innings_table["legal_balls"] > 0,
        innings_table["total_runs"] * 6 / innings_table["legal_balls"],
        np.nan,
    )
    innings_table.to_csv(output_dir / "eda_innings_summary.csv", index=False)


def save_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Missingness (top 15 columns)
    missing = (df.isna().mean() * 100).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(missing.index[::-1], missing.values[::-1], color="#1f77b4")
    plt.xlabel("Missing %")
    plt.title("Top 15 Missing Columns")
    plt.tight_layout()
    plt.savefig(output_dir / "missingness_top15.png", dpi=150)
    plt.close()

    # 2) Year-wise average innings score
    innings = (
        df.groupby(["match_id", "innings", "batting_team"], as_index=False)
        .agg(total_runs=("runs_total", "sum"))
    )
    match_year = df.groupby("match_id", as_index=False).agg(year=("year", first_non_null))
    yearly = (
        innings.merge(match_year, on="match_id", how="left")
        .groupby(["year", "innings"], as_index=False)
        .agg(avg_runs=("total_runs", "mean"))
        .pivot(index="year", columns="innings", values="avg_runs")
        .rename(columns={1: "avg_1st", 2: "avg_2nd"})
        .sort_index()
    )
    plt.figure(figsize=(10, 6))
    if "avg_1st" in yearly.columns:
        plt.plot(yearly.index, yearly["avg_1st"], marker="o", label="1st Innings")
    if "avg_2nd" in yearly.columns:
        plt.plot(yearly.index, yearly["avg_2nd"], marker="o", label="2nd Innings")
    plt.xlabel("Year")
    plt.ylabel("Average Runs")
    plt.title("Yearly Average Runs by Innings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "yearly_avg_runs_by_innings.png", dpi=150)
    plt.close()

    # 3) Top 10 batters by runs
    bat = (
        df.groupby("batter", as_index=False)
        .agg(runs=("runs_batter", "sum"))
        .sort_values("runs", ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10, 6))
    plt.barh(bat["batter"][::-1], bat["runs"][::-1], color="#2ca02c")
    plt.xlabel("Runs")
    plt.title("Top 10 Batters by Runs")
    plt.tight_layout()
    plt.savefig(output_dir / "top10_batters_runs.png", dpi=150)
    plt.close()

    # 4) Top 10 bowlers by wickets
    bowl = (
        df.groupby("bowler", as_index=False)
        .agg(wickets=("bowler_wicket", "sum"))
        .sort_values("wickets", ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10, 6))
    plt.barh(bowl["bowler"][::-1], bowl["wickets"][::-1], color="#d62728")
    plt.xlabel("Wickets")
    plt.title("Top 10 Bowlers by Wickets")
    plt.tight_layout()
    plt.savefig(output_dir / "top10_bowlers_wickets.png", dpi=150)
    plt.close()

    # 5) Toss decision distribution
    toss_decision = (
        df.groupby("match_id", as_index=False)
        .agg(toss_decision=("toss_decision", first_non_null))
        ["toss_decision"]
        .value_counts(dropna=False)
    )
    plt.figure(figsize=(7, 5))
    plt.bar(toss_decision.index.astype(str), toss_decision.values, color="#9467bd")
    plt.xlabel("Toss Decision")
    plt.ylabel("Matches")
    plt.title("Toss Decision Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "toss_decision_distribution.png", dpi=150)
    plt.close()

    # 6) Highest scoring venues (avg first innings, min 10 matches)
    innings_with_venue = (
        df.groupby(["match_id", "innings", "venue"], as_index=False)
        .agg(total_runs=("runs_total", "sum"))
    )
    venue_scores = (
        innings_with_venue[innings_with_venue["innings"] == 1]
        .groupby("venue", as_index=False)
        .agg(matches=("match_id", "nunique"), avg_first_innings=("total_runs", "mean"))
    )
    venue_scores = venue_scores[venue_scores["matches"] >= 10].sort_values(
        "avg_first_innings", ascending=False
    ).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(
        venue_scores["venue"][::-1],
        venue_scores["avg_first_innings"][::-1],
        color="#ff7f0e",
    )
    plt.xlabel("Average First Innings Score")
    plt.title("Top 10 Highest Scoring Venues (1st Innings)")
    plt.tight_layout()
    plt.savefig(output_dir / "top10_venues_avg_first_innings.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run IPL cleaning and EDA, and export cleaned data + report."
    )
    parser.add_argument("--input", default="IPL.csv", help="Input CSV path")
    parser.add_argument(
        "--clean-output", default="IPL_cleaned.csv", help="Cleaned CSV output path"
    )
    parser.add_argument(
        "--report-output", default="IPL_eda_report.md", help="EDA markdown report output path"
    )
    parser.add_argument(
        "--eda-dir",
        default="eda_outputs",
        help="Directory to store EDA tables (CSV summaries)",
    )
    parser.add_argument(
        "--plots-dir",
        default="eda_plots",
        help="Directory to store EDA plot images",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    clean_path = Path(args.clean_output)
    report_path = Path(args.report_output)
    eda_dir = Path(args.eda_dir)
    plots_dir = Path(args.plots_dir)

    df = load_data(input_path)
    clean_df = clean_ipl_data(df)

    clean_df.to_csv(clean_path, index=False)
    report_text = build_eda_report(clean_df)
    report_path.write_text(report_text, encoding="utf-8")
    save_eda_tables(clean_df, eda_dir)
    save_eda_plots(clean_df, plots_dir)

    print(f"Cleaned file written: {clean_path}")
    print(f"EDA report written: {report_path}")
    print(f"EDA tables directory: {eda_dir}")
    print(f"EDA plots directory: {plots_dir}")


if __name__ == "__main__":
    main()
