# IPL Data Cleaning + EDA Report

## Dataset Summary
- Rows: **278,205**
- Columns: **63**
- Unique matches: **1,169**
- Unique teams: **19**
- Date range: **2008-04-18** to **2025-06-03**

## Missing Values (Top 12)
- `match_number`: 100.00%
- `team_reviewed`: 99.69%
- `review_decision`: 99.69%
- `umpire`: 99.69%
- `review_batter`: 99.69%
- `method`: 98.60%
- `superover_winner`: 98.60%
- `result_type`: 98.31%
- `fielders`: 96.40%
- `next_batter`: 95.21%
- `new_batter`: 95.21%
- `wicket_kind`: 95.03%

## Match Dynamics
- Average 1st innings score: **167.02**
- Average 2nd innings score: **153.54**
- Toss winner also won: **51.57%**
- Defend wins (decisive games): **45.81%**
- Chase wins (decisive games): **54.19%**

## Top Batters By Runs
- V Kohli: 8671 runs, SR 132.9 (6523 balls)
- RG Sharma: 7048 runs, SR 132.1 (5337 balls)
- S Dhawan: 6769 runs, SR 127.1 (5326 balls)
- DA Warner: 6567 runs, SR 139.7 (4702 balls)
- SK Raina: 5536 runs, SR 136.8 (4046 balls)
- MS Dhoni: 5439 runs, SR 137.5 (3957 balls)
- KL Rahul: 5235 runs, SR 136.0 (3848 balls)
- AB de Villiers: 5181 runs, SR 151.9 (3411 balls)
- AM Rahane: 5032 runs, SR 125.0 (4025 balls)
- CH Gayle: 4997 runs, SR 149.3 (3346 balls)

## Top Bowlers By Wickets
- YS Chahal: 221 wickets, economy 7.96
- B Kumar: 198 wickets, economy 7.69
- SP Narine: 192 wickets, economy 6.80
- PP Chawla: 192 wickets, economy 7.96
- R Ashwin: 187 wickets, economy 7.20
- JJ Bumrah: 186 wickets, economy 7.25
- DJ Bravo: 183 wickets, economy 8.38
- A Mishra: 174 wickets, economy 7.38
- RA Jadeja: 170 wickets, economy 7.67
- SL Malinga: 170 wickets, economy 7.14

## Highest Scoring Venues (Avg 1st Inns, min 10 matches)
- Arun Jaitley Stadium, Delhi: 200.3 (23 matches)
- Eden Gardens, Kolkata: 196.8 (23 matches)
- Rajiv Gandhi International Stadium, Uppal, Hyderabad: 189.4 (19 matches)
- M Chinnaswamy Stadium, Bengaluru: 189.1 (19 matches)
- Sawai Mansingh Stadium, Jaipur: 187.2 (17 matches)
- Narendra Modi Stadium, Ahmedabad: 186.6 (33 matches)
- Brabourne Stadium: 180.4 (10 matches)
- Brabourne Stadium, Mumbai: 177.4 (17 matches)
- Wankhede Stadium, Mumbai: 176.8 (52 matches)
- Punjab Cricket Association IS Bindra Stadium: 175.6 (10 matches)

## Yearly Scoring Trend (Last 10 Years)
- 2016: 1st inns 162.6, 2nd inns 151.8
- 2017: 1st inns 165.8, 2nd inns 152.3
- 2018: 1st inns 172.5, 2nd inns 159.2
- 2019: 1st inns 166.7, 2nd inns 156.6
- 2020: 1st inns 169.5, 2nd inns 153.0
- 2021: 1st inns 159.3, 2nd inns 151.1
- 2022: 1st inns 171.1, 2nd inns 158.5
- 2023: 1st inns 182.7, 2nd inns 166.7
- 2024: 1st inns 189.6, 2nd inns 176.2
- 2025: 1st inns 188.8, 2nd inns 174.0

## Insights
- Most null values are context-driven (reviews, dismissals, super-over fields).
- Recent seasons are materially higher-scoring than earlier IPL seasons.
- Chasing has a slight edge in decisive results in this dataset.
- Toss matters, but effect size is moderate rather than dominant.