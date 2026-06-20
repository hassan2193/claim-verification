"""
Looks up a user's historical risk flags from dataset/user_history.csv.
"""


def get_risk_flags(user_id, history_df):
    if history_df is None or "user_id" not in history_df.columns:
        return "none"

    user_row = history_df[history_df["user_id"] == user_id]
    if len(user_row) == 0:
        return "none"

    flag = user_row.iloc[0].get("history_flags", "none")
    flag_str = str(flag).strip()
    if flag_str == "" or flag_str.lower() == "nan":
        return "none"
    return flag_str


if __name__ == "__main__":
    import pandas as pd
    from config import USER_HISTORY_CSV

    df = pd.read_csv(USER_HISTORY_CSV)
    print(get_risk_flags("user_005", df))
