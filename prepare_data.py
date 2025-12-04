import glob
import os
import sys

import pandas as pd


def _find_excel_file(preferred: str = "data/Online_Retail.xlsx") -> str:
    """
    Try to find the UCI Excel file in the data/ folder.
    - First, look for the preferred name.
    - If not found, fall back to the first .xlsx in data/.
    """
    if os.path.exists(preferred):
        return preferred

    candidates = glob.glob("data/*.xlsx")
    if candidates:
        # Take the first .xlsx we find
        print(f"Preferred file '{preferred}' not found. Using '{candidates[0]}' instead.")
        return candidates[0]

    raise FileNotFoundError(
        "No Excel file found in 'data/'. "
        "Download 'Online Retail.xlsx' from the UCI repository and place it in the 'data' folder."
    )


def prepare_uci_online_retail(
    input_path: str | None = None,
    output_path: str = "data/online_retail_daily.csv",
) -> None:
    """
    Convert the UCI Online Retail Excel dataset into a daily sales CSV
    that the Streamlit dashboard can use.
    """
    excel_path = input_path or _find_excel_file()

    print(f"Loading raw data from: {excel_path}")
    df = pd.read_excel(excel_path)

    df = df.dropna(subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Group by date (normalize to date only, no time)
    df["date"] = df["InvoiceDate"].dt.date
    daily = (
        df.groupby("date", as_index=False)["Sales"]
        .sum()
        .rename(columns={"Sales": "sales"})
    )

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily[["date", "sales"]]
    daily["category"] = "All Products"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving daily sales to: {output_path}")
    daily.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    # If a path is passed as first argument, use it directly.
    if len(sys.argv) > 1:
        excel_arg = sys.argv[1]
        prepare_uci_online_retail(input_path=excel_arg)
    else:
        prepare_uci_online_retail()

