from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "healthindexscoresengland.xlsx"
REFERENCE_CSV = BASE_DIR / "health_index_combined_2021.csv"
OUTPUT_PATTERN = "health_index_combined_{year}.csv"
INDEX_SHEETS = {
    2021: "Table_3_2021_Index",
    2020: "Table_4_2020_Index",
    2019: "Table_5_2019_Index",
    2018: "Table_6_2018_Index",
    2017: "Table_7_2017_Index",
    2016: "Table_8_2016_Index",
    2015: "Table_9_2015_Index",
}
imd_sheet = "Table_10_IMD_quintile"


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def build_combined_csv(year: int, columns: list[str]) -> Path:
    sheet = INDEX_SHEETS[year]
    area_df = pd.read_excel(XLSX_PATH, sheet_name=sheet, header=4)
    area_df = area_df.rename(columns={"Area Type [Note 3]": "Area Type"})
    area_df = area_df.dropna(subset=["Area Code"])  # remove header/footer rows
    area_df["Area Code"] = area_df["Area Code"].astype(str).str.strip()
    area_df = area_df.assign(Source="Area", Year=year)
    area_df = ensure_columns(area_df, columns)

    imd_df = pd.read_excel(XLSX_PATH, sheet_name=imd_sheet, header=4)
    imd_year = imd_df[imd_df["Year"] == year].copy()
    imd_year = imd_year.rename(columns={"Health Index ": "Health Index"})
    imd_year["Area Code"] = "IMD_Q" + imd_year["Quintile"].astype(int).astype(str)
    imd_year["Area Name"] = "IMD Quintile " + imd_year["Quintile"].astype(int).astype(str)
    imd_year["Area Type"] = "IMD Quintile"
    imd_year["Source"] = "IMD Quintile"
    imd_year = imd_year.drop(columns=["Quintile"], errors="ignore")
    imd_year = ensure_columns(imd_year, columns)

    combined = pd.concat([area_df, imd_year], ignore_index=True)
    output_path = BASE_DIR / OUTPUT_PATTERN.format(year=year)
    combined.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    columns = pd.read_csv(REFERENCE_CSV, nrows=0).columns.tolist()
    for year in sorted(INDEX_SHEETS.keys(), reverse=True):
        path = build_combined_csv(year, columns)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
