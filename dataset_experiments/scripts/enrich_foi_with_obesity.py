from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parents[1]
FOI_INPUT = DATA_DIR / "foi-02477.csv"
INDICATOR_INPUT = DATA_DIR / "indicator-93881-all-areas.data.csv"
OUTPUT_PATH = DATA_DIR / "foi-02477-with-obesity.csv"

# Mapping from FOI region labels to the indicator area names we should aggregate.
REGION_TARGETS: Dict[str, Tuple[str, ...]] = {
    "LONDON": ("LONDON REGION (STATISTICAL)",),
    "SOUTH WEST": ("SOUTH WEST REGION (STATISTICAL)",),
    "SOUTH EAST": ("SOUTH EAST REGION (STATISTICAL)",),
    "EAST OF ENGLAND": ("EAST OF ENGLAND REGION (STATISTICAL)",),
    "NORTH WEST": ("NORTH WEST REGION (STATISTICAL)",),
    # NHS Midlands needs East + West Midlands from the indicator dataset.
    "MIDLANDS": ("EAST MIDLANDS REGION (STATISTICAL)", "WEST MIDLANDS REGION (STATISTICAL)"),
    # NHS North East and Yorkshire spans two statistical regions.
    "NORTH EAST AND YORKSHIRE": (
        "NORTH EAST REGION (STATISTICAL)",
        "YORKSHIRE AND THE HUMBER REGION (STATISTICAL)",
    ),
    # Unidentified buckets get the England-wide average.
    "UNIDENTIFIED DOCTORS": ("ENGLAND",),
    "UNIDENTIFIED DEPUTISING SERVICES": ("ENGLAND",),
}

OBESITY_COLUMN = "Adult obesity rate 2022/23 (%)"
TARGET_YEAR = "2022/23"
TARGET_SEX = "Persons"
TARGET_AGE = "18+ yrs"


def _to_decimal(value: str) -> Optional[Decimal]:
    value = value.strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def load_indicator_values(path: Path) -> Dict[str, Tuple[Decimal, Optional[Decimal]]]:
    """Return {area_name_upper: (value, denominator)} for the desired slice."""
    results: Dict[str, Tuple[Decimal, Optional[Decimal]]] = {}
    with path.open(encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (
                row["Time period"] != TARGET_YEAR
                or row["Sex"] != TARGET_SEX
                or row["Age"] != TARGET_AGE
                or row["Category Type"].strip()
            ):
                continue
            value = _to_decimal(row["Value"])
            if value is None:
                continue
            denominator = _to_decimal(row["Denominator"])
            area_name = row["Area Name"].strip().upper()
            results[area_name] = (value, denominator)
    if "ENGLAND" not in results:
        raise RuntimeError("England-wide value not found in indicator dataset.")
    return results


def weighted_value(
    area_names: Iterable[str],
    indicator_data: Dict[str, Tuple[Decimal, Optional[Decimal]]],
) -> Optional[Decimal]:
    """Compute a weighted average for the requested set of indicator areas."""
    numerators: List[Decimal] = []
    denominators: List[Decimal] = []
    fallback_value: Optional[Decimal] = None
    for name in area_names:
        record = indicator_data.get(name)
        if record is None:
            continue
        value, denom = record
        fallback_value = value
        if denom is not None:
            numerators.append(value * denom)
            denominators.append(denom)
    if numerators and denominators:
        total_denom = sum(denominators)
        if total_denom > 0:
            return sum(numerators) / total_denom
    return fallback_value


def build_region_rates(indicator_data: Dict[str, Tuple[Decimal, Optional[Decimal]]]) -> Dict[str, Decimal]:
    """Map FOI region labels to a Decimal obesity value."""
    region_rates: Dict[str, Decimal] = {}
    england_value = indicator_data["ENGLAND"][0]
    for region in REGION_TARGETS:
        value = weighted_value(REGION_TARGETS[region], indicator_data)
        if value is None:
            value = england_value
        region_rates[region] = value
    return region_rates


def enrich_foi_data(
    foi_path: Path,
    output_path: Path,
    region_rates: Dict[str, Decimal],
) -> List[str]:
    """Copy FOI rows while appending the obesity column."""
    missing_regions: List[str] = []
    with foi_path.open(encoding="latin-1", newline="") as source, output_path.open(
        "w", encoding="latin-1", newline=""
    ) as target:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        if OBESITY_COLUMN not in fieldnames:
            fieldnames.append(OBESITY_COLUMN)
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            region_key = row["Region"].strip().upper()
            value = region_rates.get(region_key)
            if value is None:
                missing_regions.append(region_key)
                formatted = ""
            else:
                formatted = f"{value:.5f}"
            row[OBESITY_COLUMN] = formatted
            writer.writerow(row)
    return sorted(set(missing_regions))


def main() -> None:
    indicator_data = load_indicator_values(INDICATOR_INPUT)
    region_rates = build_region_rates(indicator_data)
    missing_regions = enrich_foi_data(FOI_INPUT, OUTPUT_PATH, region_rates)
    print(f"Wrote enriched dataset to {OUTPUT_PATH}")
    if missing_regions:
        print("Regions without obesity data:", ", ".join(missing_regions))


if __name__ == "__main__":
    main()
