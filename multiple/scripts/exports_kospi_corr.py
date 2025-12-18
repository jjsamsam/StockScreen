"""
Fetch Korea daily export values and KOSPI index, then compute the correlation.

Usage examples:
python scripts/exports_kospi_corr.py --start 2018-01-01 --end 2024-12-31 \\
    --ecos-key "$ECOS_API_KEY" --ecos-stat 024Y001 --ecos-cycle D --ecos-items 0

If you already have a CSV of exports (columns: date,value), skip ECOS and do:
python scripts/exports_kospi_corr.py --exports-csv path/to/exports.csv

Outputs:
- cache/exports_kospi_kor.csv  (joined dataset)
- correlation printed to stdout
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


EXPORTS_OUTPUT = Path("cache/exports_kospi_kor.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch exports and KOSPI, then compute correlation.")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD (default: 2018-01-01)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--ecos-key", default=os.getenv("ECOS_API_KEY"), help="ECOS API key (env ECOS_API_KEY if omitted)")
    parser.add_argument("--ecos-stat", help="ECOS STAT_CODE for exports (required unless --exports-csv is used)")
    parser.add_argument("--ecos-cycle", default="D", help="ECOS cycle (D=Daily, M=Monthly, etc.)")
    parser.add_argument(
        "--ecos-items",
        nargs="*",
        default=None,
        help="Optional ECOS ITEM_CODE1..4 list, depends on STAT_CODE definition.",
    )
    parser.add_argument("--exports-csv", help="Path to existing exports CSV with columns date,value")
    parser.add_argument("--kospi-symbol", default="^KS11", help="Symbol for KOSPI (default: ^KS11 via yfinance)")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds (default: 20)")
    return parser.parse_args()


def _to_yyyymmdd(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")


def build_ecos_url(api_key: str, stat_code: str, cycle: str, start: str, end: str, items: list[str] | None) -> str:
    segments = [
        "http://ecos.bok.or.kr/api/StatisticSearch",
        api_key,
        "json",
        "kr",
        "1",
        "100000",
        stat_code,
        cycle,
        _to_yyyymmdd(start),
        _to_yyyymmdd(end),
    ]
    if items:
        segments.extend(items)
    return "/".join(segments)


def fetch_exports_from_ecos(api_key: str, stat_code: str, cycle: str, start: str, end: str, items, timeout: int) -> pd.DataFrame:
    url = build_ecos_url(api_key, stat_code, cycle, start, end, items)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("StatisticSearch", {}).get("row", [])
    if not rows:
        raise ValueError("ECOS response has no data rows. Check STAT_CODE / items / date range.")
    df = pd.DataFrame(rows)
    if "TIME" not in df.columns or "DATA_VALUE" not in df.columns:
        raise ValueError(f"Unexpected ECOS columns: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["TIME"])
    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    df = df[["date", "value"]].dropna()
    return df.sort_values("date")


def load_exports_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "value" not in df.columns:
        raise ValueError("CSV must contain a 'value' column.")
    return df[["date", "value"]].dropna().sort_values("date")


def fetch_kospi(symbol: str, start: str, end: str) -> pd.DataFrame:
    kospi = yf.download(symbol, start=start, end=end)
    if kospi.empty:
        raise ValueError("No KOSPI data returned; check symbol or date range.")
    kospi = kospi.reset_index()[["Date", "Close"]]
    kospi.columns = ["date", "close"]
    return kospi.dropna().sort_values("date")


def compute_correlation(exports: pd.DataFrame, kospi: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    merged = exports.merge(kospi, on="date", how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates between exports and KOSPI.")
    corr = merged["value"].corr(merged["close"])
    return merged, corr


def main() -> None:
    args = parse_args()
    end_date = args.end or datetime.today().strftime("%Y-%m-%d")

    if args.exports_csv:
        exports = load_exports_from_csv(args.exports_csv)
    else:
        if not args.ecos_key or not args.ecos_stat:
            raise SystemExit("Provide either --exports-csv or ECOS params (--ecos-key, --ecos-stat).")
        exports = fetch_exports_from_ecos(
            api_key=args.ecos_key,
            stat_code=args.ecos_stat,
            cycle=args.ecos_cycle,
            start=args.start,
            end=end_date,
            items=args.ecos_items,
            timeout=args.timeout,
        )

    kospi = fetch_kospi(args.kospi_symbol, args.start, end_date)
    merged, corr = compute_correlation(exports, kospi)

    EXPORTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(EXPORTS_OUTPUT, index=False)

    print(f"Saved merged dataset to {EXPORTS_OUTPUT}")
    print(f"Overlapping rows: {len(merged)}")
    print(f"Correlation (exports vs KOSPI close): {corr:.4f}")


if __name__ == "__main__":
    main()
