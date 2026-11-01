"""
CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)
Dataset: books_dataset.csv, scraped in Task 1 from books.toscrape.com

Usage:
    python eda.py                          # uses data/books_dataset.csv
    python eda.py --file path/to/file.csv  # use a different file

Outputs:
    outputs/*.png         charts
    outputs/eda_report.txt  all printed findings in one text file
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

OUT_DIR = Path("outputs")
sns.set_theme(style="whitegrid")
REPORT = []


def say(text=""):
    """Print a line and also store it for the report file."""
    print(text)
    REPORT.append(str(text))


def section(title):
    say("\n" + "=" * 70)
    say(title)
    say("=" * 70)


def save_chart(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    say(f"[chart saved: {path}]")


# --------------------------------------------------------------------------
# 1. Questions to answer
# --------------------------------------------------------------------------
def questions():
    section("1. QUESTIONS WE WANT TO ANSWER")
    say("Q1. What does the dataset look like, and is the data clean?")
    say("Q2. How are book prices distributed?")
    say("Q3. How are ratings distributed?")
    say("Q4. Do higher-rated books cost more? (hypothesis test)")
    say("Q5. Are there price outliers?")
    say("Q6. How many books are in stock?")
    say("Q7. (If available) Which categories are biggest and most expensive?")


# --------------------------------------------------------------------------
# 2. Structure and data quality
# --------------------------------------------------------------------------
def structure(df):
    section("2. DATA STRUCTURE AND QUALITY (Q1)")
    say(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
    say("\nColumn names and data types:")
    say(df.dtypes.to_string())
    say("\nMissing values per column:")
    say(df.isna().sum().to_string())
    say(f"\nDuplicate rows: {df.duplicated().sum()}")
    say(f"Duplicate titles: {df['title'].duplicated().sum()} (different books can share a title)")
    say("\nNumeric summary:")
    say(df.describe().round(2).to_string())


# --------------------------------------------------------------------------
# 3. Distributions
# --------------------------------------------------------------------------
def distributions(df):
    section("3. DISTRIBUTIONS (Q2, Q3, Q6)")

    p = df["price_gbp"]
    say(f"Price: min £{p.min():.2f}, max £{p.max():.2f}, mean £{p.mean():.2f}, median £{p.median():.2f}")
    say(f"Price skewness: {p.skew():.2f} (close to 0 means roughly symmetric)")

    plt.figure(figsize=(8, 5))
    sns.histplot(p, bins=30, kde=True, color="#4C72B0")
    plt.axvline(p.mean(), color="red", linestyle="--", label=f"Mean £{p.mean():.2f}")
    plt.axvline(p.median(), color="green", linestyle="--", label=f"Median £{p.median():.2f}")
    plt.title("Distribution of Book Prices")
    plt.xlabel("Price (GBP)")
    plt.legend()
    save_chart("01_price_distribution.png")

    counts = df["rating"].value_counts().sort_index()
    say("\nBooks per rating:")
    say(counts.to_string())

    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, color="#55A868")
    plt.title("Number of Books by Star Rating")
    plt.xlabel("Rating (stars)")
    plt.ylabel("Number of books")
    save_chart("02_rating_distribution.png")

    stock = df["in_stock"].value_counts()
    say("\nStock status:")
    say(stock.to_string())
    say(f"Share in stock: {df['in_stock'].mean() * 100:.1f}%")


# --------------------------------------------------------------------------
# 4. Relationships and hypothesis tests
# --------------------------------------------------------------------------
def relationships(df):
    section("4. PRICE VS RATING AND HYPOTHESIS TESTS (Q4)")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="rating", y="price_gbp", color="#8172B2")
    plt.title("Price by Star Rating")
    plt.xlabel("Rating (stars)")
    plt.ylabel("Price (GBP)")
    save_chart("03_price_by_rating.png")

    say("Average price per rating:")
    say(df.groupby("rating")["price_gbp"].agg(["count", "mean", "median"]).round(2).to_string())

    # Test 1: correlation between rating and price
    rho, p_corr = stats.spearmanr(df["rating"], df["price_gbp"])
    say(f"\nTest 1: Spearman correlation between rating and price")
    say(f"  H0: there is no relationship between rating and price")
    say(f"  rho = {rho:.3f}, p-value = {p_corr:.4f}")
    say("  Result: " + ("reject H0, a relationship exists" if p_corr < 0.05
                        else "fail to reject H0, no significant relationship (alpha = 0.05)"))

    # Test 2: do prices differ across rating groups?
    groups = [g["price_gbp"].values for _, g in df.groupby("rating")]
    h, p_kw = stats.kruskal(*groups)
    say(f"\nTest 2: Kruskal-Wallis test, price across the 5 rating groups")
    say(f"  H0: price distributions are the same for every rating")
    say(f"  H = {h:.3f}, p-value = {p_kw:.4f}")
    say("  Result: " + ("reject H0, prices differ between ratings" if p_kw < 0.05
                        else "fail to reject H0, prices do not differ by rating (alpha = 0.05)"))

    # Test 3: in stock vs price
    in_s = df.loc[df["in_stock"], "price_gbp"]
    out_s = df.loc[~df["in_stock"], "price_gbp"]
    if len(out_s) > 1:
        u, p_mw = stats.mannwhitneyu(in_s, out_s)
        say(f"\nTest 3: Mann-Whitney U, price of in-stock vs out-of-stock books, p = {p_mw:.4f}")
    else:
        say("\nTest 3 skipped: (almost) all books are in stock, so there is nothing to compare.")


# --------------------------------------------------------------------------
# 5. Outliers
# --------------------------------------------------------------------------
def outliers(df):
    section("5. OUTLIER DETECTION (Q5)")
    q1, q3 = df["price_gbp"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = df[(df["price_gbp"] < low) | (df["price_gbp"] > high)]
    say(f"IQR method: Q1 = £{q1:.2f}, Q3 = £{q3:.2f}, allowed range £{low:.2f} to £{high:.2f}")
    say(f"Outliers found: {len(out)}")
    if len(out):
        say(out[["title", "price_gbp"]].head(10).to_string(index=False))

    plt.figure(figsize=(8, 3.5))
    sns.boxplot(x=df["price_gbp"], color="#C44E52")
    plt.title("Price Outlier Check (Boxplot)")
    plt.xlabel("Price (GBP)")
    save_chart("04_price_outliers.png")

    z = np.abs(stats.zscore(df["price_gbp"]))
    say(f"\nBooks with |z-score| > 3: {(z > 3).sum()}")


# --------------------------------------------------------------------------
# 6. Categories (only if the scraper was run with --details)
# --------------------------------------------------------------------------
def categories(df):
    if "category" not in df.columns:
        section("6. CATEGORIES (Q7)")
        say("No 'category' column found. Run `python scraper.py --details` in the Task 1 project")
        say("and use that CSV if you want category analysis too. Skipping this part.")
        return

    section("6. CATEGORIES (Q7)")
    top = df["category"].value_counts().head(10)
    say("Top 10 categories by number of books:")
    say(top.to_string())

    plt.figure(figsize=(9, 5))
    sns.barplot(x=top.values, y=top.index, color="#4C72B0")
    plt.title("Top 10 Categories by Number of Books")
    plt.xlabel("Number of books")
    plt.ylabel("")
    save_chart("05_top_categories.png")

    avg = (df.groupby("category")["price_gbp"].agg(["count", "mean"])
             .query("count >= 10").sort_values("mean", ascending=False).head(10))
    say("\nMost expensive categories (at least 10 books):")
    say(avg.round(2).to_string())

    plt.figure(figsize=(9, 5))
    sns.barplot(x=avg["mean"], y=avg.index, color="#DD8452")
    plt.title("Average Price by Category (top 10)")
    plt.xlabel("Average price (GBP)")
    plt.ylabel("")
    save_chart("06_price_by_category.png")


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EDA on the scraped books dataset")
    parser.add_argument("--file", default="data/books_dataset.csv", help="path to the CSV")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"Could not find {csv_path}. Put books_dataset.csv inside the 'data' folder.")
        return

    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(csv_path)

    questions()
    structure(df)
    distributions(df)
    relationships(df)
    outliers(df)
    categories(df)

    section("7. DONE")
    say("Read the findings above and write 4 to 5 key points in the README.")

    (OUT_DIR / "eda_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\nFull report saved to {OUT_DIR / 'eda_report.txt'}")


if __name__ == "__main__":
    main()
