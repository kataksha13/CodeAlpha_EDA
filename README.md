# CodeAlpha_EDA

**CodeAlpha Data Analytics Internship, Task 2: Exploratory Data Analysis**

Exploratory analysis of the book dataset scraped in Task 1
([CodeAlpha_WebScraping](https://github.com/YOUR-USERNAME/CodeAlpha_WebScraping)).
The dataset has 1,000 books from Books to Scrape with title, price, star rating and stock status.

## Questions asked

1. What does the dataset look like, and is the data clean?
2. How are book prices distributed?
3. How are ratings distributed?
4. Do higher-rated books cost more? (hypothesis tests)
5. Are there price outliers?
6. How many books are in stock?
7. Which categories are biggest and most expensive? (only if category data is available)

## Methods

- Structure and quality checks: shape, data types, missing values, duplicates
- Descriptive statistics and skewness
- Histograms, bar charts and boxplots (matplotlib, seaborn)
- Hypothesis tests (scipy): Spearman correlation, Kruskal-Wallis, Mann-Whitney U
- Outlier detection with the IQR rule and z-scores

## Key findings

> Replace this section with your own results after running the script.
> Open `outputs/eda_report.txt` and write 4 to 5 short points, for example:

- The dataset has 1000 rows and no missing values or duplicates.
- Prices range from £__ to £__ with an average of £__; the distribution is (roughly even / skewed).
- Most books have a rating of __ stars.
- Spearman correlation between rating and price was __ (p = __), so (there is / is no) significant relationship.
- __ price outliers were found using the IQR method.

## Charts

All charts are saved in the `outputs/` folder:

| File | What it shows |
|---|---|
| 01_price_distribution.png | Histogram of prices with mean and median |
| 02_rating_distribution.png | Number of books per star rating |
| 03_price_by_rating.png | Price spread for each rating |
| 04_price_outliers.png | Boxplot used for outlier detection |

## How to run

```bash
pip install -r requirements.txt
python eda.py
```

The script reads `data/books_dataset.csv` and writes charts plus `outputs/eda_report.txt`.

## Project structure

```
CodeAlpha_EDA/
├── data/books_dataset.csv
├── outputs/            (charts and report)
├── eda.py
├── requirements.txt
└── README.md
```

## Tools

Python, pandas, NumPy, matplotlib, seaborn, SciPy
