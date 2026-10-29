import io
import math

import warnings
from pathlib import Path

import certifi
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import scipy.stats as stats
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Job Market Statistical Analysis", page_icon="📊", layout="wide")

st.title("📊 Web-Based Job Market Analysis")
st.caption("Statistical Sampling • Probability • Hypothesis Testing • Regression • Machine Learning")

st.markdown("""
This interactive lab application analyzes job-market data. Upload a CSV/Excel dataset, or use the built-in sample dataset.
For web scraping, only scrape pages you are permitted to access; respect the site's terms, robots.txt, and rate limits.
""")

@st.cache_data(show_spinner=False)
def sample_data():
    rng = np.random.default_rng(42)
    n = 500
    roles = np.array(["Data Analyst", "Software Engineer", "Data Scientist", "HR Analyst", "Business Analyst"])
    cities = np.array(["Bengaluru", "Pune", "Hyderabad", "Mumbai", "Delhi", "Nagpur"])
    exp = rng.integers(0, 16, n)
    role = rng.choice(roles, n, p=[.24, .28, .18, .12, .18])
    city = rng.choice(cities, n)
    base = {"Data Analyst": 6.5, "Software Engineer": 9.0, "Data Scientist": 11.0, "HR Analyst": 5.2, "Business Analyst": 7.1}
    salary = np.array([base[r] for r in role]) + exp * 0.65 + rng.normal(0, 2.4, n)
    salary = np.clip(salary, 2.2, None).round(2)
    remote = rng.choice(["Remote", "Hybrid", "On-site"], n, p=[.25, .4, .35])
    education = rng.choice(["Bachelor", "Master", "PhD", "Diploma"], n, p=[.5, .28, .06, .16])
    demand = np.clip((salary * .7 + exp * .2 + rng.normal(0, 2, n)), 0, 20).round(1)
    return pd.DataFrame({
        "Job_Title": role, "City": city, "Experience_Years": exp,
        "Salary_LPA": salary, "Work_Mode": remote, "Education": education,
        "Demand_Score": demand
    })

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df

def numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def categorical_cols(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

def show_download(df, name="processed_job_market_data.csv"):
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), name, "text/csv")

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose dataset", ["Built-in sample", "Upload CSV / Excel", "Scrape HTML tables"])
    uploaded = None
    scrape_url = ""
    if source == "Upload CSV / Excel":
        uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
    elif source == "Scrape HTML tables":
        scrape_url = st.text_input("Webpage URL", placeholder="https://example.com/jobs")
        st.caption("The page must expose a readable HTML table. JavaScript-rendered pages may not work.")
    st.divider()
    st.caption("Tip: For a real job-market project, use a permitted public dataset with salary, role, location, experience, skills, and employment type.")

df = None
if source == "Built-in sample":
    df = sample_data()
elif source == "Upload CSV / Excel" and uploaded is not None:
    try:
        df = pd.read_excel(uploaded) if uploaded.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
elif source == "Scrape HTML tables" and scrape_url:
    if st.button("Fetch HTML tables"):
        try:
            response = requests.get(
                scrape_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30,
                verify=certifi.where()
            )
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text))
            if tables:
                st.session_state["scraped_tables"] = tables
                st.success(f"Found {len(tables)} HTML table(s).")
            else:
                st.warning("No HTML tables found on that page.")
        except Exception as e:
            st.error(f"Could not scrape tables: {e}")
    if "scraped_tables" in st.session_state and st.session_state["scraped_tables"]:
        idx = st.selectbox("Choose table", range(len(st.session_state["scraped_tables"])), format_func=lambda x: f"Table {x+1}")
        df = st.session_state["scraped_tables"][idx].copy()

if df is None:
    st.info("Choose a data source in the sidebar to begin. The built-in sample dataset is selected by default.")
    st.stop()

df = clean_columns(df)
raw_df = df.copy()
st.success(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")

menu = st.sidebar.selectbox("Analysis module", [
    "1. Dataset overview", "2. Cleaning & preprocessing", "3. Descriptive statistics",
    "4. Visual EDA", "5. Sampling methods", "6. Probability",
    "7. Central Limit Theorem", "8. Confidence intervals & Z/t tests",
    "9. Correlation & covariance", "10. Correlation hypothesis tests",
    "11. Chi-square tests", "12. ANOVA", "13. Regression",
    "14. Classification & model selection", "15. Prediction interface"
])

if menu == "1. Dataset overview":
    st.header("1. Dataset and assignment")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{df.shape[0]:,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Duplicate rows", f"{df.duplicated().sum():,}")
    t1, t2 = st.tabs(["First 5 rows", "Last 5 rows"])
    with t1: st.dataframe(df.head(), use_container_width=True)
    with t2: st.dataframe(df.tail(), use_container_width=True)
    st.subheader("Column names")
    st.write(df.columns.tolist())
    st.subheader("Data types")
    st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame(), use_container_width=True)
    st.subheader("Dataset information")
    info = pd.DataFrame({
        "Non-null count": df.notna().sum(),
        "Missing count": df.isna().sum(),
        "Unique values": df.nunique(dropna=True),
        "Data type": df.dtypes.astype(str)
    })
    st.dataframe(info, use_container_width=True)
    st.subheader("Missing values")
    st.dataframe(df.isna().sum().rename("missing_count").to_frame(), use_container_width=True)
    st.subheader("Duplicate records")
    st.write(f"{df.duplicated().sum()} duplicate row(s)")
    if df.duplicated().any(): st.dataframe(df[df.duplicated(keep=False)].head(100), use_container_width=True)
    st.subheader("Raw data")
    st.dataframe(df, use_container_width=True)
    show_download(df)

elif menu == "2. Cleaning & preprocessing":
    st.header("2. Data cleaning and preprocessing")
    st.write("Cleaning preview is generated from the uploaded/source dataset. The original data remains available in other modules.")
    work = df.copy()
    before_rows = len(work)
    missing_mode = st.selectbox("Missing-value strategy", ["Keep as-is", "Drop rows with missing values", "Impute numeric median / categorical mode"])
    drop_dupes = st.checkbox("Remove duplicate records", value=True)
    if missing_mode == "Drop rows with missing values":
        work = work.dropna()
    elif missing_mode.startswith("Impute"):
        for col in work.columns:
            if pd.api.types.is_numeric_dtype(work[col]):
                work[col] = work[col].fillna(work[col].median())
            else:
                mode = work[col].mode(dropna=True)
                work[col] = work[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    if drop_dupes:
        work = work.drop_duplicates()
    st.metric("Rows removed", before_rows - len(work))
    st.subheader("Numerical variables")
    st.write(numeric_cols(work))
    st.subheader("Categorical variables")
    st.write(categorical_cols(work))
    st.subheader("Cleaned data")
    st.dataframe(work, use_container_width=True)
    show_download(work, "cleaned_job_market_data.csv")
    st.info("Categorical variables are one-hot encoded automatically in the model pipelines. For descriptive analysis, the displayed categories are preserved.")

elif menu == "3. Descriptive statistics":
    st.header("3. Descriptive statistical measures")
    nums = numeric_cols(df)
    if not nums:
        st.warning("No numeric columns found.")
    else:
        st.subheader("Summary statistics")
        st.dataframe(df[nums].describe().T, use_container_width=True)
        measures = pd.DataFrame(index=nums)
        measures["Mean"] = df[nums].mean()
        measures["Median"] = df[nums].median()
        measures["Mode"] = df[nums].mode().iloc[0]
        measures["Variance"] = df[nums].var()
        measures["Standard deviation"] = df[nums].std()
        for q in [0.25, 0.5, 0.75]:
            measures[f"Q{int(q*100)}"] = df[nums].quantile(q)
        measures["IQR"] = df[nums].quantile(.75) - df[nums].quantile(.25)
        st.subheader("Mean, median, mode, variance, standard deviation, quartiles")
        st.dataframe(measures, use_container_width=True)
        st.caption("Variance and standard deviation use sample estimates (ddof=1).")

elif menu == "4. Visual EDA":
    st.header("4. Exploratory data analysis and graphs")
    nums, cats = numeric_cols(df), categorical_cols(df)
    if nums:
        col = st.selectbox("Numeric variable", nums)
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(px.histogram(df, x=col, marginal="rug", title=f"Histogram: {col}"), use_container_width=True)
        with g2:
            st.plotly_chart(px.box(df, y=col, points="outliers", title=f"Box plot: {col}"), use_container_width=True)
    if cats:
        cat = st.selectbox("Categorical variable", cats)
        counts = df[cat].astype(str).value_counts().head(20).rename_axis(cat).reset_index(name="Count")
        g1, g2 = st.columns(2)
        with g1: st.plotly_chart(px.bar(counts, x=cat, y="Count", title=f"Bar chart: {cat}"), use_container_width=True)
        with g2: st.plotly_chart(px.pie(counts, names=cat, values="Count", title=f"Pie chart: {cat}"), use_container_width=True)
    if len(nums) >= 2:
        xcol = st.selectbox("X-axis", nums, index=0, key="eda_x")
        ycol = st.selectbox("Y-axis", nums, index=min(1, len(nums)-1), key="eda_y")
        st.plotly_chart(px.scatter(df, x=xcol, y=ycol, color=cats[0] if cats else None, title=f"{xcol} vs {ycol}"), use_container_width=True)
        line_df = df.reset_index(names="Record")
        st.plotly_chart(px.line(line_df, x="Record", y=ycol, title=f"Line chart: {ycol} by record order"), use_container_width=True)
    else:
        st.info("At least two numeric variables are needed for scatter and line relationship charts.")

elif menu == "5. Sampling methods":
    st.header("5. Probability and sampling methods")
    st.markdown("""
**Sampling** selects part of a population to estimate population characteristics.
- **Simple random:** each record has an equal chance.
- **Systematic:** select every *k*th record after a random start.
- **Stratified:** sample within each group to preserve group representation.
- **Cluster:** select groups/clusters, then include records from selected clusters.
- **Convenience:** choose readily available records (biased, not probability sampling).
- **Quota:** select a fixed number from each group (not necessarily random).
- **Purposive:** select records meeting a chosen rule.
- **Snowball:** participants refer additional participants; often used for hard-to-reach populations.
""")
    n = len(df)
    method = st.selectbox("Sampling technique", ["Simple random", "Systematic", "Stratified", "Cluster", "Convenience", "Quota", "Purposive", "Snowball (simulated referrals)"])
    size = st.slider("Sample size", 1, max(1, n), min(50, n))
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
    cats = categorical_cols(df)
    if method in ["Stratified", "Quota", "Cluster"] and cats:
        group_col = st.selectbox("Grouping column", cats)
    else:
        group_col = None
    if method == "Purposive":
        filter_col = st.selectbox("Column for selection rule", df.columns.tolist())
        rule = st.text_input("Rule: numeric threshold (e.g. 8) or category text", "8")
    if st.button("Draw sample"):
        rng = np.random.default_rng(int(seed))
        if method == "Simple random":
            sample = df.sample(n=size, random_state=int(seed))
        elif method == "Systematic":
            step = max(1, n // size)
            start = int(rng.integers(0, min(step, n)))
            ids = list(range(start, n, step))[:size]
            if len(ids) < size:
                ids += [i for i in range(n) if i not in ids][:size-len(ids)]
            sample = df.iloc[ids]
        elif method == "Stratified" and group_col:
            pieces = []
            for _, grp in df.groupby(group_col, dropna=False):
                take = max(1, round(size * len(grp) / n))
                pieces.append(grp.sample(n=min(take, len(grp)), random_state=int(seed)))
            sample = pd.concat(pieces).head(size)
        elif method == "Cluster" and group_col:
            groups = df[group_col].dropna().unique().tolist()
            if groups:
                cluster_count = max(
                    1,
                    min(
                        len(groups),
                        math.ceil(size / max(1, n / len(groups)))
                    )
                )
                chosen = rng.choice(groups, size=cluster_count, replace=False)
                sample = df[df[group_col].isin(chosen)].head(size)
            else:
                st.warning("No non-missing groups are available for cluster sampling.")
                sample = df.head(0)
        elif method == "Convenience":
            sample = df.head(size)
        elif method == "Quota" and group_col:
            each = max(1, size // max(1, df[group_col].nunique()))
            sample = df.groupby(group_col, group_keys=False, dropna=False).head(each).head(size)
        elif method == "Purposive":
            try:
                threshold = float(rule)
                sample = df[pd.to_numeric(df[filter_col], errors="coerce") >= threshold].head(size)
            except ValueError:
                sample = df[df[filter_col].astype(str).str.contains(rule, case=False, na=False)].head(size)
        else:
            # Snowball is simulated: start with a random seed row, then add rows sharing its category.
            start_idx = int(rng.integers(0, n))
            seed_row = df.iloc[start_idx]
            if cats:
                c = cats[0]
                linked = df[df[c].astype(str) == str(seed_row[c])]
                sample = pd.concat([df.iloc[[start_idx]], linked]).drop_duplicates().head(size)
            else:
                sample = df.sample(n=size, random_state=int(seed))
        st.success(f"{method} sample: {len(sample)} rows")
        st.dataframe(sample, use_container_width=True)
        show_download(sample, f"{method.lower().replace(' ', '_')}_sample.csv")
        if numeric_cols(df):
            compare = pd.DataFrame({"Population mean": df[numeric_cols(df)].mean(), "Sample mean": sample[numeric_cols(df)].mean()})
            st.subheader("Population vs sample means")
            st.dataframe(compare, use_container_width=True)
            st.caption("Convenience, quota, purposive and simulated snowball samples can be biased; treat them as non-probability samples.")

elif menu == "6. Probability":
    st.header("6. Probability calculations")
    st.markdown("For an event A, empirical probability is **P(A) = count(A) / total observations**. Conditional probability is **P(A|B) = P(A and B) / P(B)**.")
    cats = categorical_cols(df)
    if cats:
        col = st.selectbox("Event variable", cats)
        vals = df[col].astype(str).value_counts()
        event = st.selectbox("Event A", vals.index.tolist())
        p_a = float((df[col].astype(str) == event).mean())
        st.metric(f"P({event})", f"{p_a:.4f}", f"{p_a*100:.2f}%")
        st.dataframe(pd.DataFrame({"Count": vals, "Empirical probability": vals / vals.sum(), "Percent": vals / vals.sum() * 100}))
        if len(cats) >= 2:
            other = st.selectbox("Condition variable B", [c for c in cats if c != col])
            bvals = df[other].astype(str).unique().tolist()
            bval = st.selectbox("Condition B value", bvals)
            mask_b = df[other].astype(str) == bval
            if mask_b.sum():
                p_cond = ((df[col].astype(str) == event) & mask_b).sum() / mask_b.sum()
                st.metric(f"P(A | {other}={bval})", f"{p_cond:.4f}")
            else:
                st.warning("Condition has no observations.")
    st.subheader("Bayes' theorem calculator")
    p_a = st.number_input("P(A)", 0.0, 1.0, 0.3, 0.01, key="bayes_a")
    p_b_given_a = st.number_input("P(B | A)", 0.0, 1.0, 0.8, 0.01)
    p_b_given_not_a = st.number_input("P(B | not A)", 0.0, 1.0, 0.2, 0.01)
    denom = p_b_given_a*p_a + p_b_given_not_a*(1-p_a)
    if denom > 0: st.metric("P(A | B)", f"{p_b_given_a*p_a/denom:.4f}")

elif menu == "7. Central Limit Theorem":
    st.header("7. Central Limit Theorem (CLT)")
    st.markdown("The CLT states that, under common conditions, the distribution of sample means becomes approximately normal as sample size increases, even when the population itself is not normal.")
    nums = numeric_cols(df)
    if not nums: st.warning("No numeric columns found.")
    else:
        col = st.selectbox("Population variable", nums)
        sample_size = st.slider("Size of each repeated sample", 2, max(2, min(200, len(df))), min(30, max(2, len(df))))
        repetitions = st.slider("Number of repeated samples", 100, 3000, 1000, 100)
        rng = np.random.default_rng(42)
        arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        means = [rng.choice(arr, size=sample_size, replace=True).mean() for _ in range(repetitions)]
        expected_se = np.std(arr, ddof=1) / np.sqrt(sample_size)
        c1, c2, c3 = st.columns(3)
        c1.metric("Population mean", f"{np.mean(arr):.4f}")
        c2.metric("Mean of sample means", f"{np.mean(means):.4f}")
        c3.metric("Theoretical standard error", f"{expected_se:.4f}")
        st.plotly_chart(px.histogram(x=means, nbins=35, title="Sampling distribution of sample means", labels={"x":"Sample mean", "count":"Frequency"}), use_container_width=True)
        st.write(f"Observed standard deviation of sample means: **{np.std(means, ddof=1):.4f}**")
        st.caption("Larger samples generally reduce the standard error; results depend on the data and sampling design.")

elif menu == "8. Confidence intervals & Z/t tests":
    st.header("8. Confidence intervals and one-sample hypothesis tests")
    nums = numeric_cols(df)
    if not nums: st.warning("A numeric column is required.")
    else:
        col = st.selectbox("Variable", nums)
        x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        conf = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)
        alpha = 1 - conf
        n, mean, sd = len(x), np.mean(x), np.std(x, ddof=1)
        if n > 1:
            tcrit = stats.t.ppf(1-alpha/2, n-1)
            margin = tcrit * sd / np.sqrt(n)
            st.subheader("t-based confidence interval for the mean")
            st.write(f"Sample mean = **{mean:.4f}**; standard deviation = **{sd:.4f}**; n = **{n}**")
            st.write(f"{conf:.0%} confidence interval: **({mean-margin:.4f}, {mean+margin:.4f})**")
        st.divider()
        test = st.radio("Test type", ["One-sample t-test", "One-sample Z-test (known population SD)"], horizontal=True)
        mu0 = st.number_input("Null-hypothesis mean (H₀: μ = μ₀)", value=float(mean))
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "greater", "less"])
        alpha_test = st.selectbox("Significance level α", [0.01, 0.05, 0.10], index=1)
        if test.startswith("One-sample t"):
            result = stats.ttest_1samp(x, popmean=mu0, alternative=alternative)
            statistic, pvalue = float(result.statistic), float(result.pvalue)
            test_name = "One-sample t-test"
        else:
            sigma = st.number_input("Known population standard deviation σ", min_value=0.000001, value=float(sd if sd > 0 else 1.0))
            z = (mean-mu0)/(sigma/np.sqrt(n))
            if alternative == "two-sided": pvalue = 2*stats.norm.sf(abs(z))
            elif alternative == "greater": pvalue = stats.norm.sf(z)
            else: pvalue = stats.norm.cdf(z)
            statistic, test_name = float(z), "One-sample Z-test"
        st.metric(f"{test_name} statistic", f"{statistic:.4f}")
        st.metric("p-value", f"{pvalue:.6f}")
        st.success("Reject H₀ at the selected significance level." if pvalue < alpha_test else "Fail to reject H₀ at the selected significance level.")
        st.caption("A Z-test is appropriate when the population standard deviation is known and assumptions are met. Do not interpret failure to reject as proof that H₀ is true.")

elif menu == "9. Correlation & covariance":
    st.header("9. Relationship analysis: correlation and covariance")
    nums = numeric_cols(df)
    if len(nums) < 2: st.warning("At least two numeric columns are required.")
    else:
        selected = st.multiselect("Select numeric variables", nums, default=nums[:min(6, len(nums))])
        if len(selected) >= 2:
            st.subheader("Covariance matrix")
            st.dataframe(df[selected].cov(), use_container_width=True)
            st.subheader("Pearson correlation matrix")
            corr = df[selected].corr(method="pearson")
            st.dataframe(corr, use_container_width=True)
            st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Correlation heatmap"), use_container_width=True)
            xcol = st.selectbox("X variable", selected, index=0)
            ycol = st.selectbox("Y variable", selected, index=min(1, len(selected)-1))
            pair = df[[xcol, ycol]].dropna()
            r = pair[xcol].corr(pair[ycol])
            cov = pair[xcol].cov(pair[ycol])
            st.write(f"**Pearson r = {r:.4f}**; **sample covariance = {cov:.4f}**.")
            strength = "very weak" if abs(r)<.2 else "weak" if abs(r)<.4 else "moderate" if abs(r)<.6 else "strong" if abs(r)<.8 else "very strong"
            direction = "positive" if r > 0 else "negative" if r < 0 else "no linear"
            st.info(f"Interpretation: {strength} {direction} linear association. Correlation does not establish causation.")
            st.plotly_chart(px.scatter(pair, x=xcol, y=ycol, trendline="ols", title=f"{xcol} vs {ycol}"), use_container_width=True)

elif menu == "10. Correlation hypothesis tests":
    st.header("10. Pearson and Spearman correlation hypothesis testing")
    nums = numeric_cols(df)
    if len(nums) < 2: st.warning("At least two numeric columns are required.")
    else:
        xcol = st.selectbox("Variable X", nums, index=0)
        ycol = st.selectbox("Variable Y", nums, index=min(1, len(nums)-1))
        pair = df[[xcol, ycol]].dropna()
        if len(pair) < 3:
            st.warning("At least 3 complete pairs are required.")
        else:
            p = stats.pearsonr(pair[xcol], pair[ycol])
            s = stats.spearmanr(pair[xcol], pair[ycol])
            table = pd.DataFrame([
                {"Method":"Pearson", "Coefficient":p.statistic, "p-value":p.pvalue, "Null hypothesis":"Population linear correlation is zero"},
                {"Method":"Spearman", "Coefficient":s.statistic, "p-value":s.pvalue, "Null hypothesis":"Population rank correlation is zero"}
            ])
            st.dataframe(table, use_container_width=True)
            alpha = st.selectbox("Significance level", [0.01, 0.05, 0.10], index=1, key="corr_alpha")
            for label, res in [("Pearson", p), ("Spearman", s)]:
                st.write(f"**{label}:** {'Statistically significant; reject H₀.' if res.pvalue < alpha else 'Not statistically significant; fail to reject H₀.'} (p={res.pvalue:.6f})")
            st.caption("Pearson measures linear association and is sensitive to outliers. Spearman measures monotonic rank association.")

elif menu == "11. Chi-square tests":
    st.header("11. Chi-square tests")
    cats = categorical_cols(df)
    if len(cats) >= 1:
        test_kind = st.radio("Test", ["Independence test (two categorical variables)", "Goodness-of-fit (one categorical variable)"], horizontal=False)
        if test_kind.startswith("Independence"):
            if len(cats) < 2: st.warning("Two categorical variables are required.")
            else:
                c1 = st.selectbox("Variable A", cats, index=0)
                c2 = st.selectbox("Variable B", cats, index=min(1, len(cats)-1))
                contingency = pd.crosstab(df[c1], df[c2])
                st.subheader("Contingency table")
                st.dataframe(contingency, use_container_width=True)
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
                    st.write(f"χ² = **{chi2:.4f}**, df = **{dof}**, p-value = **{pvalue:.6f}**")
                    st.write("Reject independence at α=0.05." if pvalue < .05 else "Insufficient evidence of an association at α=0.05.")
                    st.caption(f"Expected counts below 5: {(expected < 5).sum()}. Small expected counts may make the approximation unreliable.")
        else:
            col = st.selectbox("Categorical variable", cats)
            counts = df[col].value_counts()
            expected_pct = st.text_input("Expected proportions, comma-separated (optional)", "")
            if st.button("Run goodness-of-fit test"):
                observed = counts.to_numpy()
                try:
                    if expected_pct.strip():
                        probs = np.array([float(v.strip()) for v in expected_pct.split(",")])
                        if len(probs) != len(observed) or np.any(probs < 0) or probs.sum() <= 0:
                            raise ValueError("Provide one non-negative expected proportion per observed category.")
                        expected = probs / probs.sum() * observed.sum()
                    else:
                        expected = np.repeat(observed.sum()/len(observed), len(observed))
                    result = stats.chisquare(observed, f_exp=expected)
                    st.write(f"χ² = **{result.statistic:.4f}**, p-value = **{result.pvalue:.6f}**")
                    st.write("Reject the goodness-of-fit null at α=0.05." if result.pvalue < .05 else "Fail to reject the goodness-of-fit null at α=0.05.")
                except Exception as e: st.error(str(e))
    else:
        st.warning("At least one categorical column is required.")

elif menu == "12. ANOVA":
    st.header("12. One-way ANOVA")
    nums, cats = numeric_cols(df), categorical_cols(df)
    if not nums or not cats: st.warning("ANOVA requires a numeric response and a categorical grouping variable.")
    else:
        ycol = st.selectbox("Numeric response", nums)
        group = st.selectbox("Grouping category", cats)
        groups = [g[ycol].dropna().to_numpy() for _, g in df.groupby(group) if len(g[ycol].dropna()) > 0]
        if len(groups) >= 2:
            result = stats.f_oneway(*groups)
            st.write(f"F-statistic = **{result.statistic:.4f}**")
            st.write(f"p-value = **{result.pvalue:.6f}**")
            st.write("At least one group mean differs significantly at α=0.05." if result.pvalue < .05 else "No statistically significant difference among group means detected at α=0.05.")
            st.dataframe(df.groupby(group)[ycol].agg(["count","mean","std","median"]).reset_index(), use_container_width=True)
            st.plotly_chart(px.box(df, x=group, y=ycol, points="outliers", title=f"{ycol} by {group}"), use_container_width=True)
            st.caption("Classical one-way ANOVA assumes independent observations, approximately normal residuals, and similar group variances.")

elif menu == "13. Regression":
    st.header("13. Linear and multiple regression")
    nums = numeric_cols(df)
    if len(nums) < 2: st.warning("At least two numeric columns are required.")
    else:
        target = st.selectbox("Numeric target (dependent variable)", nums)
        features = st.multiselect("Predictor variables (independent variables)", [c for c in df.columns if c != target], default=[c for c in nums if c != target][:2])
        test_size = st.slider("Test set fraction", .1, .4, .2, .05)
        if features and st.button("Train regression model"):
            data = df[features + [target]].dropna()
            X, y = data[features], data[target]
            cat_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
            num_features = [c for c in features if c not in cat_features]
            prep = ColumnTransformer([
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_features),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features)
            ])
            model = Pipeline([("preprocess", prep), ("model", LinearRegression())])
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            st.metric("R²", f"{r2_score(yte, pred):.4f}")
            st.metric("MAE", f"{mean_absolute_error(yte, pred):.4f}")
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(yte, pred)):.4f}")
            result_df = pd.DataFrame({"Actual": yte.to_numpy(), "Predicted": pred})
            st.plotly_chart(px.scatter(result_df, x="Actual", y="Predicted", title="Actual vs predicted"), use_container_width=True)
            st.dataframe(result_df.head(30), use_container_width=True)
            st.session_state["regression_model"] = model
            st.session_state["regression_features"] = features
            st.session_state["regression_target"] = target
            buffer = io.BytesIO()
            joblib.dump({"model": model, "features": features, "target": target, "task": "regression"}, buffer)
            st.download_button("Download regression model (.joblib)", buffer.getvalue(), "job_market_regression.joblib")
            st.caption("R² can be negative on test data; this means the model performs worse than predicting the test-set mean.")

elif menu == "14. Classification & model selection":
    st.header("14. Classification and model selection")
    st.markdown("Choose a categorical target such as Job_Title, Work_Mode, or Education. For a numeric target such as salary, use the Regression module.")
    targets = categorical_cols(df)
    if not targets:
        st.warning("No categorical target found. Create or select a categorical outcome to train classifiers.")
    else:
        target = st.selectbox("Classification target", targets)
        features = st.multiselect("Feature columns", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
        test_size = st.slider("Test set fraction", .1, .4, .2, .05, key="class_test_size")
        scale = st.checkbox("Scale numeric features (recommended for KNN/SVM/logistic regression)", value=True)
        if features and st.button("Train and compare all classifiers"):
            data = df[features + [target]].dropna(subset=[target]).copy()
            X = data[features]
            y = data[target].astype(str)
            if y.nunique() < 2:
                st.error("The target must contain at least two classes.")
            elif y.value_counts().min() < 2:
                st.error("Each class needs at least two rows for a stratified train/test split.")
            else:
                cat_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
                num_features = [c for c in features if c not in cat_features]
                numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
                if scale: numeric_steps.append(("scale", StandardScaler()))
                preprocessor = ColumnTransformer([
                    ("num", Pipeline(numeric_steps), num_features),
                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features)
                ])
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=2000),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
                    "KNN": KNeighborsClassifier(n_neighbors=max(1, min(5, len(Xtr)))),
                    "SVM": SVC(kernel="rbf", class_weight="balanced")
                }
                results, fitted = [], {}
                for name, estimator in models.items():
                    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
                    try:
                        pipe.fit(Xtr, ytr)
                        pred = pipe.predict(Xte)
                        acc = accuracy_score(yte, pred)
                        results.append({"Model": name, "Accuracy": acc})
                        fitted[name] = pipe
                    except Exception as e:
                        st.warning(f"{name} could not train: {e}")
                if results:
                    scores = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
                    st.subheader("Model comparison")
                    st.dataframe(scores, use_container_width=True)
                    st.plotly_chart(px.bar(scores, x="Model", y="Accuracy", range_y=[0,1], title="Test accuracy comparison"), use_container_width=True)
                    best_name = scores.iloc[0]["Model"]
                    best = fitted[best_name]
                    pred = best.predict(Xte)
                    st.success(f"Best model by test accuracy: {best_name} ({accuracy_score(yte, pred):.3f})")
                    st.subheader("Classification report")
                    st.code(classification_report(yte, pred, zero_division=0))
                    labels = sorted(y.unique())
                    cm = confusion_matrix(yte, pred, labels=labels)
                    st.plotly_chart(px.imshow(cm, x=labels, y=labels, text_auto=True, labels={"x":"Predicted","y":"Actual","color":"Count"}, title="Confusion matrix"), use_container_width=True)
                    model_bundle = {"model": best, "features": features, "target": target, "task": "classification", "classes": labels}
                    buffer = io.BytesIO()
                    joblib.dump(model_bundle, buffer)
                    st.download_button("⬇️ Download best model (.joblib)", buffer.getvalue(), "best_job_market_model.joblib")
                    st.session_state["best_model_bundle"] = model_bundle
                    st.session_state["best_model_scores"] = scores
                    st.session_state["best_model_name"] = best_name
                    st.session_state["prediction_features"] = features
                    st.session_state["prediction_target"] = target
                    st.session_state["prediction_model"] = best
                    st.caption("Model selection here uses held-out test accuracy. For a formal report, also use cross-validation and compare precision, recall, F1, class balance, and business costs.")
        st.info("For classification, the model learns to predict a selected category. Do not use a field that directly reveals the target as a predictor.")

elif menu == "15. Prediction interface":
    st.header("15. Prediction interface")
    st.markdown("Train a model in the Classification module first, or upload a compatible `.joblib` bundle saved by this app.")
    model_file = st.file_uploader("Optional: load saved model bundle", type=["joblib"], key="predict_upload")
    bundle = None
    if model_file is not None:
        try: bundle = joblib.load(model_file)
        except Exception as e: st.error(f"Could not load model: {e}")
    if bundle is None:
        bundle = st.session_state.get("best_model_bundle") or st.session_state.get("regression_model")
        if bundle is not None and not isinstance(bundle, dict):
            bundle = {"model": bundle, "features": st.session_state.get("regression_features", []), "target": st.session_state.get("regression_target", ""), "task": "regression"}
    if not bundle:
        st.warning("No saved model is available yet. Train a model in Classification or Regression first.")
    else:
        model = bundle["model"]
        features = bundle["features"]
        task = bundle.get("task", "classification")
        target = bundle.get("target", "prediction")
        st.write(f"**Model task:** {task} | **Target:** {target}")
        input_values = {}
        for col in features:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                default = float(vals.median()) if len(vals) else 0.0
                input_values[col] = st.number_input(col, value=default, key=f"pred_{col}")
            elif col in df.columns:
                options = df[col].dropna().astype(str).unique().tolist()
                input_values[col] = st.selectbox(col, options or ["Unknown"], key=f"pred_{col}")
            else:
                input_values[col] = st.text_input(col, key=f"pred_{col}")
        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_values], columns=features)
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted {target}: {prediction}")
                if hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(input_df)[0]
                        classes = getattr(model, "classes_", bundle.get("classes", []))
                        st.dataframe(pd.DataFrame({"Class": classes, "Probability": probabilities}).sort_values("Probability", ascending=False), use_container_width=True)
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Academic use: report your dataset source, sample design, assumptions, significance level, and limitations. Synthetic sample data is for demonstrating the application only, not for making claims about the real job market.")
