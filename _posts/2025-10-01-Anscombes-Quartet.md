---
layout: post
title: Anscombe's Quartet Analysis & Visualization
---
# Anscombe’s Quartet Analysis

In this notebook, we analyze Anscombe’s Quartet, a famous dataset consisting of four datasets with nearly identical summary statistics (mean, variance, correlation, and regression line), yet very different distributions. 

The goals of this analysis are to:
- Calculate key summary statistics (mean, variance, correlation, regression slope/intercept, and R²) for each dataset.
- Visualize all four datasets using scatter plots and regression lines.
- Apply Tufte’s principles of good visualization to maximize clarity and minimize chartjunk.
- Reflect on why visual inspection is crucial in addition to statistical summaries.



```python
# ===== Anscombe's Quartet Analysis & Visualization =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Create the dataset ---
data = {
    'x1': [10,8,13,9,11,14,6,4,12,7,5],
    'y1': [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68],
    'y2': [9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74],
    'y3': [7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73],
    'x4': [8,8,8,8,8,8,8,19,8,8,8],
    'y4': [6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.5,5.56,7.91,6.89]
}

df = pd.DataFrame(data)

# --- Function to calculate summary statistics ---
def summarize(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        'mean_x': np.mean(x),
        'mean_y': np.mean(y),
        'var_x': np.var(x, ddof=1),
        'var_y': np.var(y, ddof=1),
        'correlation': np.corrcoef(x, y)[0,1],
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2
    }

```

# Summary Statistics Analysis

The calculated summary statistics for the four datasets are:

| Dataset   | Mean X | Mean Y | Var X | Var Y | Correlation | Slope | Intercept | R² |
|-----------|--------|--------|-------|-------|-------------|-------|-----------|----|
| Dataset 1 | 9.0    | 7.50   | 11.0  | 4.13  | 0.82        | 0.50  | 3.00      | 0.67 |
| Dataset 2 | 9.0    | 7.50   | 11.0  | 4.13  | 0.82        | 0.50  | 3.00      | 0.67 |
| Dataset 3 | 9.0    | 7.50   | 11.0  | 4.13  | 0.82        | 0.50  | 3.00      | 0.67 |
| Dataset 4 | 9.0    | 7.50   | 11.0  | 4.13  | 0.82        | 0.50  | 3.00      | 0.67 |

**Observations:**
- All datasets have nearly identical summary statistics.
- Means and variances of X and Y are the same across all datasets.
- Correlation coefficients and R² values are identical, suggesting similar linear relationships.
- Despite this similarity, the underlying distributions and patterns differ significantly, as we will see in the visualizations.



```python
# --- Compute statistics for each dataset ---
datasets = {
    'Dataset 1': ('x1','y1'),
    'Dataset 2': ('x1','y2'),
    'Dataset 3': ('x1','y3'),
    'Dataset 4': ('x4','y4')
}

rows = []
for name, (x_col, y_col) in datasets.items():
    stats_dict = summarize(df[x_col], df[y_col])
    rows.append({
        'Dataset': name,
        'Mean X': stats_dict['mean_x'],
        'Mean Y': stats_dict['mean_y'],
        'Var X': stats_dict['var_x'],
        'Var Y': stats_dict['var_y'],
        'Correlation': stats_dict['correlation'],
        'Slope': stats_dict['slope'],
        'Intercept': stats_dict['intercept'],
        'R²': stats_dict['r_squared']
    })

summary_stats = pd.DataFrame(rows)

print("=== Anscombe's Quartet Summary Statistics ===")
print(summary_stats)

# --- Visualization using Tufte Principles ---
fig, axes = plt.subplots(2,2,figsize=(12,10))
axes = axes.flatten()

for i, (name, (x_col, y_col)) in enumerate(datasets.items()):
    x = df[x_col]
    y = df[y_col]
    ax = axes[i]
    # Scatter plot
    ax.scatter(x, y, color='blue', s=50, edgecolor='k')
    # Regression line
    slope, intercept, _, _, _ = stats.linregress(x, y)
    ax.plot(x, intercept + slope*x, color='red', lw=2)
    # Titles and labels
    ax.set_title(name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Remove top/right spines for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle("Anscombe's Quartet: Visualization with Regression Lines", fontsize=16)
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()

```

    === Anscombe's Quartet Summary Statistics ===
         Dataset  Mean X    Mean Y  Var X     Var Y  Correlation     Slope  \
    0  Dataset 1     9.0  7.500909   11.0  4.127269     0.816421  0.500091   
    1  Dataset 2     9.0  7.500909   11.0  4.127629     0.816237  0.500000   
    2  Dataset 3     9.0  7.500000   11.0  4.122620     0.816287  0.499727   
    3  Dataset 4     9.0  7.500909   11.0  4.123249     0.816521  0.499909   
    
       Intercept        R²  
    0   3.000091  0.666542  
    1   3.000909  0.666242  
    2   3.002455  0.666324  
    3   3.001727  0.666707  



    
![png](/assets/output_3_1.png)
    


# Visualization Insights

The scatter plots with regression lines reveal important differences that are **not captured by summary statistics**:

- **Dataset 1:** Shows a fairly linear relationship; the regression line fits the data well.
- **Dataset 2:** Displays a clear curve, so a linear model is not perfect even though R² is identical.
- **Dataset 3:** Contains an outlier that dramatically affects the slope of the regression line.
- **Dataset 4:** Has almost all points with the same X value except one outlier, causing the regression to appear similar statistically but misleading visually.

**Application of Tufte’s Principles:**
- Clean axes and minimal spines remove visual clutter.
- Small multiples allow easy comparison across datasets.
- No unnecessary colors, 3D effects, or embellishments (maximizing data-to-ink ratio).
- Titles, axis labels, and regression lines are clear and readable.

**Conclusion:**  
Even when summary statistics are identical, visualization is essential to detect non-linear patterns, outliers, or misleading trends. These plots demonstrate why exploratory data analysis should combine both **statistics and visualization**.

# Reflection

Through this assignment, I learned:

- **Statistics alone can be misleading:** Datasets can share mean, variance, correlation, and R² yet look very different.
- **Visualization reveals hidden patterns:** Scatter plots show curvature, clustering, and outliers that numeric summaries hide.
- **Importance of good visualization design:** Applying Tufte’s principles improved clarity and made comparisons easier.
- **Resourcefulness:** I used online documentation, generative AI, and example code to calculate regression, correlation, and variance.
- **Peer teaching:** Explaining the difference between R² and visual fit helped me understand why statistics without context can be deceptive.

Overall, this exercise reinforced that **exploratory data analysis should always combine numbers and visuals** to ensure accurate interpretations.
