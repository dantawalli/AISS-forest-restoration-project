# Plot Placement Guide for Modelling Notebook
## Quick Reference for Adding Visualizations

This guide shows exactly where to insert each plot in the `modelling.ipynb` notebook.

---

## Plot Placement Map

### Section 1: Data Understanding

**Plot 1: Data Overview Visualization**
- **Location**: After Cell 3 (data loading and head display)
- **Type**: 2×2 subplot grid
- **Content**:
  - Dataset shape summary
  - Missing value heatmap
  - Temporal distribution (observations per year)
  - Country coverage (number of countries per year)
- **Code Location**: Insert new cell after `df.head()` output

---

### Section 2: Data Preparation

**Plot 2: Feature Engineering Results**
- **Location**: After Cell 5 (preprocessing and feature engineering)
- **Type**: 2×2 subplot grid
- **Content**:
  - Distribution of tree cover loss (histogram, log scale)
  - Global tree cover loss over time (line plot)
  - Top 10 countries by total loss (horizontal bar chart)
  - Tree cover loss by threshold (box plot, log scale)
- **Code Location**: Already implemented in Cell 5
- **Status**: ✅ Already exists

---

### Section 3: Feature Selection

**Plot 3: Feature Selection Results**
- **Location**: After feature selection code (in Cell 5, after SelectKBest)
- **Type**: Single figure with 2 subplots
- **Content**:
  - Top 20 selected features by F-statistic (horizontal bar chart)
  - Feature count comparison: Before (180) vs. After (100)
- **Code Location**: Add after `print(feature_scores.head(20).to_string(index=False))`

---

**Plot 4: Data Split Visualization**
- **Location**: After train-test split code (in Cell 5)
- **Type**: Single figure with 2 subplots
- **Content**:
  - Temporal distribution showing train/test split (stacked area or bar chart)
  - Sample counts by year (line plot with train/test highlighted)
- **Code Location**: Add after `y_test = y_test.astype(float)`

---

### Section 4: Model Training

**Plot 5: Linear Regression Performance**
- **Location**: After Cell 9 (Linear Regression training)
- **Type**: 1×2 subplot grid
- **Content**:
  - Actual vs. Predicted scatter plot (test set)
  - Residual plot (residuals vs. predicted values)
- **Code Location**: Already implemented in Cell 9
- **Status**: ✅ Already exists

---

**Plot 6: Regularized Models Comparison**
- **Location**: After Ridge/Lasso training (in Cell 9, after Lasso evaluation)
- **Type**: 2×3 subplot grid
- **Content**:
  - Row 1: Actual vs. Predicted for Linear, Ridge, Lasso
  - Row 2: Residual plots for Linear, Ridge, Lasso
  - Additional: Coefficient magnitude comparison (Ridge vs. Lasso)
- **Code Location**: Already implemented in Cell 9
- **Status**: ✅ Already exists (2×3 grid)

---

**Plot 7: Random Forest Performance**
- **Location**: After Cell 15 (Random Forest hyperparameter tuning)
- **Type**: 1×2 subplot grid + separate feature importance plot
- **Content**:
  - Actual vs. Predicted scatter plot
  - Residual plot
  - Feature importance plot (top 15 features, horizontal bar chart)
- **Code Location**: Already implemented in Cell 15
- **Status**: ✅ Already exists

---

**Plot 8: XGBoost Performance**
- **Location**: After Cell 16 (XGBoost hyperparameter tuning)
- **Type**: 1×2 subplot grid
- **Content**:
  - Actual vs. Predicted scatter plot
  - Residual plot
- **Code Location**: Already implemented in Cell 16
- **Status**: ✅ Already exists

---

### Section 5: Model Evaluation

**Plot 9: Comprehensive Model Comparison**
- **Location**: After Cell 20 (model comparison code)
- **Type**: 2×2 subplot grid
- **Content**:
  - Top-left: RMSE comparison (bar chart)
  - Top-right: MAE comparison (bar chart)
  - Bottom-left: R² comparison (bar chart, ylim=[0,1])
  - Bottom-right: MASE comparison (bar chart with MASE=1.0 reference line)
- **Code Location**: Already implemented in Cell 20
- **Status**: ✅ Already exists (2×2 grid)

---

**Plot 10: Overfitting Analysis**
- **Location**: After Cell 17 (diagnostic analysis, overfitting check)
- **Type**: Single figure
- **Content**:
  - Train vs. Test R² comparison (grouped bar chart)
  - Gap visualization (difference between train and test)
- **Code Location**: Already implemented in Cell 17
- **Status**: ✅ Already exists

---

**Plot 11: Feature Importance**
- **Location**: After feature importance code (in Cell 17)
- **Type**: Single figure
- **Content**:
  - Top 15 feature importance (horizontal bar chart)
  - Color-coded by feature category
- **Code Location**: Already implemented in Cell 17
- **Status**: ✅ Already exists

---

**Plot 12: Combined Predictions Comparison**
- **Location**: After Cell 20 (model comparison visualization)
- **Type**: Single large scatter plot
- **Content**:
  - All models overlaid on same scatter plot
  - Different colors for each model
  - Perfect prediction line (y=x)
  - Sample of 500 points for clarity
- **Code Location**: Already implemented in Cell 20
- **Status**: ✅ Already exists

---

### Section 6: Forecasts

**Plot 13: Forecast Visualizations**
- **Location**: After Cell 31 (comprehensive forecast generation)
- **Type**: 2×2 subplot grid
- **Content**:
  - Top-left: Global forecast trend (2025-2035, line plot with markers)
  - Top-right: Top 10 countries by forecasted loss (horizontal bar chart)
  - Bottom-left: Forecast trajectories for top 5 countries (multi-line plot)
  - Bottom-right: Distribution of forecasted values (histogram, log scale)
- **Code Location**: Already implemented in Cell 31
- **Status**: ✅ Already exists

---

## Summary: Plot Status

| Plot # | Description | Status | Cell Location |
|--------|-------------|--------|---------------|
| 1 | Data Overview | ⚠️ Needs addition | After Cell 3 |
| 2 | Feature Engineering | ✅ Exists | Cell 5 |
| 3 | Feature Selection | ⚠️ Needs addition | After feature selection in Cell 5 |
| 4 | Data Split | ⚠️ Needs addition | After split in Cell 5 |
| 5 | Linear Regression | ✅ Exists | Cell 9 |
| 6 | Regularized Models | ✅ Exists | Cell 9 |
| 7 | Random Forest | ✅ Exists | Cell 15 |
| 8 | XGBoost | ✅ Exists | Cell 16 |
| 9 | Model Comparison | ✅ Exists | Cell 20 |
| 10 | Overfitting Analysis | ✅ Exists | Cell 17 |
| 11 | Feature Importance | ✅ Exists | Cell 17 |
| 12 | Combined Predictions | ✅ Exists | Cell 20 |
| 13 | Forecasts | ✅ Exists | Cell 31 |

---

## Code Templates for Missing Plots

### Plot 1: Data Overview (Add after Cell 3)

```python
# Data Overview Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Dataset summary
axes[0, 0].text(0.1, 0.5, f'Rows: {df.shape[0]:,}\nColumns: {df.shape[1]}', 
                fontsize=14, verticalalignment='center')
axes[0, 0].set_title('Dataset Summary', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# 2. Missing value heatmap
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values
    }).sort_values('Missing_Count', ascending=False).head(10)
    axes[0, 1].barh(missing_df['Column'], missing_df['Missing_Count'])
    axes[0, 1].set_xlabel('Missing Count', fontsize=12)
    axes[0, 1].set_title('Top 10 Columns with Missing Values', fontsize=14, fontweight='bold')
else:
    axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', fontsize=14)
    axes[0, 1].set_title('Missing Values Check', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

# 3. Temporal distribution
if 'year' in df.columns:
    yearly_counts = df['year'].value_counts().sort_index()
    axes[1, 0].bar(yearly_counts.index, yearly_counts.values, alpha=0.7)
    axes[1, 0].set_xlabel('Year', fontsize=12)
    axes[1, 0].set_ylabel('Number of Observations', fontsize=12)
    axes[1, 0].set_title('Observations per Year', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

# 4. Country coverage
if 'country' in df.columns:
    country_counts = df.groupby('year')['country'].nunique()
    axes[1, 1].plot(country_counts.index, country_counts.values, marker='o', linewidth=2)
    axes[1, 1].set_xlabel('Year', fontsize=12)
    axes[1, 1].set_ylabel('Number of Countries', fontsize=12)
    axes[1, 1].set_title('Country Coverage Over Time', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Plot 3: Feature Selection (Add in Cell 5 after feature selection)

```python
# Feature Selection Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 20 features
top_20 = feature_scores.head(20)
axes[0].barh(range(len(top_20)), top_20['score'].values, color='steelblue')
axes[0].set_yticks(range(len(top_20)))
axes[0].set_yticklabels(top_20['feature'].values, fontsize=9)
axes[0].set_xlabel('F-statistic Score', fontsize=12)
axes[0].set_title('Top 20 Selected Features', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Before/After comparison
comparison_data = {'Before': 180, 'After': len(selected_features)}
axes[1].bar(comparison_data.keys(), comparison_data.values(), 
            color=['coral', 'steelblue'], alpha=0.7)
axes[1].set_ylabel('Number of Features', fontsize=12)
axes[1].set_title('Feature Count: Before vs. After Selection', fontsize=14, fontweight='bold')
for i, (k, v) in enumerate(comparison_data.items()):
    axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### Plot 4: Data Split (Add in Cell 5 after train-test split)

```python
# Data Split Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Temporal split visualization
if 'year' in df_enc.columns:
    year_counts = df_enc['year'].value_counts().sort_index()
    train_years = [y for y in year_counts.index if y <= 2019]
    test_years = [y for y in year_counts.index if y >= 2020]
    
    axes[0].bar(train_years, [year_counts[y] for y in train_years], 
                label='Training Set', color='steelblue', alpha=0.7)
    axes[0].bar(test_years, [year_counts[y] for y in test_years], 
                label='Test Set', color='coral', alpha=0.7)
    axes[0].axvline(x=2019.5, color='red', linestyle='--', linewidth=2, label='Split Point')
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Train-Test Split by Year', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

# Sample counts
split_data = {
    'Training (2001-2019)': len(X_train),
    'Test (2020-2024)': len(X_test)
}
axes[1].bar(split_data.keys(), split_data.values(), 
            color=['steelblue', 'coral'], alpha=0.7)
axes[1].set_ylabel('Number of Samples', fontsize=12)
axes[1].set_title('Dataset Split Summary', fontsize=14, fontweight='bold')
for i, (k, v) in enumerate(split_data.items()):
    axes[1].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold', fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

---

## Notes for Report Integration

1. **Figure Numbering**: Use consistent numbering (Figure 1, Figure 2, etc.)
2. **Captions**: Each plot should have a descriptive caption
3. **Resolution**: Save plots at 300 DPI for publication quality
4. **Format**: Use PNG or PDF format for reports
5. **References**: Reference plots in text (e.g., "As shown in Figure 1...")

---

**Last Updated**: [Date]  
**Notebook Version**: Current

