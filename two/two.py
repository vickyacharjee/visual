import pandas as pd
from scipy import stats
# Load Titanic dataset
titanic_data = pd.read_csv('train.csv')
# ANOVA Test: Impact of passenger class on fares
anova_result = stats.f_oneway(titanic_data[titanic_data['Pclass'] == 1]['Fare'].dropna(),
titanic_data[titanic_data['Pclass'] == 2]['Fare'].dropna(),
titanic_data[titanic_data['Pclass'] == 3]['Fare'].dropna())
print("\nANOVA Test Result:")
print("F-statistic:", anova_result.statistic)
print("p-value:", anova_result.pvalue)
# Chi-Square Test: Relationship between survival status and passenger class
chi2_table = pd.crosstab(titanic_data['Survived'], titanic_data['Pclass'])
chi2_result = stats.chi2_contingency(chi2_table)
print("\nChi-Square Test Result:")
print("Chi-Square statistic:", chi2_result[0])
print("p-value:", chi2_result[1])