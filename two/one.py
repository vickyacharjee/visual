import pandas as pd
from scipy import stats
# Load Titanic dataset
titanic_data = pd.read_csv('train.csv')
# Replace 'train.csv' with your dataset file
# One Sample T-Test: Checking mean age against a hypothetical mean
hypothetical_mean_age = 30
ttest_one_sample = stats.ttest_1samp(titanic_data['Age'].dropna(),
hypothetical_mean_age)
print("One Sample T-Test:")
print("T-statistic:", ttest_one_sample.statistic)
print("p-value:", ttest_one_sample.pvalue)
# Two Independent Samples T-Test: Comparing ages of male and female passengers
male_ages = titanic_data[titanic_data['Sex'] == 'male']['Age'].dropna()
female_ages = titanic_data[titanic_data['Sex'] == 'female']['Age'].dropna()
ttest_two_ind_samples = stats.ttest_ind(male_ages, female_ages)
print("\nTwo Independent Samples T-Test:")
print("T-statistic:", ttest_two_ind_samples.statistic)
print("p-value:", ttest_two_ind_samples.pvalue)
# Paired T-Test: Comparing fares before and after
before_fares = titanic_data['Fare'].dropna()
after_fares = before_fares * 1.2 # Assuming a 20% increase in fares
ttest_paired = stats.ttest_rel(before_fares, after_fares)
print("\nPaired T-Test:")
print("T-statistic:", ttest_paired.statistic)
print("p-value:", ttest_paired.pvalue)