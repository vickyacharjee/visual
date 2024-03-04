import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, binom, expon
# Normal Distribution - Quality Control example
# Generating and plotting a normal distribution
mean = 50
std_dev = 10
samples = np.random.normal(mean, std_dev, 1000)
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=30, density=True, alpha=0.6, color='blue')
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
plt.plot(x, norm.pdf(x, mean, std_dev), 'r-', lw=2, label='Normal Distribution')
plt.title('Normal Distribution Example (Quality Control)')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
# Poisson Distribution - Service and Arrival Rates example
# Calculating the probability of a certain number of events occurring in a time frame
# Exponential Distribution - Reliability Analysis example
# Simulating and plotting an exponential distribution
exp_samples = np.random.exponential(scale=2, size=1000)
plt.figure(figsize=(8, 6))
plt.hist(exp_samples, bins=30, density=True, alpha=0.6, color='green')
x_exp = np.linspace(0, 10, 100)
plt.plot(x_exp, expon.pdf(x_exp, scale=2), 'r-', lw=2, label='Exponential Distribution')
plt.title('Exponential Distribution Example (Reliability Analysis)')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()