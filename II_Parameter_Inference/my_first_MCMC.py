"""
Created by Emille Ishida in 12 November 2017. 

Example of simple MCMC algorithm.

Goal is to determine the mean of the parent distributiono

The first part of this script was taken from 
http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
"""



import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, bernoulli, uniform

sns.set_style('white')
sns.set_context('talk')

np.random.seed(123)

# define fiducial parameters
trueA = 5
trueB = 2
trueSd = 1
sampleSize = 300

# generate independent x-values
x = uniform.rvs(loc=0, scale=1, size=sampleSize)

# generate response variable
trueMu = trueA * x + trueB

# create dependent values according to ax + b + N(0,sd)
#y <-  trueA * x + trueB + norm.rvs(loc=0, scale=trueSd, size=sampleSize)
# or
y = norm.rvs(loc=trueMu, scale=trueSd)

# plot data
plt.figure()
plt.title('Test Data')
plt.scatter(x, y, s=15)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# set starting point
startvalue = [4, 1.0, 0.1]

# set width of step
proposal_width = 0.5

# number of samples to accept
n_samples = 10000

# determine prior parameters
mu_prior_mu = 0.0
mu_prior_sd = 1.0

# store chain
chain = []

while len(chain) < n_samples:

    print str(len(posterior)) + '  samples'

    # propose a new point
    mu_proposal = norm(mu_current, proposal_width).rvs()

    # calculate likelihood in current and proposed point
    likelihood_current = norm(mu_current, 1).pdf(data).prod()
    likelihood_proposal = norm(mu_proposal, 1).pdf(data).prod()

    # Compute prior probability of current and proposed mu        
    prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
    prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)

    # Nominator of Bayes formula
    p_current = likelihood_current * prior_current
    p_proposal = likelihood_proposal * prior_proposal

    # acceptance probability
    p_accept = min([1, p_proposal / p_current])

    accept = bernoulli.rvs(p_accept)

    if accept:
        # store new point
        chain.append(mu_proposal)

        # Update position
        mu_current = mu_proposal


plt.figure()
plt.subplot(1,2,1)
plt.hist(posterior)

plt.subplot(1,2,2)
plt.plot(range(n_samples), posterior)
plt.show()
