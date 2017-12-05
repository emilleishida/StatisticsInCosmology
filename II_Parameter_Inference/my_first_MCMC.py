"""
Created by Emille Ishida in 12 November 2017. 

Example of simple MCMC algorithm.

Goal is to determine the mean of the original distribution.

Here we are assuming that:
	- we know that the correct model is Gaussian
 	- the uncertainty is known and constant

The first part of this script was taken from 
http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
"""



import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, bernoulli, uniform

# fix random seed
np.random.seed(123)

# define fiducial parameters
true_mu = 2.0
std = 0.25
n = 50

# generate observations = true + gaussian noise
data = norm.rvs(loc=true_mu, scale=std, size=n)

# plot data
plt.figure()
plt.title('Data')
plt.hist(data)
plt.show()

# set starting point
mu_current = 10.0

# proposal size of step for random walk
proposal_width = 1.0

# number of samples to accept
n_samples = 500

# determine prior parameters
# here we consider  a gaussian prior
mu_prior_mean = 5 
mu_prior_std = 1.5

# store chain
chain = []

while len(chain) < n_samples:

    print str(len(chain)) + '  samples'

    # propose a new point
    mu_proposal = norm(mu_current, proposal_width).rvs()

    # calculate likelihood in current and proposed point
    likelihood_current = norm(mu_current, std).pdf(data).prod()
    likelihood_proposal = norm(mu_proposal, std).pdf(data).prod()

    # Compute prior probability of current and proposed mu        
    prior_current = norm(mu_prior_mean, mu_prior_std).pdf(mu_current)
    prior_proposal = norm(mu_prior_mean, mu_prior_std).pdf(mu_proposal)

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
plt.title('Chain convergence')
plt.plot(range(n_samples), chain)
plt.xlabel('sample')
plt.ylabel('mu')
plt.show()

# from the above plot you can see that the first 200 iterations where very
# far from the convergence point, so we can consider them burn-in

burn_in = 200

plt.figure()
plt.title('Chain after burn-in/warm-up')
plt.plot(range(n_samples - burn_in), chain[burn_in:])
plt.plot([0, n_samples - 200], [true_mu, true_mu], color='black', ls='--', lw=2.0)
plt.xlabel('sample')
plt.ylabel('mu')
plt.show()

# from this we can estimate
mean_posterior, std_posterior = norm.fit(chain[burn_in:])


x_axis = np.arange(min(chain[burn_in:]), max(chain[burn_in:]), 0.001)
y_axis = norm.pdf(x_axis, loc=mean_posterior, scale=std_posterior)

plt.figure()
plt.hist(chain[burn_in:], normed=True, alpha=0.5)
plt.plot(x_axis, y_axis, color='red', lw=2.0)
plt.xlabel('mu')
plt.ylabel('PDF')
plt.show()


##############################################################
##############################################################

# We can also extend this example to 2 dimensions
# if now our mu is described by a linear model...

# define fiducial parameters
true_a = 2.0
true_b = 1.0
std = 0.1
n = 1000

# generate exploratory random variable within [-1, 1]
x = uniform.rvs(loc=-1, scale=2, size=n)

# generate observations = true + gaussian noise
true_mu = true_a * x + true_b
y = norm.rvs(loc=true_mu, scale=std)

# plot data
plt.figure()
plt.title('Data')
plt.scatter(x, y, s=20)
plt.show()

# set starting point
a_current = 1.0
b_current = 0.5
mu_current = a_current * x + b_current

# proposal size of step for random walk
proposal_width = 0.25

# number of samples to accept
n_samples = 50000

# determine prior parameters
# here we consider gaussian priors
a_prior_mean = 1 
a_prior_std = 0.5

b_prior_mean = 0 
b_prior_std = 0.5

# store chain
chain = []

while len(chain) < n_samples:

    print str(len(chain)) + '  samples'

    # propose a new point
    a_proposal = norm(a_current, proposal_width).rvs()
    b_proposal = norm(b_current, proposal_width).rvs()
    mu_proposal = norm.rvs(loc=a_proposal * x + b_proposal, scale=std)

    # calculate likelihood in current and proposed point
    likelihood_current = norm(mu_current, std).pdf(y).prod()
    likelihood_proposal = norm(mu_proposal, std).pdf(y).prod()

    # Compute prior probability of current and proposed mu        
    prior_current = norm(a_prior_mean, a_prior_std).pdf(a_current) * \
                    norm(b_prior_mean, b_prior_std).pdf(b_current)
    prior_proposal = norm(a_prior_mean, a_prior_std).pdf(a_proposal) * \
                     norm(b_prior_mean, b_prior_std).pdf(b_proposal)

    # Nominator of Bayes formula
    p_current = likelihood_current * prior_current
    p_proposal = likelihood_proposal * prior_proposal

    # acceptance probability
    p_accept = min([1, p_proposal / p_current])

    accept = bernoulli.rvs(p_accept)

    if accept:
        # store new point
        chain.append([a_proposal, b_proposal])

        # Update position
        a_current = a_proposal
        b_current = b_proposal


# convert chain into array
chain = np.array(chain)

plt.figure()
plt.suptitle('Chain convergence')

plt.subplot(1,2,1)
plt.plot(range(n_samples), chain[:,0])
plt.xlabel('sample')
plt.ylabel('a')

plt.subplot(1,2,2)
plt.plot(range(n_samples), chain[:,1])
plt.xlabel('sample')
plt.ylabel('b')

plt.show()

# from the above plot you can see that the first 200 iterations where very
# far from the convergence point, so we can consider them burn-in

burn_in = 200

plt.figure()
plt.title('Chain after burn-in/warm-up')
plt.plot(range(n_samples - burn_in), chain[burn_in:])
plt.plot([0, n_samples - 200], [true_mu, true_mu], color='black', ls='--', lw=2.0)
plt.xlabel('sample')
plt.ylabel('mu')
plt.show()

