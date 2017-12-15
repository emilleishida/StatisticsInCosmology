# Statistics In Cosmology  
developed by [Emille E. O. Ishida](www.emilleishida.com)

[11th TRR33 Winter school](http://darkuniverse.uni-hd.de/view/Main/WinterSchool17)  
*Passo del Tornale, Italy - 10-16 December 2017*



## Day 1 - Introduction

*"Begin at the beginning," the King said, very gravely, "and go on till you come to the end: then stop".*  
L. Carroll, Alice in Wonderland (1865)

> Discussions on Frequentists and Bayesian statistics  
    
[Slides](https://github.com/emilleishida/StatisticsInCosmology/tree/master/I_Frequentist_vs_Bayesian/StatisticsInCosmology_day1.pdf)  
[References](https://github.com/emilleishida/StatisticsInCosmology/tree/master/I_Frequentist_vs_Bayesian/references)  

## Day 2 - Parameter Inference  

*“Everything we care about lies somewhere in the middle, where pattern and randomness interlace.”*  
J. Gleick, The Information: A History, a Theory, a Flood (2011)  

> Maximum Likelihood estimators  
> Monte Carlo Markov Chain methods  

[Slides](https://github.com/emilleishida/StatisticsInCosmology/tree/master/II_Parameter_Inference/StatisticsInCosmology_day2.pdf)  
[Example - my_mcmc](https://github.com/emilleishida/StatisticsInCosmology/tree/master/II_Parameter_Inference/my_first_MCMC.py)  
[Example - Stan](https://github.com/astrobayes/BMAD/blob/master/chapter_4/code_4.3.py)  
[References](https://github.com/emilleishida/StatisticsInCosmology/tree/master/II_Parameter_Inference/references)


## Day 3 - Model Selection

*“Crying is all right in its way while it lasts. But you have to stop sooner or later, and then you still have to decide what to do.”*  
C.S. Lewis - The Silver Chair (Chronicles of Narnia, #4, 1953)

> Information criterias  
> Cross-validation  
> Bayesian evidence  
 
[Slides](https://github.com/emilleishida/StatisticsInCosmology/tree/master/III_Model_Selection/StatisticsInCosmology_day3.pdf)  
[References](https://github.com/emilleishida/StatisticsInCosmology/tree/master/III_Model_Selection)

## Day 4 - Approximate Bayesian Computation

*“Essentially, all models are wrong, but some are useful.”*  
 G. E. P. Box and N. R. Draper, Response Surfaces, Mixtures, and Ridge Analyses (2007)

> Importance Sampling  
> Likelihood-free parameter inference  
 
[Slides](https://github.com/emilleishida/StatisticsInCosmology/tree/master/IV_ABC/StatisticsInCosmology_day4.pdf)  
[Examples](https://github.com/emilleishida/StatisticsInCosmology/tree/master/IV_ABC)  
[References](https://github.com/emilleishida/StatisticsInCosmology/tree/master/IV_ABC/references)


## Day 5 - Working group activities

#### Bayesian and Frequentist statistics
     
> Q1: One of the main arguments against Bayesian analysis is that the theory does not provide a recipe on how to calculate priors. As a consequence, there is an inherented subjectivity in the method which many people see as problematic. In your view, does frequentist approach carry similar subjectivity?

#### Application

> Q2: Suppose that 60% of the stellar systems in a galaxy far,  far away host an Earth-like planet. Let us also assume that every system that hosts an Earth-like planet also hosts a Jupiter-like planet, while only half of the systems who fail to host Earth-like planets host Jupiter-like planets. Now, let us suppose that we observe a system with a Jupiter-like planet. What is the probability that this system also hosts an Earth-like planet?*   

*This example was taken from [Bayesian Models for Astrophysical Data - using R, JAGS, Python and Stan](www.bayesianmodelsforastrophysicaldata.com), by Hilbe, de Souza and Ishida, 2017, Cambridge University Press  

#### Errors in Bayesian modelling

> I show [here](https://github.com/emilleishida/StatisticsInCosmology/blob/master/sncosmology.R) I direct implementation of Bayesian inference in the case of supernova cosmology. The code is written in [R](https://www.r-project.org/) using [Stan](http://mc-stan.org/).  
> Q3: what are the assumptions underlying this particular construction of the statistical model? Are they realistic?  
> Q4: What modifications do you suggest if one wishes to take into account the errors in the observed quantities?  

[Normal linear model in Python using Stan](https://github.com/astrobayes/BMAD/blob/master/chapter_4/code_4.3.py)  
[Normal linear model in Python using Stan considering measurement errors](https://github.com/astrobayes/BMAD/blob/master/chapter_4/code_4.11.py)


