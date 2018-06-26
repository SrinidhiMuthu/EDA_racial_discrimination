
# coding: utf-8

# # Examining Racial Discrimination in the US Job Market
# 
# ### Background
# Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.
# 
# ### Data
# In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.
# 
# Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.

# <div class="span5 alert alert-info">
# ### Exercises
# You will perform a statistical analysis to establish whether race has a significant impact on the rate of callbacks for resumes.
# 
# Answer the following questions **in this notebook below and submit to your Github account**. 
# 
#    1. What test is appropriate for this problem? Does CLT apply?
#    2. What are the null and alternate hypotheses?
#    3. Compute margin of error, confidence interval, and p-value. Try using both the bootstrapping and the frequentist statistical approaches.
#    4. Write a story describing the statistical significance in the context or the original problem.
#    5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?
# 
# You can include written notes in notebook cells using Markdown: 
#    - In the control panel at the top, choose Cell > Cell Type > Markdown
#    - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# 
# 
# #### Resources
# + Experiment information and data source: http://www.povertyactionlab.org/evaluation/discrimination-job-market-united-states
# + Scipy statistical methods: http://docs.scipy.org/doc/scipy/reference/stats.html 
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# + Formulas for the Bernoulli distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution
# </div>
# ****

# In[10]:

import pandas as pd
import numpy as np
from scipy import stats


# In[11]:

data = pd.io.stata.read_stata('data/us_job_market_discrimination.dta')


# In[12]:

# number of callbacks for black-sounding names
sum(data[data.race=='w'].call)


# In[13]:

data.head()


# <div class="span5 alert alert-success">
# <p>Your answers to Q1 and Q2 here</p>
# </div>

# In[14]:

w = data[data.race=='w']
b = data[data.race=='b']


# Quetion1:
# The test used to compare two categorical vaiables - hypothesis test(Comparing 2 sample proportions.)
# 
# The Central Limit Theorem applies even to binomial populations like this provided that the minimum of np and n(1-p) is at least 5, where "n" refers to the sample size, and "p" is the probability of "success" on any given trial. In this case, we will take samples of n=20 with replacement, so min(np, n(1-p)) = min(20(0.3), 20(0.7)) = min(6, 14) = 6. Therefore, the criterion is met.
# Yes CLT applies as the binomial disturbution is normal for large samples.
# 
# Null hypothesis : pw=pb
# 
# Alternative hypothesis : pw!=pb

# DOES CLT APLY?
# Sample observations must be independent
# 
# 1.random sample assignment
# 
# 2.if sampling wihtout replacement n<10% of population
# 
# Sample size / skew:
# np>=10 and n(1-p)>=10

# <div class="span5 alert alert-success">
# <p> Your answers to Q4 and Q5 here </p>
# </div>

# In[15]:


#Number of resumes by race 
data_white=(data[data.race=='w'])
data_black=(data[data.race=='b'])

#Number of CV per race:
w_resume=len(data_white.race)
b_resume=len(data_black.race)

#Number of calls per race:
w_calls=sum(data[data.race=='w'].call)
b_calls=sum(data[data.race=='b'].call)
#Sample proportions = 

w_sample_p = w_calls / w_resume
b_sample_p = b_calls / b_resume

print(w_sample_p,b_sample_p)


# In[16]:

p_pooled = round(((w_calls+b_calls)/(w_resume+b_resume)),2)
np_w=w_resume*p_pooled
n1p_w=w_resume*(1-p_pooled)

# and for n2p_pool >=10, n2(1-p_pool) >=10:
np_b=b_resume*p_pooled
n1p_b=b_resume*(1-p_pooled)

print(np_w,n1p_w, np_b,n1p_b)


# np>10 and n(1-p)>=10.Hence it satisfies the CLT criterion.

# # Your solution to Q3 here
# 3.Compute margin of error, confidence interval, and p-value. Try using both the bootstrapping and the frequentist statistical approaches

# m=z* - sqrt(p(1-p)/n)
# z* = pw-pb/(sqrt(pb(1-pb)/n))
# To calculate Standard deviaton we assume null hypothesis is true.
# ME = sqrt(pw(1-pw)/n1 + pb(1-pb)/n2)

# In[18]:

#Margin Error: 
import numpy as np
SE= round(np.sqrt(((p_pooled*(1-p_pooled))/w_resume) + ((p_pooled*(1-p_pooled))/b_resume)),2)
Zvalue= round((w_sample_p - b_sample_p)/SE,2)
ME = round((Zvalue * SE),2)
print("Margin of Error",ME)


# Confidence inetrval = To find a confidence interval for a proportion, estimate the standard deviation sp from the data by replacing the unknown value p with the sample proportion , giving the standard error SE

# In[19]:

UL = (w_sample_p-b_sample_p)+ME
LL = (w_sample_p-b_sample_p)-ME
print("Confidence Interval", LL, UL)


# P-Value
#  The test statistic z is used to compute the P-value for the standard normal distribution, the probability that a value at least as extreme as the test statistic would be observed under the null hypothesis. Given the null hypothesis that the population proportion p is equal to a given value p0, the P-values for testing H0 against each of the possible alternative hypotheses are: 
# P(Z > z) for Ha: p > p0 
# P(Z < z) for Ha: p < p0	
# 2P(Z>|z|) for Ha: p p0.

# In[20]:

#P(Z>ZValue) = P(Z>3.2) 
import scipy.stats as st
print("P-value: ", round(st.norm.sf(Zvalue),3))


# 4.Write a story describing the statistical significance in the context or the original problem.
# 5.Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?
# 

# Since P-value <0.05 we can reject the null hypothesis ,i.e pw = pb and this means that there is significance impact on race parameter while picking resumes and calling back.  

# In[ ]:



