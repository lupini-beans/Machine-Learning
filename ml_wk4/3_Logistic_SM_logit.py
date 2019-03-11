import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

#fp = open(“binary.csv”, “r”)

# read the data in
df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")

# take a look at the dataset
print (df.head())

# rename the 'rank' column because there is also a DataFrame method called 'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]
print (df.columns)
# array([admit, gre, gpa, prestige], dtype=object)

# summarize the data
print (df.describe())
#             admit         gre         gpa   prestige
# count  400.000000  400.000000  400.000000  400.00000
# mean     0.317500  587.700000    3.389900    2.48500
# std      0.466087  115.516536    0.380567    0.94446
# min      0.000000  220.000000    2.260000    1.00000
# 25%      0.000000  520.000000    3.130000    2.00000
# 50%      0.000000  580.000000    3.395000    2.00000
# 75%      1.000000  660.000000    3.670000    3.00000
# max      1.000000  800.000000    4.000000    4.00000

# take a look at the standard deviation of each column
print (df.std())
# admit      0.466087
# gre      115.516536
# gpa        0.380567
# prestige   0.944460

# frequency table cutting presitge and whether or not someone was admitted
print (pd.crosstab(df['admit'], df['prestige'], rownames=['admit']))
# prestige   1   2   3   4
# admit                   
# 0         28  97  93  55
# 1         33  54  28  12

# plot all of the columns
df.hist()
pl.show()

# dummify rank
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
print (dummy_ranks.head())
#    prestige_1  prestige_2  prestige_3  prestige_4
# 0           0           0           1           0
# 1           0           0           1           0
# 2           1           0           0           0
# 3           0           0           0           1
# 4           0           0           0           1

# create a clean data frame for the regression
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
print (data.head())
#    admit  gre   gpa  prestige_2  prestige_3  prestige_4
# 0      0  380  3.61           0           1           0
# 1      1  660  3.67           0           1           0
# 2      1  800  4.00           0           0           0
# 3      1  640  3.19           0           0           1
# 4      0  520  2.93           0           0           1

# manually add the intercept
data['intercept'] = 1.0

train_cols = data.columns[1:]
# Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)

logit = sm.Logit(data['admit'], data[train_cols])

# fit the model
result = logit.fit()

# cool enough to deserve it's own gist
print (result.summary())

# look at the confidence interval of each coeffecient
print (result.conf_int())
#                    0         1
# gre         0.000120  0.004409
# gpa         0.153684  1.454391
# prestige_2 -1.295751 -0.055135
# prestige_3 -2.016992 -0.663416
# prestige_4 -2.370399 -0.732529
# intercept  -6.224242 -1.755716

# odds ratios only
print (np.exp(result.params))
# gre           1.002267
# gpa           2.234545
# prestige_2    0.508931
# prestige_3    0.261792
# prestige_4    0.211938
# intercept     0.018500

# odds ratios and 95% CI
#params = result.params
#conf = result.conf_int()
#conf['OR'] = params
#conf.columns = ['2.5%', '97.5%', 'OR']
#print (np.exp(conf))
##                   2.5%     97.5%        OR
## gre           1.000120  1.004418  1.002267
## gpa           1.166122  4.281877  2.234545
## prestige_2    0.273692  0.946358  0.508931
## prestige_3    0.133055  0.515089  0.261792
## prestige_4    0.093443  0.480692  0.211938
## intercept     0.001981  0.172783  0.018500