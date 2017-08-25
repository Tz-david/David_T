#数据预处理#
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
credit_df = pd.read_csv( "german.data.txt", delim_whitespace = True, header = None )
credit_df.head()
columns = ['checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
         'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
         'other_debtors', 'residing_since', 'property', 'age',
         'inst_plans', 'housing', 'num_credits',
         'job', 'dependents', 'telephone', 'foreign_worker', 'status']
credit_df.columns = columns
print(credit_df.head())
print(credit_df.info())
credit_df.status = credit_df.status - 1

#数据可视化
import matplotlib.pyplot as plt
import seaborn as sn

#账户信用支付额度分布直方图
#Note: Most of the credit amounts are less than 5000 with some higher credit amounts. The largest amount disbursed is as high as 18000+.
sn.distplot( credit_df.amount, kde = False )
plt.title( "Histogram of Credit Amount Disbursed", fontsize = 15)
plt.ylabel( "Frequency")
credit_df.amount.describe()

#信用支付额度箱线图
#Note: The middle 50% of the population lies between 1300 to 3900.
plt.figure()
sn.boxplot( credit_df.amount, orient = 'v' )
plt.title( "Boxplot of Credit Amount Disbursed", fontsize = 15)

#不同信用状况的箱线图
# Note: Lot of higher credit amounts seem to have been defaulted.
plt.figure()
sn.boxplot( x = 'status', y = 'amount', data = credit_df, orient = 'v' )
plt.title( "Boxplot of Credit Amount Disbursed by Credit Status", fontsize = 15)

#对于不同的信用状况量对比分布图
plt.figure()
sn.distplot( credit_df[credit_df.status == 0].amount, color = 'g', hist = False )
sn.distplot( credit_df[credit_df.status == 1].amount, color = 'r', hist = False )
plt.title( "Distribution plot of Amount comparison for Different Credit Status", fontsize = 10 )
plt.ylabel( "Frequency")

#Note: Amounts higher than 10000 have been mostly defaulted.
plt.figure()
g = sn.FacetGrid(credit_df, col="status", size = 6)
g.map(sn.distplot, "amount", kde = False, bins = 20 )

plt.figure()
d_rate_df = pd.DataFrame( credit_df.status.value_counts( normalize=True ) )
d_rate_df
sn.barplot( x = d_rate_df.index, y = d_rate_df.status )

plt.figure()
credit_df.amount.describe()
amount_desc = credit_df.amount.describe()
outliers = amount_desc['75%'] + 1.5 * ( amount_desc['75%'] - amount_desc['25%'] )
d_rate_outliers_df = pd.DataFrame( credit_df[credit_df.amount >outliers ].status.value_counts( normalize = True ) )
sn.barplot( x = d_rate_outliers_df.index, y = d_rate_outliers_df.status )

plt.figure()
extreme_outliers = amount_desc['75%'] + 3 * ( amount_desc['75%'] - amount_desc['25%'] )
extreme_outliers_df = pd.DataFrame( credit_df[credit_df.amount >extreme_outliers ].status.value_counts( normalize = True ) )
sn.barplot( x = extreme_outliers_df.index, y = extreme_outliers_df.status )

plt.figure()
credit_df.inst_rate.unique()
rate_count = credit_df[['inst_rate', 'status']].groupby(['inst_rate', 'status']).size().reset_index()
rate_count.columns = ['inst_rate', 'status', 'count']
g=sn.factorplot(x="inst_rate", y='count', hue="status", data=rate_count,size=6, kind="bar", palette="muted")

#信用证金额的信用状况发放箱线图
plt.figure()
sn.boxplot( x = 'inst_rate', y = 'amount', hue = 'status', data = credit_df, orient = 'v' )
plt.title( "Boxplot of Credit Amount Disbursed by Credit Status", fontsize = 12)

plt.figure()
sn.lmplot( x = 'inst_rate', y = 'amount', data = credit_df )

#Distribution plot of Amount Disbured for inst_rate = 1
plt.figure()
credit_inst_rate_1_df = credit_df[ credit_df.inst_rate == 1 ]
sn.distplot( credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount, color = 'g', hist = False )
sn.distplot( credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1].amount, color = 'r', hist = False )
sn.plt.axvline( x = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount.mean(), color = 'g' )
sn.plt.axvline( x = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount.mean(), color = 'r' )
plt.title( "Distribution plot of Amount Disbured for inst_rate = 1", fontsize = 10 )
plt.ylabel( "Frequency")

stats.ttest_ind( credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount,credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount)
credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount.mean()
credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount.mean()
stats.ttest_1samp( credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount, 4700 )
stats.norm.cdf( 4700,loc = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount.mean())
                   # scale = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount.bstd())
credit_inst_rate_1_df[ (credit_inst_rate_1_df.status == 0)& (credit_inst_rate_1_df.amount > 4700)].amount.sum()

plt.figure()
sn.barplot( x = 'checkin_acc', y = 'amount', hue = 'status', data = credit_df )
plt.title( "Average credit amount by different checkin account holders")
plt.figtext(1, 0.5,"""A11 : < 0 DM \n A12 : 0 <= ... < 200 DM \n A13 : >= 200 DM \n A14 : no checking account 
""", wrap=True, horizontalalignment='left', fontsize=12)

plt.figure()
sn.barplot( x = 'checkin_acc',
         y = 'amount',
         hue = 'status',
         data = credit_df,
         estimator = sum )
plt.title( "Total credit amount by different checkin account holders")
plt.figtext(1, 0.5,"""A11 : < 0 DM \n A12 : 0 <= ... < 200 DM \n A13 : >= 200 DM \n A14 : no checking account 
""", wrap=True, horizontalalignment='left', fontsize=12)

plt.figure()
sn.countplot( y = 'checkin_acc', hue = 'status', data = credit_df )

plt.figure()
figure_text = """A30 : no credits taken/ all credits paid back duly \n
A31 : all credits at this bank paid back duly \n
A32 : existing credits paid back duly till now \n
A33 : delay in paying off in the past \n
A34 : critical account/ other credits existing (not at this bank) """
sn.barplot( x = 'credit_history', y = 'amount', hue = 'status', data = credit_df, estimator = sum )
plt.figtext(1, 0.5,figure_text, wrap=True, horizontalalignment='left', fontsize=12)

plt.figure()
sn.countplot( y = 'credit_history', hue = 'status', data = credit_df )
plt.figtext(1, 0.5,figure_text, wrap=True, horizontalalignment='left', fontsize=12)

plt.figure()
purpose_text = '''
A40 : car (new) \n
A41 : car (used) \n
A42 : furniture/equipment \n
A43 : radio/television \n
A44 : domestic appliances \n 
A45 : repairs \n
A46 : education \n
A47 : (vacation - does not exist?) \n
A48 : retraining \n
A49 : business \n
A410 : others '''
sn.barplot( x = 'purpose', y = 'amount', hue = 'status', data = credit_df )
plt.figtext(1, 0.3,purpose_text, wrap=True, horizontalalignment='left', fontsize=8)

plt.figure()
sn.barplot( x = 'purpose', y = 'amount', hue = 'status', data = credit_df, estimator = sum )
plt.figtext(1, 0.3,purpose_text, wrap=True, horizontalalignment='left', fontsize=8)

plt.figure()
credit_used_car_df = credit_df[ credit_df.purpose == 'A41' ]
sn.distplot( credit_used_car_df[credit_used_car_df.status == 0 ].amount, color = 'g', hist = False )
sn.distplot( credit_used_car_df[credit_used_car_df.status == 1].amount, color = 'r', hist = False )
sn.plt.axvline( x = credit_used_car_df[credit_used_car_df.status == 0 ].amount.mean(), color = 'g' )
sn.plt.axvline( x = credit_used_car_df[credit_used_car_df.status == 1 ].amount.mean(), color = 'r' )
plt.title( "Distribution plot of Amount Disbured for Used Car Purchase and Credit Status", fontsize = 10 )
plt.ylabel( "Frequency")

plt.figure()
stats.ttest_ind( credit_used_car_df[credit_used_car_df.status == 0 ].amount,
               credit_used_car_df[credit_used_car_df.status == 1 ].amount)
credit_new_car_df = credit_df[ credit_df.purpose == 'A40' ]
sn.distplot( credit_new_car_df[credit_new_car_df.status == 0 ].amount, color = 'g', hist = False )
sn.distplot( credit_new_car_df[credit_new_car_df.status == 1].amount, color = 'r', hist = False )
sn.plt.axvline( x = credit_new_car_df[credit_new_car_df.status == 0 ].amount.mean(), color = 'g' )
sn.plt.axvline( x = credit_new_car_df[credit_new_car_df.status == 1 ].amount.mean(), color = 'r' )
plt.title( "Distribution plot of Amount Disbured for Used Car Purchase and Credit Status", fontsize = 10 )
plt.ylabel( "Frequency")

plt.figure()
stats.ttest_ind( credit_new_car_df[credit_new_car_df.status == 0 ].amount,
               credit_new_car_df[credit_new_car_df.status == 1 ].amount)
credit_appliances_df = credit_df[ credit_df.purpose == 'A44' ]
sn.distplot( credit_appliances_df[credit_appliances_df.status == 0 ].amount, color = 'g', hist = False )
sn.distplot( credit_appliances_df[credit_appliances_df.status == 1].amount, color = 'r', hist = False )
sn.plt.axvline( x = credit_appliances_df[credit_appliances_df.status == 0 ].amount.mean(), color = 'g' )
sn.plt.axvline( x = credit_appliances_df[credit_appliances_df.status == 1 ].amount.mean(), color = 'r' )
plt.title( "Distribution plot of Amount Disbured for Used Car Purchase and Credit Status", fontsize = 10 )
plt.ylabel( "Frequency")

plt.figure()
stats.ttest_ind( credit_appliances_df[credit_appliances_df.status == 0 ].amount,
               credit_appliances_df[credit_appliances_df.status == 1 ].amount)
sn.countplot( x = 'purpose', hue = 'status', data = credit_df )
plt.figtext(1, 0.3,purpose_text, wrap=True, horizontalalignment='left', fontsize=8)

plt.figure()
sn.lmplot( x = 'duration', y = 'amount', fit_reg = False, data = credit_df )

plt.figure()
sn.lmplot( x = 'duration', y = 'amount', fit_reg = True, data = credit_df )

sn.lmplot( x = 'duration', y = 'amount', hue = 'status', fit_reg = False, data = credit_df )

sn.lmplot( x = 'age', y = 'amount', hue = 'status', fit_reg = True, data = credit_df )

plt.figure()
credit_df.select_dtypes(include = ['float64', 'int64'])[0:5]
sn.pairplot( credit_df.select_dtypes(include = ['float64', 'int64']).iloc[:, :-1] )
plt.show()
