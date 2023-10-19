import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


data = pd.read_excel(r"/Users/micha/STAT2120/VoterDataOmit.xlsx")

#%%
X = data["Median Income By County in 2018 dollars"]
X = sm.add_constant(X)
y = data["VoterTurnout"]

model = sm.OLS(y,X).fit()

#%%
model_output = model.summary()
print(model_output)

#%%
# Verify the confidence interval
tstar = stats.t.ppf(0.975,39)
LB = 0.8888 - tstar*0.182
UB = 0.8888 + tstar*0.182

print("CI for Income = ")
print(LB)
print(UB)

#%%
b0 = round(model.params[0], 2)
b1 = round(model.params[1], 2)

print("The regression equation for Median Income is: y = " + str(b0) + " + " + str(b1) + "x.")
#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


data = pd.read_excel(r"/Users/micha/STAT2120/VoterDataOmit.xlsx")

#%%
X = data["RegisteredVoters"]
X = sm.add_constant(X)
y = data["VoterTurnout"]

model = sm.OLS(y,X).fit()

#%%
# Specify value for prediction
Xnew = pd.DataFrame([[1,15218]])

# Predict for that value
pred_new = model.predict(Xnew)
print(pred_new)

#%%
# Determine the confidence interval and prediction interval 
pred = model.get_prediction(Xnew)
CI_PI = pred.summary_frame()
with pd.option_context('display.max_columns', None):  
    print(CI_PI)


#%%
# Verify the confidence interval
tstar = stats.t.ppf(0.975,39)
LB = 0.2470 - tstar*0.019
UB = 0.2470 + tstar*0.019

print("CI for Registered Voters = ")
print(LB)
print(UB)

#%%
b0 = round(model.params[0], 2)
b1 = round(model.params[1], 2)

print("The regression equation for Registered Voters is: y = " + str(b0) + " + " + str(b1) + "x.")


#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


data = pd.read_excel(r"/Users/micha/STAT2120/VoterDataOmit.xlsx")

#%%
X = data["Bachelor's Degree or High Percentage By County (%)"]
X = sm.add_constant(X)
y = data["VoterTurnout"]

model = sm.OLS(y,X).fit()

#%%
model_output = model.summary()
print(model_output)

#%%
# Specify value for prediction
Xnew = pd.DataFrame([[1, 9000]])

# Predict for that value
pred_new = model.predict(Xnew)
print(pred_new)

#%%
# Determine the confidence interval and prediction interval 
pred = model.get_prediction(Xnew)
CI_PI = pred.summary_frame()
with pd.option_context('display.max_columns', None):  
    print(CI_PI)

#%%
# Verify the confidence interval
tstar = stats.t.ppf(0.975,39)
LB = 1232.8508 - tstar*265.145
UB = 1232.8508 + tstar*265.145

print("CI for Bachelor's Degree = ")
print(LB)
print(UB)


#%%
b0 = round(model.params[0], 2)
b1 = round(model.params[1], 2)

print("The regression equation for Bachelor's Degree is: y = " + str(b0) + " + " + str(b1) + "x.")
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_excel(r"/Users/micha/STAT2120/VoterData.xlsx")

#%%
sns.regplot(x="Median Income By County in 2018 dollars", y="VoterTurnout", data=data, ci=0)
plt.title("The Association Between Voter Turnout and Median Income By County in 2018 Dollars")
plt.ylabel("Voter Turnout")
plt.xlabel("Median Income By County in 2018 Dollars")
plt.xlim(0, 120000)
plt.ylim(0, 130000)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_excel(r"/Users/micha/STAT2120/VoterData.xlsx")

#%%
sns.regplot(x="RegisteredVoters", y="VoterTurnout", data=data, ci=0)
plt.title("The Association Between Voter Turnout and Registered Voters")
plt.ylabel("Voter Turnout")
plt.xlabel("Registered Voters")
plt.xlim(0, 180000)
plt.ylim(0, 130000)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_excel(r"/Users/micha/STAT2120/VoterData.xlsx")

#%%
sns.regplot(x="Bachelor's Degree or High Percentage By County (%)", y="VoterTurnout", data=data, ci=0)
plt.title("The Association Between Voter Turnout and Bachelor's Degree or High Percentage By County (%)")
plt.ylabel("Voter Turnout")
plt.xlabel("Bachelor's Degree or High Percentage By County (%)")
plt.xlim(0, 80)
plt.ylim(0, 130000)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_excel(r"/Users/micha/STAT2120/VoterDataComplete.xlsx")

#%%
X = data["RegisteredVoters"]
X = sm.add_constant(X)
y = data["VoterTurnout"]

model = sm.OLS(y,X).fit()

#%%
model_output = model.summary()
print(model_output)

#%%
b0 = round(model.params[0], 2)
b1 = round(model.params[1], 2)

print("The regression equation is: y = " + str(b0) + " + " + str(b1) + "x.")

