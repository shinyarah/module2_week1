Loading...
Moringa_Data_Science_Core_Module2_W1_Independent_Project_2021_05_Wanjiru_Kinyara_Python_Notebook"
Moringa_Data_Science_Core_Module2_W1_Independent_Project_2021_05_Wanjiru_Kinyara_Python_Notebook"_
[ ]
DEFINING THE QUESTION
a) Specifying the question
As a data analyst under Mchezopesa, there is a need to predict the outcome of a game based on who's home and who's away and whether or not the game is friendly

b)Defining the metric for success
The project is successful if the analysis from model developed predicts the result of a game between any two teams.

Understanding context
Mchezopesa Ltd wishes to implement a model that can effectively predict the results of two teams for purposes of managing the amount of money that they will attach to bets from their customers.

c) Experimental design
For the project to be a success, the following steps will be followed:

1) Reading the data.

2) Feature engineering.

3) Exploratory Data Analysis.

4) Check for multicollinearity.

5) Building the model.

6) Cross-validate the model.

7) Computing RMSE.

8) Create residual plots for your models.

9) Perform appropriate regressions on the data and provide justification.

10) Challenge the developed solution.

e) Data relevance
The data provided is relevant to this research as it includes past inofmation on perfoemance of various teams as well as the current ranking of the teams on FIfa.

f) Data validation
DATA PREPARATION
Importing libraries
[60]

import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline



fifa results preview
[61]
#loading the results dataset and having a look at it
fifa_results = pd.read_csv("/content/results.csv")
fifa_results

[62]
#fifa results head
fifa_results.head()

[63]
X = fifa_results.iloc[:, 2:6].values
y = fifa_results.iloc[:, 6].values
print(X)
print(y)
[['England' 0 0 'Friendly']
 ['Scotland' 4 2 'Friendly']
 ['England' 2 1 'Friendly']
 ...
 ['Algeria' 0 1 'African Cup of Nations']
 ['North Korea' 0 1 'Intercontinental Cup']
 ['Fiji' 1 1 'Pacific Games']]
['Glasgow' 'London' 'Glasgow' ... 'Cairo' 'Ahmedabad' 'Apia']
[64]
#fifa results tail
fifa_results.tail()

[65]
fifa_results.info
<bound method DataFrame.info of              date         home_team  ...   country  neutral
0      1872-11-30          Scotland  ...  Scotland    False
1      1873-03-08           England  ...   England    False
2      1874-03-07          Scotland  ...  Scotland    False
3      1875-03-06           England  ...   England    False
4      1876-03-04          Scotland  ...  Scotland    False
...           ...               ...  ...       ...      ...
40834  2019-07-18    American Samoa  ...     Samoa     True
40835  2019-07-18              Fiji  ...     Samoa     True
40836  2019-07-19           Senegal  ...     Egypt     True
40837  2019-07-19        Tajikistan  ...     India     True
40838  2019-07-20  Papua New Guinea  ...     Samoa     True

[40839 rows x 9 columns]>
[66]
#shape of our fifa results 
fifa_results.shape
(40839, 9)
[67]
qstn = fifa_results[['home_score','away_score','tournament']]
qstn.head()

fifa ranking dataset preview
[ ]
↳ 5 cells hidden
DATA CLEANING
fifa_results clean-up
[ ]
↳ 2 cells hidden
fifa_ranking dataset cleaning
[ ]
↳ 4 cells hidden
merging the datasets
[ ]
↳ 26 cells hidden
EXPLORATORY DATA ANALYSIS
[102]
#univariate analysis
results.describe()

[103]

#home score measures of central tendency and dispersion
print(f'Min: {results.home_score.min()}')
print(f'Q1: {results.home_score.quantile(.25)}')
print(f'Q2: {results.home_score.quantile(.50)}')
print(f'Q3: {results.home_score.quantile(.75)}')
print(f'Max: {results.home_score.max()}')
print('*'*15)

print(f'Mean: {results.home_score.mean()}')

Min: 0.0
Q1: 0.0
Q2: 1.0
Q3: 2.0
Max: 15.0
***************
Mean: 1.571214841412328
Median: 1.0
Mode: 1.0
***************
Skew: 1.7982608240013744
Kurtosis: 5.94148317400417
[104]
#away score measures of central tendency and dispersion
print(f'Min: {results.away_score.min()}')
print(f'Q1: {results.away_score.quantile(.25)}')
print(f'Q2: {results.away_score.quantile(.50)}')
print(f'Q3: {results.away_score.quantile(.75)}')
print(f'Max: {results.away_score.max()}')
print('*'*15)

print(f'Mean: {results.away_score.mean()}')
print(f'Median: {results.away_score.median()}')

Min: 0.0
Q1: 0.0
Q2: 1.0
Q3: 2.0
Max: 12.0
***************
Mean: 1.0780969479353681
Median: 1.0
Mode: 0.0
***************
Skew: 1.8083752108223807
Kurtosis: 5.3688198142673
[105]

#Count plots
plt.figure(figsize = [10,8])
sns.countplot(x='year',data=results)
plt.title('Games Held per Year')
plt.xticks(rotation = 90)
plt.show()

[106]
#Tournaments Top 10
plt.figure(figsize = [10,8])
top_10 = results['tournament'].value_counts().sort_values(ascending=False).head(10)
top_10.sort_values(ascending=True).plot(kind='barh')
plt.xlabel('Number of Matches')
plt.ylabel('Competition')
plt.title('Number of Matches played by Tournament Type')
plt.show()

[107]
#Country played in Top 10
plt.figure(figsize = [10,8])
top_10 = results['country'].value_counts().sort_values(ascending=False).head(10)
top_10.sort_values(ascending=True).plot(kind='barh')
plt.xlabel('Number of Matches')
plt.ylabel('Country')
plt.title('Number of Matches played in a Country')
plt.show()

[108]
#Histogram of home scores
plt.figure(figsize = [10,8])
plt.hist(results['home_score'])
plt.title('Histogram of Home Scores')
plt.show()

[109]

#Histogram of away scores
plt.figure(figsize = [10,8])
plt.hist(results['away_score'])
plt.title('Histogram of Away Scores')
plt.show()

[110]
neutral = pd.get_dummies(results['neutral'],drop_first=True)
neutral.rename(columns = {True:'neutral_encod'}, inplace = True)
neutral.head()

[111]
results.head(1)

[112]
games = pd.concat([results,neutral],axis=1)
games.head()

[113]
sns.pairplot(games[['home_score', 'away_score', 'rank', 'year']])

[114]
#finding correlations
games.corr()

[115]
#games correlation heatmap
sns.heatmap(games.corr(),annot=True)
plt.show()

Feature engineering
[ ]
↳ 2 cells hidden
POLYNOMIAL REGRESSION MODEL
Model 1
[ ]
↳ 16 cells hidden
Model 2
Multicolinearity
[131]

# Model 2: Predict how many goals the home team scores. dependent variable here is away_score

independent_away_goals = games1.drop(columns=['home_team', 'away_team', 'away_score', 'tournament'])
correlations_away_goals = independent_away_goals.corr()
correlations_away_goals

[132]

# Let's use these correlations to compute the VIF score for each variable.
pd.DataFrame(np.linalg.inv(correlations_away_goals.values), index = correlations_away_goals.index, 
             columns=correlations_away_goals.columns)

VIF scores are less than 5, hence little multicolinearity

Model Building
[ ]
↳ 4 cells hidden
Cross Validation
[137]

folds = KFold(n_splits=5)
print('we are using ' +str(folds.get_n_splits(A)) + ' folds')

RMSES = [] # We will use this array to keep track of the RSME of each model
count = 1 # This will just help 
for train_index, test_index in folds.split(A):
  print('\nTraining model ' + str(count))
  
  # set up the train and test based on the split determined by KFold


both models don't have a widespread range. model 2 however has the lowest RMSEs

Residual Plots and Heteroscedasticity
[ ]
↳ 4 cells hidden
LOGISTIC REGRESSION MODEL
Feature Engineering
[141]
games1.head()

[142]

# Feature Engineering: Figure out from the home team’s perspective if the game is a Win, Lose or Draw (W, L, D)

def match_result(row):
  if row['home_score'] > row['away_score']:
    outcome = 'Win'
  elif row['home_score'] < row['away_score']:
    outcome = 'Lose'
  else:
    outcome = 'Draw'


[143]
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
games1['result'] = labelencoder.fit_transform(games1['result'])
games1

Building The Model
[ ]
↳ 1 cell hidden
Hyperparameter Testing
[147]

# Alternative Solution

# scaling data as advised by the warning after running the previous cell.
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(C_train, d_train)
C_train = scaler.transform(C_train)


[148]
# Creating regularization penalty space
penalty = ['l1', 'l2']

# Creating regularization hyperparameter space
hyp_C = np.logspace(0, 4, 10)

solver = [ 'liblinear', 'sag', 'saga']

# Creating hyperparameter options
hyperparameters = dict(C=hyp_C, penalty=penalty, solver = solver, max_iter = (10,100))


[149]

# Viewing best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Solver:', best_model.best_estimator_.get_params()['solver'])
print('Best max_iter:', best_model.best_estimator_.get_params()['max_iter'])

[150]
# Predicting target vector
best_model.predict(C)

[151]
# Creating the logistic regression
logistic = linear_model.LogisticRegression().fit(X_train,y_train)
metrics.accuracy_score(d_test, d_pred)

[152]

logistic = linear_model.LogisticRegression(penalty='l1', C=1, max_iter=10, solver='saga').fit(C_train,d_train)

dc_pred = logistic.predict(C_test)
metrics.accuracy_score(d_test, dc_pred)

[153]

best_model.best_score_

CONCLUSION
The logistic regression was able to split results into win, lose and draw hence providing a 100% accuracy

we did have the right data and the right question for this dataset


