DEFINING THE QUESTION
i) Specifying the Question
As a football recruit under Mchezopesa Ltd, prdict the result of a game between tean A and team B, taking who's home, who's away and wheteher or not the game is friendly as your factors of interest. include rank in your training

ii) Defining the Metric for Success
The project is successful if the analysis from models developed predicts the result of a game between any two teams.

iii) Understanding the Context
Mchezopesa Ltd wishes to implement a model that can effectively predict the results of two teams for purposes of managing the amount of money that they will attach to bets from their customers.

iv) Experimental Design
Perform EDA

Check for multicollinearity

Build the Polynomial Regression Model

Cross-validate the model

Create residual plots and test for heterescedasticity using Bartlett's test

Feaature engineering

Build the Logistic Regression Model

Hyperparameter tuning

DATA PREPARATION
[1]

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.linear_model import LogisticRegression
[2]
#loading our datasets
fifa_ranking = pd.read_csv("/content/fifa_ranking.csv")
fifa_ranking
[3]
fifa_results = pd.read_csv("/content/results.csv")
fifa_results
[4]
#a look at the shape of the datasets
fifa_ranking.shape
[5]
fifa_results.shape
[6]
#datasets information
fifa_ranking.info()
[7]
fifa_results.info()
[8]
#a look at fifa_ranking head 
fifa_ranking.head()
[9]
#tail
fifa_ranking.tail()
[10]
#fifa_results head
fifa_results.head()
[11]
#tail
fifa_results.tail()
DATA VALIDATION
[85]
fifa_ranking_valid = fifa_ranking[:1]
fifa_ranking_valid
[ ]
fifa_ranking_valid1 = pd.read_html("https://en.wikipedia.org/wiki/1993_UEFA_Cup_Final")
DATA CLEANING
[12]
#looking for null values
fifa_ranking.isna().sum()
[13]
#looking for duplicates
fifa_ranking.duplicated().sum()
[14]
fifa_ranking[fifa_ranking.duplicated()]
#the duplicated data is from Sudan and there doesn't appear any errors within
[15]
#results missing values
fifa_results.isna().sum()
[16]
#results duplicates
fifa_results.duplicated().sum()
Merging the datasets
[17]
fifa_ranking.head(1)
[18]
fifa_rankingdf = fifa_ranking.drop(columns=['country_abrv','total_points','previous_points','rank_change','cur_year_avg','cur_year_avg_weighted','last_year_avg','last_year_avg_weighted','two_year_ago_avg','two_year_ago_weighted','three_year_ago_avg','three_year_ago_weighted','confederation'])
fifa_rankingdf
[21]
#we have to convert rank_ddate to a dtaetime
fifa_rankingdf.rank_date = pd.to_datetime(fifa_rankingdf.rank_date)
fifa_rankingdf.head(5)
[22]
fifa_rankingdf.tail()
[23]
fifa_rankingdf['year'] = fifa_rankingdf['rank_date'].dt.year
fifa_rankingdf.tail(10)
#the last year is 2018
[24]
#let's have a look at our fifa_results dataset
fifa_results.head(5)
[26]
#converting the date to a datetime
fifa_results.date = pd.to_datetime(fifa_results.date)
fifa_results['year'] = fifa_results['date'].dt.year
fifa_results.tail(5)
#the last entries for our results dataset is 2019
[27]
#dropping all entries beyond 2018 in order to merge our datasets
fifa_results1 = fifa_results[fifa_results.year < 2019]
fifa_results1.tail()
[28]
#fifa ranking considers last four years therefore we'll be using four years worth of data
fifa_results1 = fifa_results1[fifa_results1.year > 2014]
fifa_results1.head(10)
[30]
#merging the sets
fifa_merged = fifa_results1.merge(fifa_rankingdf, left_on=['home_team', 'year'], right_on=['country_full', 'year'], how='inner')
fifa_merged
[31]

fifa_merged = fifa_merged.merge(fifa_rankingdf, left_on=['away_team', 'year'], right_on=['country_full', 'year'], how='inner')
fifa_merged
[32]
#dropping unnecessary columns
fifa_merged = fifa_merged.drop(columns=['city', 'country', 'neutral', 'year', 'country_full_x', 'country_full_y'])

fifa_merged = fifa_merged[fifa_merged.rank_date_x == fifa_merged.rank_date_y]
fifa_merged
[33]
#duplicates
fifa_merged.duplicated(subset=['date','home_team','away_team','home_score','away_score','tournament']).sum()
[35]
#keeping only the first entry of each match
fifa_merged.drop_duplicates(subset=['date','home_team','away_team','home_score','away_score','tournament'], keep= 'first', inplace= True)
fifa_merged
[36]
#renaming columns for better understanding
fifa_merged.rename(columns={'rank_x':'home_team_rank', 'rank_y':'away_team_rank'},inplace=True)
fifa_merged
[38]
#changing tournament type to binary numbers
fifa_merged.tournament.nunique()
[40]
  else:
    return 2

fifa_merged['competition'] = fifa_merged['tournament'].apply(lambda x: Tourna(x))
fifa_merged['competition'].unique()
[41]
#dropping off more unnecessary columns
fifa_merged = fifa_merged.drop(columns=['rank_date_x', 'rank_date_y'])
EXPLORATORY DATA ANALYSIS
[42]
fifa_merged.info()
[43]
col_names = ['home_score','away_score', 'home_team_rank', 'away_team_rank', 'competition']

fig, ax = plt.subplots(len(col_names), figsize= (8,40))

for i, col_val in enumerate(col_names):
  sns.boxplot(y = fifa_merged[col_val], ax= ax[i])
  ax[i].set_title('Box plot - {}'.format(col_val), fontsize= 10)
  ax[i].set_xlabel(col_val, fontsize= 8)
plt.show()
[88]
print(f'Mode: {fifa_merged.home_score.mode().values[0]}')
print('*'*15)

print(f'Skew: {fifa_merged.home_score.skew()}')
print(f'Kurtosis: {fifa_merged.home_score.kurt()}')
[89]
print(f'Mode: {fifa_merged.away_score.mode().values[0]}')
print('*'*15)

print(f'Skew: {fifa_merged.away_score.skew()}')
print(f'Kurtosis: {fifa_merged.away_score.kurt()}')
[91]
#Tournaments Top 10
plt.figure(figsize = [10,8])
top_10 = fifa_merged['tournament'].value_counts().sort_values(ascending=False).head(10)
top_10.sort_values(ascending=True).plot(kind='barh')
plt.xlabel('Number of Matches')
plt.ylabel('Competition')
plt.title('Number of Matches played by Tournament Type')
plt.show()
[92]
#Histogram of home scores
plt.figure(figsize = [10,8])
plt.hist(fifa_merged['home_score'])
plt.title('Histogram of Home Scores')
plt.show()
[93]
#Histogram of away scores
plt.figure(figsize = [10,8])
plt.hist(fifa_merged['away_score'])
plt.title('Histogram of Away Scores')
plt.show()
[95]
#games correlation heatmap
sns.heatmap(fifa_merged.corr(),annot=True)
plt.show()
[44]
#summary statistics
fifa_merged.describe()
A) POLYNOMIAL REGRESSION MODEL
MODEL 1
checking multicolinearity
[45]
fifa_merged.head()
[46]
#using home_score as the dependent variable, let's prdeict number of goals scored by the home team
independent_home_goals = fifa_merged.drop(columns=['date', 'home_team', 'away_team', 'home_score', 'tournament'])
correlated_home_goals = independent_home_goals.corr()
correlated_home_goals
[48]
#computing the VIF scores
pd.DataFrame(np.linalg.inv(correlated_home_goals.values), index = correlated_home_goals.index, 
             columns=correlated_home_goals.columns)
all the VIF scores are below 5, hence there is little multicolinearity in our data

Building the model
[49]
X = independent_home_goals.values
y = fifa_merged['home_score'].values

# Split the dataset into train and test sets
X_train, X_test, y_train,  y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

# Fit polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree = 2) 
X_poly = poly_reg.fit_transform(X)


[50]
#using 3 degrees of freedom
poly_reg_3 = PolynomialFeatures(degree = 3) 
X_poly_3 = poly_reg_3.fit_transform(X)

pol_3_reg = LinearRegression()
pol_3_reg.fit(X_poly_3, y)

y_pred_3 = pol_3_reg.predict(poly_reg_3.fit_transform(X_test))
[51]
#testing using 4 degrees of freedom
poly_reg_4 = PolynomialFeatures(degree = 4) 
X_poly_4 = poly_reg_4.fit_transform(X)

pol_4_reg = LinearRegression()
pol_4_reg.fit(X_poly_4, y)

y_pred_4 = pol_4_reg.predict(poly_reg_4.fit_transform(X_test))
[52]
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_3)))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_4)))
we have 4 features. It therefore is prudent to use the 4 degrees of freedom as iit has the lowest RMSE. this also reduces chances of overfitting and underfitting our model

Cross-validating the model
[53]
from sklearn.model_selection import KFold

folds = KFold(n_splits=5)
print('we are using ' +str(folds.get_n_splits(X)) + ' folds')

RMSES = [] #keeping track of RSME in each model
count = 1 
for train_index, test_index in folds.split(X):
  print('\nTraining model ' + str(count))
  
…  yc_pred = regressor.predict(Xc_test)
  
  rmse_value =  np.sqrt(metrics.mean_squared_error(yc_test, yc_pred))
  RMSES.append(rmse_value)
  
  print('Model ' + str(count) + ' Root Mean Squared Error:',rmse_value)
  count = count + 1
[54]

np.mean(RMSES)
The avaerage RMSE is very closely related to the inital RMSE chosen above. Model 3 however has the least RMSE in this case so that's what we are going with

Residual plots and heteroscedasticity - Bartlett's test
[55]
residuals_home_score = np.subtract(y_pred_4, y_test)

pd.DataFrame(residuals_home_score).describe()
[56]
plt.scatter(y_pred_4, residuals_home_score, color='black')
plt.ylabel('residual')
plt.xlabel('fitted values')
plt.axhline(y= residuals_home_score.mean(), color='red', linewidth=1)
plt.show()
there is a centered residual about the mean, close to 0. It shows the model is good

[58]
#performing the heteroscedasticity test 
import scipy as sp

test_result, p_value = sp.stats.bartlett(y_pred_4, residuals_home_score)

# computing a chi squared distribution critical value
degree_of_freedom = len(y_pred_4)-1
probability = 1 - p_value

critical_value = sp.stats.chi2.ppf(probability, degree_of_freedom)
print(critical_value)

if (test_result > critical_value):
  print('the variances are unequal, and the model should be reassessed')
else:
  print('the variances are homogeneous!')
MODEL 2
Checking multicolinearity
[59]
# Model 2: predicting how many goals the home team scores depending on the away_score

independent_away_goals = fifa_merged.drop(columns=['date', 'home_team', 'away_team', 'away_score', 'tournament'])
correlated_away_goals = independent_away_goals.corr()
correlated_away_goals
[60]
# computing the VIF scores
pd.DataFrame(np.linalg.inv(correlated_away_goals.values), index = correlated_away_goals.index, 
             columns=correlated_away_goals.columns)
All the VIF scores are below 5, hence this data has minimal colinearity

Building the model
[61]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

A = independent_away_goals.values
b = fifa_merged['away_score'].values

# Split the dataset into train and test sets
A_train, A_test, b_train, b_test = train_test_split(A,b, test_size = 0.2, random_state=0)

[62]
poly_reg_3 = PolynomialFeatures(degree = 3) 
A_poly_3 = poly_reg_3.fit_transform(A)

pol_3_reg = LinearRegression()
pol_3_reg.fit(A_poly_3, b)

b_pred_3 = pol_3_reg.predict(poly_reg_3.fit_transform(A_test))
[63]
poly_reg_4 = PolynomialFeatures(degree = 4) 
A_poly_4 = poly_reg_4.fit_transform(A)

pol_4_reg = LinearRegression()
pol_4_reg.fit(A_poly_4, b)

b_pred_4 = pol_4_reg.predict(poly_reg_4.fit_transform(A_test))
[64]

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(b_test, b_pred)))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(b_test, b_pred_3)))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(b_test, b_pred_4)))
we have 4 features. It therefore is prudent to use the 4 degrees of freedom as iit has the lowest RMSE. this also reduces chances of overfitting and underfitting our model

Cross-validating the model
[67]
  # setting of train and test as per the Kfold
  # 80% in the training set, 20% in test test
  Ac_train, Ac_test = A[train_index], X[test_index]
  bc_train, bc_test = b[train_index], y[test_index]
  
  # fitting a model
  regressor = LinearRegression()  
  regressor.fit(Xc_train, yc_train)
  
  # this will sesses the accuracy of the model fitted

[69]
np.mean(RMSES)
The average RMSE is very close to the initial value. We are going to be going with model 2 as it has the least RMSE and closer to the polynomial model degree of 4

Residual plots and heteroscedasticity using Bartlett's test
[70]
residuals_away_score = np.subtract(b_pred_4, b_test)

pd.DataFrame(residuals_away_score).describe()
[71]
plt.scatter(b_pred_4, residuals_away_score, color='black')
plt.ylabel('residual')
plt.xlabel('fitted values')
plt.axhline(y= residuals_away_score.mean(), color='red', linewidth=1)
plt.show()
there is a centered residual about the mean, close to 0. It shows the model is good

[72]
print(critical_value)

if (test_result > critical_value):
  print('the variances are unequal, and the model should be reassessed')
else:
  print('the variances are homogeneous!')
model 2 can predict the home team goals very sufficiently. the variance after cross-validation are homogenous. This is a good model

LOGISTIC REGRESSION MODEL
Feature Engineering
[73]
fifa_merged.head()
[74]
fifa_merged['result'] = fifa_merged.apply(match_result, axis=1)
fifa_merged
[75]
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
fifa_merged['result'] = labelencoder.fit_transform(fifa_merged['result'])
fifa_merged
Building the model
[77]
C = fifa_merged.drop(columns= ['date',	'home_team',	'away_team', 'tournament', 'result'])
d = fifa_merged['result']

from sklearn.model_selection import train_test_split
C_train, C_test, d_train, d_test = train_test_split(C, d, test_size = .2, random_state=20)

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression()
LogReg.fit(C_train, d_train)

Hyperparameter testing
[78]
# as per warning above
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(C_train, d_train)
C_train = scaler.transform(C_train)

#logistic regression to improve the regression 


[79]
hyperparameters = dict(C=hyp_C, penalty=penalty, solver = solver, max_iter = (10,100))

# grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# Fitting grid search
best_model = clf.fit(C_train, d_train)
[80]
# Viewing best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Solver:', best_model.best_estimator_.get_params()['solver'])
print('Best max_iter:', best_model.best_estimator_.get_params()['max_iter'])
[81]
# Predict target vector
best_model.predict(C)
[82]
# Creating the logistic regression
logistic = linear_model.LogisticRegression().fit(X_train,y_train)
metrics.accuracy_score(d_test, d_pred)
[83]
logistic = linear_model.LogisticRegression(penalty='l1', C=1, max_iter=10, solver='saga').fit(C_train,d_train)

dc_pred = logistic.predict(C_test)
metrics.accuracy_score(d_test, dc_pred)
[84]

best_model.best_score_
there is a 100% accuracy with this model. the c=1 and L1 penalty are our best parameters

CONCLUSION
Both the Polynomial and Logistic regression are fairly accurate in result prediction. the Logistic regression was able to provide a 100% accuracy since it was able to predict and categorise the results as either a win, loss or a draw.

The data provided was right for this project. We had to drop a few duplicates in order to work with the right entries.

Extra information may not be necessary as what is provided in the datasets was adequate for the project at hand.

The question was indeed interesting. Game predictions are useful to people and companies who place bets on playing teams


