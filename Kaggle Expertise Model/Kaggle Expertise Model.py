# Databricks notebook source
#download the full data set via the kaggle python CLI
#instructions for getting an API token: https://github.com/Kaggle/kaggle-api
#!pip install kaggle
#!kaggle datasets download -d kaggle/meta-kaggle

# COMMAND ----------

# MAGIC %md #### Import Python Libraries

# COMMAND ----------

import pandas as pd
import numpy as np
import os
import requests
import json
import urllib
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn import linear_model, metrics, model_selection
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import databricks.koalas as ks

# Enable PyArrow to optimize moving from pandas to and from Saprk Dataframes
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

# MAGIC %md #### Read the Kaggle Data

# COMMAND ----------

CompetitionTags = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/CompetitionTags.csv')
Datasets = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Datasets.csv')
DatasetTags = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/DatasetTags.csv')
Competitions = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Competitions.csv')
DatasetVotes = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/DatasetVotes.csv')
Datasources = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Datasources.csv')
DatasetVersions = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/DatasetVersions.csv')
ForumMessageVotes = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/ForumMessageVotes.csv')
Forums = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Forums.csv')
ForumTopics = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/ForumTopics.csv')
KernelLanguages = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/KernelLanguages.csv')
Kernels = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Kernels.csv')
KernelTags = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/KernelTags.csv')
KernelVersionCompetitionSources = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/KernelVersionCompetitionSources.csv')
KernelVersionDatasetSources = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/KernelVersionDatasetSources.csv')
KernelVersionKernelSources = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/KernelVersionKernelSources.csv')
KernelVersionOutputFiles = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/KernelVersionOutputFiles.csv')
KernelVersions = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/KernelVersions.csv')
KernelVotes = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/KernelVotes.csv')
Organizations = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Organizations.csv')
Submissions = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Submissions.csv')
Tags = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Tags.csv')
ForumMessages = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/ForumMessages.csv')
TeamMemberships = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/TeamMemberships.csv')
UserAchievements = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/UserAchievements.csv')
UserFollowers = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/UserFollowers.csv')
UserOrganizations = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/UserOrganizations.csv')
Users = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Users.csv')
Teams = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Teams.csv')

# COMMAND ----------

# MAGIC %md ## User Expertise Quantification

# COMMAND ----------

# MAGIC %md ### Users

# COMMAND ----------

# rename Id column to UserId
Users = Users.withColumnRenamed('Id','UserId')

# COMMAND ----------

display(Users)

# COMMAND ----------

# Remove Performance Tier 0 (inactive) users 
Users = Users.filter((Users['PerformanceTier'] == 1) | (Users['PerformanceTier'] == 2) | (Users['PerformanceTier'] == 3) | (Users['PerformanceTier'] == 4) | (Users['PerformanceTier'] == 5))

# COMMAND ----------

# MAGIC %md Below, feature engineering is used to generate user-level expertise features that can be used to model overall user expertise

# COMMAND ----------

# MAGIC %md ### User Organizations

# COMMAND ----------

# MAGIC %md Determine the number organizations that a user is part of as a measure of "experience"

# COMMAND ----------

display(UserOrganizations)

# COMMAND ----------

# MAGIC %md #### Calculate # of Organizations Per User

# COMMAND ----------

Users_and_Orgs = Users.join(UserOrganizations, on = 'UserId', how = 'left')
NumOrgs = Users_and_Orgs.groupBy('UserId').agg(countDistinct('OrganizationId')).withColumnRenamed('count(DISTINCT OrganizationId)','NumOrgs')
Users = Users.join(NumOrgs, on = 'UserId', how = 'left')

# COMMAND ----------

display(Users)

# COMMAND ----------

# MAGIC %md ### User Followers

# COMMAND ----------

# MAGIC %md Determine the number of followers a user has as a measure of "popularity"

# COMMAND ----------

display(UserFollowers)

# COMMAND ----------

# MAGIC %md Calculate # of Followers

# COMMAND ----------

Users_and_Followers = Users.join(UserFollowers, on = 'UserId', how = 'left')
NumFollowers = Users_and_Followers.groupBy('UserId').agg(countDistinct('FollowingUserId')).withColumnRenamed('count(DISTINCT FollowingUserId)','NumFollowers')
Users = Users.join(NumFollowers, on = 'UserId', how = 'left')

# COMMAND ----------

display(Users)

# COMMAND ----------

# MAGIC %md ### User Achievements

# COMMAND ----------

# MAGIC %md Determine the number of user achievements in terms of ranking and medals as a measure of "competency"

# COMMAND ----------

display(UserAchievements)

# COMMAND ----------

Users_and_Achievements = Users.join(UserAchievements, on = 'UserId', how = 'left')
Medals = Users_and_Achievements[['UserId','AchievementType','TotalGold','TotalSilver','TotalBronze']].groupBy(['UserId','AchievementType']).sum()
Medals = Medals.withColumnRenamed('sum(TotalGold)', 'TotalGold').withColumnRenamed('sum(TotalSilver)', 'TotalSilver').withColumnRenamed('sum(TotalBronze)', 'TotalBronze').drop('sum(UserId)')

# COMMAND ----------

display(Medals)

# COMMAND ----------

# MAGIC %md #### Discussion Medals

# COMMAND ----------

# Gold Discussion Medals
TotalGoldDisc = Medals.filter(Medals['AchievementType'] == 'Discussion').select(['UserId', 'TotalGold']).withColumnRenamed('TotalGold', 'TotalGoldDisc')

# Silver Discussion Medals
TotalSilverDisc = Medals.filter(Medals['AchievementType'] == 'Discussion').select(['UserId', 'TotalSilver']).withColumnRenamed('TotalSilver', 'TotalSilverDisc')

# Bronze Discussion Medals
TotalBronzeDisc = Medals.filter(Medals['AchievementType'] == 'Discussion').select(['UserId', 'TotalBronze']).withColumnRenamed('TotalBronze', 'TotalBronzeDisc')

# Total Discussion Medals
TotalDisc = TotalGoldDisc.join(TotalSilverDisc, on = 'UserId').join(TotalBronzeDisc, on = 'UserId')
TotalDisc = TotalDisc.withColumn('TotalDisc', TotalDisc['TotalGoldDisc'] + TotalDisc['TotalSilverDisc'] + TotalDisc['TotalBronzeDisc'])

# COMMAND ----------

display(TotalDisc)

# COMMAND ----------

# MAGIC %md #### Competition Medals

# COMMAND ----------

# Gold Competition Medals
TotalGoldComp = Medals.filter(Medals['AchievementType'] == 'Competitions').select(['UserId', 'TotalGold']).withColumnRenamed('TotalGold', 'TotalGoldComp')

# Silver Competition Medals
TotalSilverComp = Medals.filter(Medals['AchievementType'] == 'Competitions').select(['UserId', 'TotalSilver']).withColumnRenamed('TotalSilver', 'TotalSilverComp')

# Bronze Competition Medals
TotalBronzeComp = Medals.filter(Medals['AchievementType'] == 'Competitions').select(['UserId', 'TotalBronze']).withColumnRenamed('TotalBronze', 'TotalBronzeComp')

# Total Competition Medals
TotalComp = TotalGoldComp.join(TotalSilverComp, on = 'UserId').join(TotalBronzeComp, on = 'UserId')
TotalComp = TotalComp.withColumn('TotalComp', TotalComp['TotalGoldComp'] + TotalComp['TotalSilverComp'] + TotalComp['TotalBronzeComp'])

# COMMAND ----------

display(TotalComp)

# COMMAND ----------

# MAGIC %md #### Kernel Medals

# COMMAND ----------

display(Medals)

# COMMAND ----------

# Gold Kernel Medals
TotalGoldScript = Medals.filter(Medals['AchievementType'] == 'Scripts').select(['UserId', 'TotalGold']).withColumnRenamed('TotalGold', 'TotalGoldScript')

# Silver Kernel Medals
TotalSilverScript = Medals.filter(Medals['AchievementType'] == 'Scripts').select(['UserId', 'TotalSilver']).withColumnRenamed('TotalSilver', 'TotalSilverScript')

# Bronze Kernel Medals
TotalBronzeScript = Medals.filter(Medals['AchievementType'] == 'Scripts').select(['UserId', 'TotalBronze']).withColumnRenamed('TotalBronze', 'TotalBronzeScript')

# Total Kernel Medals
TotalScript = TotalGoldScript.join(TotalSilverScript, on = 'UserId').join(TotalBronzeScript, on = 'UserId')
TotalScript = TotalScript.withColumn('TotalScript', TotalScript['TotalGoldScript'] + TotalScript['TotalSilverScript'] + TotalScript['TotalBronzeScript'])

# COMMAND ----------

display(TotalScript)

# COMMAND ----------

# Join Users Table with Medals tables
Users = Users.join(TotalDisc, on='UserId', how='left').join(TotalComp, on='UserId', how='left').join(TotalScript, on='UserId', how='left')

# Add Total Medals
Users = Users.withColumn('TotalMedals', Users['TotalDisc'] + Users['TotalComp'] + Users['TotalScript'])

# COMMAND ----------

display(Users)

# COMMAND ----------

# MAGIC %md ### Kernels Created
# MAGIC Calculate the number of Kernels, or scripts created on Kaggle as a measure of "experience"

# COMMAND ----------

display(Kernels)

# COMMAND ----------

Kernels = Kernels.withColumnRenamed('AuthorUserId', 'UserId')
Users_and_Kernels_Created = Users.join(Kernels, on = 'UserId', how = 'left')
KernelsCreated = Users_and_Kernels_Created.groupBy(['UserId']).agg(countDistinct('Id')).withColumnRenamed('count(DISTINCT Id)','KernelsCreated')
Users = Users.join(KernelsCreated, on='UserId', how='left')

# COMMAND ----------

display(Users)

# COMMAND ----------

# MAGIC %md ### Kaggle Rankings
# MAGIC Create training set of "known experts" and "non-experts" from Kaggle User Rankings: https://www.kaggle.com/rankings

# COMMAND ----------

# Calculate Percentile Ranking for Competitions, Discussions, and Scripts from the Kaggle Rankings
Rankings = UserAchievements.select(['UserId', 'AchievementType','HighestRanking']).dropna()

CompetitionRankings = Rankings.filter(Rankings['AchievementType'] == 'Competitions').drop('AchievementType').withColumnRenamed('HighestRanking', 'CompetitionRanking')
DiscussionRankings = Rankings.filter(Rankings['AchievementType'] == 'Discussion').drop('AchievementType').withColumnRenamed('HighestRanking', 'DiscussionRanking')
ScriptRankings = Rankings.filter(Rankings['AchievementType'] == 'Scripts').drop('AchievementType').withColumnRenamed('HighestRanking', 'ScriptRanking')

CompetitionPerRanking = CompetitionRankings.select('UserId', 'CompetitionRanking', percent_rank().over(Window.partitionBy().orderBy(CompetitionRankings['CompetitionRanking'].desc())).alias('CompetitionPerRanking')).drop('CompetitionRanking')

DiscussionPerRanking = DiscussionRankings.select('UserId', 'DiscussionRanking', percent_rank().over(Window.partitionBy().orderBy(DiscussionRankings['DiscussionRanking'].desc())).alias('DiscussionPerRanking')).drop('DiscussionRanking')

ScriptPerRanking = ScriptRankings.select('UserId', 'ScriptRanking', percent_rank().over(Window.partitionBy().orderBy(ScriptRankings['ScriptRanking'].desc())).alias('ScriptPerRanking')).drop('ScriptRanking')

Users = Users.join(CompetitionPerRanking, on='UserId', how='left').join(DiscussionPerRanking, on='UserId', how='left').join(ScriptPerRanking, on='UserId', how='left')

# COMMAND ----------

# MAGIC %md 
# MAGIC Define experts as users who are:
# MAGIC - Currently ranked for competitions, scripts, and/or discussions
# MAGIC - Performance Tier 2+
# MAGIC - Have at least 5 followers
# MAGIC - Have least 2 medals for competitions, scripts, and/or discussions and 3 medals overall
# MAGIC - Have created at least 3 kernels

# COMMAND ----------

display(Users)

# COMMAND ----------

experts = Users.filter(((col('CompetitionPerRanking') > 0) | (col('ScriptPerRanking') >= 0) | (col('DiscussionPerRanking') >= 0)) \
                       &((col('PerformanceTier') > 1) & (col('NumFollowers') >= 5)) \
                       &((col('TotalDisc') >= 2) | (col('TotalComp') >= 2) | (col('TotalScript') >= 2)) \
                       &((col('KernelsCreated') >= 3) & (col('TotalMedals') >= 3)))

# COMMAND ----------

experts.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Define non-experts as users who are:
# MAGIC - Performance Tier 1
# MAGIC - Unranked in Competitions, Scripts, and Discussions
# MAGIC - Have less than 5 followers
# MAGIC - Have at most 1 medal in competitions, scripts, discussions
# MAGIC - Created at most 1 kernel

# COMMAND ----------

non_experts = Users.filter((col('CompetitionPerRanking').isNull()) & (col('ScriptPerRanking').isNull()) & (col('DiscussionPerRanking').isNull()) & (col('PerformanceTier') == 1) \
                           & (col('NumFollowers') < 5) & (Users['TotalDisc'] == 0) & (Users['TotalComp'] == 0) & (Users['TotalScript'] == 0) \
                           & (col('KernelsCreated') < 2))

# COMMAND ----------

non_experts.count()

# COMMAND ----------

Users.count()

# COMMAND ----------

# create label column 'Expert'
non_experts = non_experts.withColumn('Expert', lit(0))
experts = experts.withColumn('Expert', lit(1))

# COMMAND ----------

# MAGIC %md ### Addressing Class Imbalance: Downsample

# COMMAND ----------

# reduce the number of non-expert examples to the same as the number of experts
non_experts_downsampled = non_experts.sample(False, 0.1, 1).limit(experts.count())

# COMMAND ----------

non_experts_downsampled.count()

# COMMAND ----------

# create training set with experts and non-experts
train = experts.union(non_experts_downsampled)

# COMMAND ----------

# create test set with the remaining active users
test = Users.join(train.drop('TotalMedals'), how = 'leftanti', on = 'UserId')

# COMMAND ----------

train.count()

# COMMAND ----------

test.count()

# COMMAND ----------

# Convert train Spark DF to a Pandas DF now that it has been pared down
train_df = train.toPandas()
test_df = test.toPandas()

# COMMAND ----------

train_df.head()

# COMMAND ----------

test_df.head()

# COMMAND ----------

# MAGIC %md ### Quantifying Overall User Expertise - Logistic Regression
# MAGIC In order to model overall expertise, logistic regression was used to determine the "probability" that a user is an expert based on the engineered features (# of Organizations, # of Followers, # of Achievements, # Datasets Created, # of Kernels created).
# MAGIC 
# MAGIC This model reflects the official Kaggle Progression System.

# COMMAND ----------

# MAGIC %md ### Categories of Expertise
# MAGIC Three Kaggle categories of data science expertise: <b>Competitions</b>, <b>Kernels</b>, and <b>Discussion</b>.
# MAGIC <br> Advancement through performance tiers is done independently within each category of expertise.

# COMMAND ----------

# MAGIC %md ###  Performance Tiers
# MAGIC Within each category of expertise, there are five performance tiers that can be achieved in accordance with the quality and quantity of work you produce: <b>Novice (1)</b>, <b>Contributor (2)</b>, <b>Expert (3)</b>, <b>Master (4)</b>, and <b>Grandmaster (5)</b>. 
# MAGIC <br> https://www.kaggle.com/progression

# COMMAND ----------

# MAGIC %md #### Determine if there is any multicolinearity between variables

# COMMAND ----------

plt.subplots(figsize=(10,8))
display(sns.heatmap(train_df.corr(), annot=True, cmap='RdBu'))

# COMMAND ----------

# MAGIC %md #### Split into Training and Test Split and fit Logistic Regression Model to Data

# COMMAND ----------

train_df.columns

# COMMAND ----------

#train_df = train_df.drop(['ExpertisePred', 'ExpertiseScore'], axis = 'columns')

# COMMAND ----------

### Add Tier as a categorical variable
### Create a "Ranked" column for 0 or 1 ranked or not ranked in any of the 3 - categorical column

# COMMAND ----------

# fill missing values with 0
train_df = train_df.fillna(0)

# calculate average percentile ranking across competition, discussion, and scripts
train_df['AvgPerRanking'] = (train_df['CompetitionPerRanking'] + train_df['DiscussionPerRanking'] + train_df['ScriptPerRanking'])/3

# create a Ranked column that indicates whether the Kaggle user is ranked in any of competition, discussion, or scripts
ranked = []
for i in train_df['AvgPerRanking']:
  if i > 0:
    ranked.append(1)
  else:
    ranked.append(0)
    
train_df['Ranked'] = ranked

# COMMAND ----------

# function to encode categorical variables
def encode_categorical(data, cols, replace=False):
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)

# COMMAND ----------

# encode PerformenaceTier and Ranked Columns
train_df, _, _ = encode_categorical(train_df, ['PerformanceTier', 'Ranked'], replace = True)

# COMMAND ----------

train_df.head(3)

# COMMAND ----------

## Prepare the Test DF

# fill missing values with 0
test_df = test_df.fillna(0)

# calculate average percentile ranking across competition, discussion, and scripts
test_df['AvgPerRanking'] = (test_df['CompetitionPerRanking'] + test_df['DiscussionPerRanking'] + test_df['ScriptPerRanking'])/3

# create a Ranked column that indicates whether the Kaggle user is ranked in any of competition, discussion, or scripts
ranked = []
for i in test_df['AvgPerRanking']:
  if i > 0:
    ranked.append(1)
  else:
    ranked.append(0)
    
test_df['Ranked'] = ranked

# encode PerformenaceTier and Ranked Columns
test_df, _, _ = encode_categorical(test_df, ['PerformanceTier', 'Ranked'], replace = True)

# COMMAND ----------

y = train_df['Expert']
X = train_df[['NumFollowers', 'TotalMedals', 'KernelsCreated', 'Ranked']]

# Split into 80-20 Training-Test Set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

lr = LogisticRegression()
lr.fit(X_train,y_train)

# COMMAND ----------

# MAGIC %md #### Metrics on Logit Model

# COMMAND ----------

y_pred = lr.predict(X_val)

print('Intercept', lr.intercept_)
print('Coefs:', lr.coef_)
print()
print('MSE:', metrics.mean_squared_error(y_val, y_pred))
print('Accuracy:', metrics.accuracy_score(y_val, y_pred))
print('Precision:', metrics.precision_score(y_val, y_pred))
print('Recall:', metrics.recall_score(y_val, y_pred))
print('AUC:', metrics.roc_auc_score(y_val, y_pred))

# COMMAND ----------

# MAGIC %md #### Confusion Matrix
# MAGIC For threshold > 0.50 probability of being an expert

# COMMAND ----------

score = lr.score(X_val, y_val)
cm = metrics.confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 5);
display()

# COMMAND ----------

df = train_df.append(test_df, sort = 'False')

# COMMAND ----------

#df = df.drop(['ExpertisePred', 'ExpertiseScore'], axis = 'columns')

# COMMAND ----------

df['ExpertisePred'] = pd.Series(lr.predict(df[['NumFollowers', 'TotalMedals', 'KernelsCreated', 'Ranked']]))
                                
df['ExpertiseScore'] = pd.DataFrame(lr.predict_proba(df[['NumFollowers', 'TotalMedals', 'KernelsCreated', 'Ranked']]))[1]

# COMMAND ----------

df.sort_values('ExpertiseScore', ascending = False)

# COMMAND ----------

train_df[train_df['Expert']==1]['KernelsCreated'].value_counts()

# COMMAND ----------

df.sort_values('ExpertiseScore', ascending = False)[['UserName', 'ExpertiseScore', 'ExpertisePred', 'PerformanceTier', 'NumOrgs', 'NumFollowers', 'KernelsCreated', 'TotalDisc', 'TotalComp', 'TotalScript', 'CompetitionPerRanking', 'DiscussionPerRanking', 'ScriptPerRanking']]

# COMMAND ----------

#display(Users2.groupBy('PerformanceTier').count().orderBy('count'))

# COMMAND ----------

'''### Save for kernel skill quantification

# Kernel Rank Categorical Variable: Gold = 1, Silver = 2, Bronze = 3, No Medal = 4
Kernels = Kernels.fillna(4, subset = ['Medal']).withColumnRenamed('Medal','MedalRank')

df_basket1.select("Item_group","Item_name","Price", F.percent_rank().over(Window.partitionBy(df_basket1['Item_group']).orderBy(df_basket1['price'])).alias("percent_rank"))
df_basket1.show()'''