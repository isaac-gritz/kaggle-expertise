# Databricks notebook source
#download the full data set via the kaggle python CLI
#instructions for getting an API token: https://github.com/Kaggle/kaggle-api
#!pip install kaggle
#!kaggle datasets download -d kaggle/meta-kaggle
#!pip install flashtext
#!pip install nltk
#%sh
#python -m nltk.downloader all
#!pip install beautifulsoup4

# COMMAND ----------

# MAGIC %sh
# MAGIC python -m nltk.downloader all

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
from flashtext import KeywordProcessor
import glob
import json
import pprint
import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import re
import string
from collections import Counter
import codecs
import html as ihtml
from bs4 import BeautifulSoup

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

# MAGIC %md ### Kernels
# MAGIC Calculate the number of Kernels, or scripts created and their popularity on Kaggle as a measure of "experience"

# COMMAND ----------

display(Kernels)

# COMMAND ----------

Kernels = Kernels.withColumnRenamed('AuthorUserId', 'UserId')
Users_and_Kernels_Created = Users.join(Kernels, on = 'UserId', how = 'left')
KernelsCreated = Users_and_Kernels_Created.groupBy(['UserId']).agg(countDistinct('Id')).withColumnRenamed('count(DISTINCT Id)','KernelsCreated')
Users = Users.join(KernelsCreated, on='UserId', how='left')

# COMMAND ----------

# Calculate a Kernel Score that represents the average of the percentile ranking for kernel views, comments, and votes
KernelsAvgViews = Users_and_Kernels_Created.groupBy(['UserId']).mean('TotalViews').withColumnRenamed('avg(TotalViews)','KernelsAvgViews').fillna(0)
KernelsAvgComments = Users_and_Kernels_Created.groupBy(['UserId']).mean('TotalComments').withColumnRenamed('avg(TotalComments)','KernelsAvgComments').fillna(0)
KernelsAvgVotes = Users_and_Kernels_Created.groupBy(['UserId']).mean('TotalVotes').withColumnRenamed('avg(TotalVotes)','KernelsAvgVotes').fillna(0)

KernelsViewsPerRanking = KernelsAvgViews.select('UserId', 'KernelsAvgViews', percent_rank().over(Window.partitionBy().orderBy(KernelsAvgViews['KernelsAvgViews'])).alias('KernelsViewsPerRanking')).drop('KernelsAvgViews')

KernelsCommentsPerRanking = KernelsAvgComments.select('UserId', 'KernelsAvgComments', percent_rank().over(Window.partitionBy().orderBy(KernelsAvgComments['KernelsAvgComments'])).alias('KernelsCommentsPerRanking')).drop('KernelsAvgComments')

KernelsVotesPerRanking = KernelsAvgVotes.select('UserId', 'KernelsAvgVotes', percent_rank().over(Window.partitionBy().orderBy(KernelsAvgVotes['KernelsAvgVotes'])).alias('KernelsVotesPerRanking')).drop('KernelsAvgVotes')

Users = Users.join(KernelsViewsPerRanking, on='UserId', how='left').join(KernelsCommentsPerRanking, on='UserId', how='left').join(KernelsVotesPerRanking, on='UserId', how='left')

Users = Users.withColumn('KernelScore', (col('KernelsViewsPerRanking') + col('KernelsCommentsPerRanking') + col('KernelsVotesPerRanking'))/3)

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
# MAGIC - Kernel score is in the bottom 25%

# COMMAND ----------

non_experts = Users.filter((col('CompetitionPerRanking').isNull()) & (col('ScriptPerRanking').isNull()) & (col('DiscussionPerRanking').isNull()) & (col('PerformanceTier') == 1) \
                           & (col('NumFollowers') < 5) & (Users['TotalDisc'] <= 1) & (Users['TotalComp'] <= 1) & (Users['TotalScript'] <= 1) \
                           & (col('KernelsCreated') < 2) & (col('KernelScore') <= 0.25))

# COMMAND ----------

#non_experts.count()

# COMMAND ----------

#Users.count()

# COMMAND ----------

# create label column 'Expert'
non_experts = non_experts.withColumn('Expert', lit(0))
experts = experts.withColumn('Expert', lit(1))

# COMMAND ----------

# MAGIC %md ### Addressing Class Imbalance: Downsample

# COMMAND ----------

# reduce the number of non-expert examples to the same as the number of experts
non_experts_downsampled = non_experts.sample(False, 0.2, 1).limit(experts.count())

# COMMAND ----------

#non_experts_downsampled.count()

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
train_df, _, _ = encode_categorical(train_df, ['Ranked'], replace = True)

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
test_df, _, _ = encode_categorical(test_df, ['Ranked'], replace = True)

# COMMAND ----------

y = train_df['Expert']
X = train_df[['NumFollowers', 'TotalMedals', 'KernelsCreated', 'KernelScore', 'Ranked']]

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

df['ExpertisePred'] = pd.Series(lr.predict(df[['NumFollowers', 'TotalMedals', 'KernelsCreated', 'KernelScore', 'Ranked']]))
df['ExpertiseScore'] = pd.DataFrame(lr.predict_proba(df[['NumFollowers', 'TotalMedals', 'KernelsCreated', 'KernelScore', 'Ranked']]))[1]

# COMMAND ----------

df[['UserName','NumFollowers', 'TotalMedals', 'KernelsCreated','KernelScore', 'Ranked', 'ExpertiseScore']].head(10)

# COMMAND ----------

df[(df['UserName']=='jiweiliu')]

# COMMAND ----------

# MAGIC %md
# MAGIC ## User Skill Quantification
# MAGIC ### Skill Taxonomy
# MAGIC Tags are associated to Kaggle competitions, datasets, and kernels. These tags will be used as the taxonomy for skills.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tags

# COMMAND ----------

display(Tags.limit(10))

# COMMAND ----------

# rename Id column to TagId
Tags = Tags.withColumnRenamed('Id','TagId')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Kernel-Skill Quantification
# MAGIC Determine a kernel score for each associated tag/skill

# COMMAND ----------

# MAGIC %md
# MAGIC #### Kernel Tags

# COMMAND ----------

display(Kernels.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Determine kernel scores per skill

# COMMAND ----------

Kernels = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Kernels.csv')
Kernels = Kernels.withColumnRenamed('AuthorUserId', 'UserId')

# COMMAND ----------

# rename Id column to TagId
Kernels = Kernels.withColumnRenamed('Id','KernelId')

# Calculate a Kernel Score that represents the percentile ranking for kernel views, comments, and votes
KernelsViewsPercentile = Kernels.select('KernelId', 'TotalViews', percent_rank().over(Window.partitionBy().orderBy(Kernels['TotalViews'])).alias('KernelsViewsPercentile')).drop('TotalViews')
KernelsCommentsPercentile= Kernels.select('KernelId', 'TotalComments', percent_rank().over(Window.partitionBy().orderBy(Kernels['TotalComments'])).alias('KernelsCommentsPercentile')).drop('TotalComments')
KernelsVotesPercentile = Kernels.select('KernelId', 'TotalVotes', percent_rank().over(Window.partitionBy().orderBy(Kernels['TotalVotes'])).alias('KernelsVotesPercentile')).drop('TotalVotes')

Kernels = Kernels.join(KernelsViewsPercentile, on='KernelId', how='left').join(KernelsCommentsPercentile, on='KernelId', how='left').join(KernelsVotesPercentile, on='KernelId', how='left')
Kernels = Kernels.withColumn('KernelScore', (col('KernelsViewsPercentile') + col('KernelsCommentsPercentile') + col('KernelsVotesPercentile'))/3)

# COMMAND ----------

Kernels2 = Kernels[['KernelId','UserId','Medal','TotalViews','TotalComments','TotalVotes', 'KernelScore']]
KernelTags = KernelTags[['KernelId','TagId']]

Kernel_and_Tags = Kernels2.join(KernelTags, on = 'KernelId', how = 'left')
Kernel_and_Tags = Kernel_and_Tags.join(Tags, on = 'TagId', how = 'left')
Kernel_and_Tags = Kernel_and_Tags.dropna(subset=['Name'])

# COMMAND ----------

display(Kernel_and_Tags.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Kernel Scores Per User Per Skill

# COMMAND ----------

# Calculate Average Kernel Score Per User Per skill
KernelScores = Kernel_and_Tags[['UserId','KernelScore','Name']].groupBy(['UserId','Name']).mean().fillna(0).drop('avg(UserId)').withColumnRenamed('avg(KernelScore)', 'KernelScore')

# COMMAND ----------

display(KernelScores.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Competition-Skill Quantification
# MAGIC Determine a competition score for each associated skill/tag

# COMMAND ----------

# Joining Competitions, CompetitionTags, Teams, TeamMemberships, and User tables to get Competition Data and Skills per user
display(Competitions.limit(3))

# COMMAND ----------

display(CompetitionTags.limit(3))

# COMMAND ----------

Competitions2 = Competitions[['Id','OrganizationId']]

# rename Id column to CompetitionId
Competitions2 = Competitions2.withColumnRenamed('Id','CompetitionId')

# COMMAND ----------

display(Teams.limit(3))

# COMMAND ----------

display(TeamMemberships.limit(3))

# COMMAND ----------

# rename Id column to TeamId
Teams = Teams.withColumnRenamed('Id','TeamId')

# Linking Teams to Competitions, Competitions to Tags, and Teams to Users to get competition data and tags per user
Teams2 = Teams.dropna(subset = ['ScoreFirstSubmittedDate'])
Teams2 = Teams2[['TeamId','CompetitionId','Medal','PublicLeaderboardRank']]

# Calculate each team's public leaderboard percentile rank
TeamsRankPer = Teams2.select('CompetitionId', 'TeamId', 'PublicLeaderboardRank', percent_rank().over(Window.partitionBy().orderBy(Teams2['PublicLeaderboardRank'].desc())).alias('RankPer')).drop('PublicLeaderboardRank')

# Calculate each team's medal percentile rank
TeamsMedalPer = Teams2.fillna(4, subset = ['Medal']).select('CompetitionId', 'TeamId', 'Medal', percent_rank().over(Window.partitionBy().orderBy(Teams2['Medal'])).alias('MedalPer')).drop('Medal')

# Calcualte competition score as an average of the team's rank percentile and medal percentile
Teams2 = Teams2.join(TeamsRankPer, on = 'TeamId', how = 'left').join(TeamsMedalPer, on = 'TeamId', how = 'left')
Teams2 = Teams2.withColumn('CompScore', (col('RankPer') + col('MedalPer'))/2)

# Join teams with competition tags and team membership to get team competition scores associated with skills and users
Teams_and_Tags = Teams2.join(CompetitionTags, on = 'CompetitionId', how = 'left')
Teams_and_Tags = Teams_and_Tags.dropna(subset = ['TagId'])
Teams_and_Tags = Teams_and_Tags.join(Tags, on = 'TagId', how = 'left')

# rename Id column to TeamId
TeamMemberships = TeamMemberships[['TeamId', 'UserId']]
Users_and_Tags = Teams_and_Tags.join(TeamMemberships, on = 'TeamId', how = 'left')

# COMMAND ----------

display(Users_and_Tags.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Competition Score Per User Per Skill

# COMMAND ----------

# Calculate Competition Score Per User Per skill
CompScores = Users_and_Tags[['UserId', 'Name', 'CompScore']].groupBy(['UserId','Name']).mean().fillna(0).drop('avg(UserId)').withColumnRenamed('avg(CompScore)', 'CompScore')

# COMMAND ----------

display(CompScores.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussion-Skill Quantification
# MAGIC Determine a discussion score for each associated skill/tag using NLP

# COMMAND ----------

display(ForumMessages.limit(10))

# COMMAND ----------

display(Forums.limit(3))

# COMMAND ----------

Forums2 = Forums.withColumnRenamed('Id','ForumTopicId')

# COMMAND ----------

# Rename Ids
Forums = Forums.withColumnRenamed('Id','ForumTopicId')
ForumMessages = ForumMessages.withColumnRenamed('Id','MessageId').withColumnRenamed('PostUserId','UserId')

# Join Forums with Messages
Forums_and_Messages = Forums2.join(ForumMessages, how = 'left', on = 'ForumTopicId')

# Remove blank messages and keep necessary columns
Forums_and_Messages = Forums_and_Messages[['UserId', 'ForumTopicId', 'Title', 'Message', 'Medal']].dropna(subset = ['Message'])

# COMMAND ----------

# convert to pandas df to use NLP techniques
Forums_and_Messages_df = Forums_and_Messages.toPandas()

# COMMAND ----------

Forums_and_Messages_df.head(3)

# COMMAND ----------

# function to remove html tags, hyperlinks, punctuation, numbers, and stem text from a df
def clean_text_df(df, col):
    cleaned_text = []
    for i in df[col]:
        # remove html tags
        text = BeautifulSoup(ihtml.unescape(i), "lxml").text
        # standardize spacing
        text = re.sub(r"\s+", " ", text)
        # remove hyperlinks
        text = re.sub(r"http[s]?://\S+", "", text)
        filtered_tokens = []
        if str(text) != 'nan' and str(text) != '' and str(text) != ' ':
            # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            for token in tokens:
                # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
                if re.search('[a-zA-Z]', token):
                    # remove punctuation
                    filtered_tokens.append(token.translate(str.maketrans({key: None for key in string.punctuation})))
        else:
            filtered_tokens.append('')
        stems = [stemmer.stem(t) for t in filtered_tokens]
        phrase = ' '.join(stems)
        cleaned_text.append(phrase)
    return cleaned_text

# COMMAND ----------

# function to remove punctuation, numbers, and stem text from a list
def clean_list(lst):
    filtered_tokens = []
    for token in lst:
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        if re.search('[a-zA-Z]', token):
            # remove punctuation
            filtered_tokens.append(token.translate(str.maketrans({key: None for key in string.punctuation})))
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

# COMMAND ----------

# define stopwords
stopwords = nltk.corpus.stopwords.words('english')

# define stemmer
stemmer = SnowballStemmer("english")

# COMMAND ----------

Forums_and_Messages_df['Message Clean'] = clean_text_df(Forums_and_Messages_df, 'Message')

# COMMAND ----------

Forums_and_Messages_df.head(5)

# COMMAND ----------

Tags_df = Tags.toPandas()

# COMMAND ----------

Tags_df.head(3)

# COMMAND ----------

# Create a skill dictionary 
synonyms = []
for i in Tags_df['Name']:
    synonyms.append([i])
    
Tags_df['Name Syns'] = synonyms

skills = {}
for idx, i in enumerate(Tags_df['Name Syns']):
    skills.update({Tags_df['Name'][idx]: i})

# COMMAND ----------

# stem all of the skill names
for key, value in skills.items():
    skills[key] = clean_list(skills[key])

# COMMAND ----------

skills

# COMMAND ----------

# add skill dictionary as keywords
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_dict(skills)

# COMMAND ----------

matches = []

for i in Forums_and_Messages_df['Message Clean']:
    if i != '':
        if str(i) != 'nan':
            if keyword_processor.extract_keywords(i) != []:
                matches.append(keyword_processor.extract_keywords(i))
            else:
                matches.append(np.nan)
        else:
            matches.append(np.nan)
    else:
        matches.append(np.nan)

# join back into a dataframe
Forums_and_Messages_df['Name'] = matches

# COMMAND ----------

# drop rows where no skills were found
Forums_and_Messages_df = Forums_and_Messages_df.dropna(subset = ['Name'])

# COMMAND ----------

# Split the values of a column and expand so the new DataFrame has one split value per row.
# Filters rows where the column is missing
def tidy_split(df, column, sep=',', keep=False):
    indexes = list()
    new_values = list()
    #df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    new_df[column] = new_df[column].str.strip()
    new_df[column] = new_df[column].str.replace('[','').str.replace(']','').str.replace("'",'')
    return new_df

# COMMAND ----------

# explode skill list into rows
Forums_and_Messages_Exploded = tidy_split(Forums_and_Messages_df[['UserId', 'Name']], 'Name').reset_index(drop=True)
Forums_and_Messages_Exploded.head(5)

# COMMAND ----------

DiscScores = Forums_and_Messages_Exploded.groupby(['UserId', 'Name']).size().reset_index(name = 'Count')
DiscScores.head(5)                                                                                                     

# COMMAND ----------

DiscScores['DiscScore'] = DiscScores.groupby('Name')['Count'].rank(pct=True)
DiscScores = DiscScores.drop(['Count'], axis = 'columns')

# COMMAND ----------

# convert competition scores and kernel scores to pandas dfs
CompScores_df = ks.DataFrame(CompScores).toPandas()
KernelScores_df = ks.DataFrame(KernelScores).toPandas()

# COMMAND ----------

DiscScores.head(3)

# COMMAND ----------

CompScores_df.head(3)

# COMMAND ----------

KernelScores_df.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overall Skill Score
# MAGIC Combine kernel score, competition score and dataset scores into a single skill score heuristic.

# COMMAND ----------

CompScores_df['UserId'] = CompScores_df['UserId'].astype(str)
KernelScores_df['UserId'] = KernelScores_df['UserId'].astype(str)

SkillScores = CompScores_df.merge(KernelScores_df, how = 'outer', on = ['UserId', 'Name'])
SkillScores = SkillScores.merge(DiscScores, how = 'outer', on = ['UserId', 'Name'])
SkillScores = SkillScores.fillna(0)

SkillScores['SkillScore'] = (SkillScores['CompScore'] + SkillScores['KernelScore'] + SkillScores['DiscScore'])/3

# COMMAND ----------

SkillScores.sort_values('SkillScore', ascending = False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expertise Finding
# MAGIC Incorporating both skill scores and overall expertise scores for ranking skill based expert searches

# COMMAND ----------

df['UserId'] = df['UserId'].astype(str)
Users_and_Skills = df.drop('KernelScore', axis = 'columns').merge(SkillScores, on = 'UserId', how = 'left').fillna(0)
Users_and_Skills['CombinedScore'] = (Users_and_Skills['ExpertiseScore'] * 0.25) + (Users_and_Skills['SkillScore'] * 0.75)

# COMMAND ----------

Users_and_Skills[(Users_and_Skills['UserId']=='808') & ((Users_and_Skills['Name']=='linear regression') | (Users_and_Skills['Name']=='neural networks'))][['UserId', 'DisplayName', 'Name', 'DiscScore']]

# COMMAND ----------

Users_and_Skills.head(5)

# COMMAND ----------

def ExpertiseFinding(skill):
  Experts = Users_and_Skills[Users_and_Skills['Name'] == skill.lower()]
  Experts = Experts[['DisplayName', 'ExpertiseScore', 'SkillScore', 'CombinedScore', 'PerformanceTier', 'NumOrgs', 'KernelsCreated', 'NumFollowers', 'TotalComp', 'TotalDisc', 'TotalScript', 'TotalMedals', 'Ranked']]
  Experts = Experts.rename(columns={'DisplayName': 'Name', 'SkillScore': skill.title() +' Score',
                                    'ExpertiseScore':'Expertise Score',
                                    'CombinedScore': 'Combined Score',
                                    'PerformanceTier':'Performance Tier', 
                                    'NumOrgs':'# of Orgs',
                                    'KernelsCreated':'# of Kernels Created', 
                                    'NumFollower':'# of Followers', 
                                    'TotalComp':'# of Competition Medals',
                                    'TotalDisc':'# of Discussion Medals',
                                    'TotalScript':'# of Script Medals',
                                    'TotalMedals':'# of Total Medals'
                                   })
  Experts.iloc[0:,4:] = Experts.iloc[0:,4:].astype('int')
  Experts = Experts.sort_values(['Combined Score'], ascending=[False]).reset_index(drop=True)
  Experts.index = Experts.index + 1
  Experts = Experts.head(25)
  return Experts

# COMMAND ----------

skill = 'time series'
ExpertiseFinding(skill)

# COMMAND ----------

# Kaggle Skills
display(Tags.select('Name'))