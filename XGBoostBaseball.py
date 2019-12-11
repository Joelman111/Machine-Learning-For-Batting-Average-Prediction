#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
import sqlite3
import math
import csv
import time


# In[2]:


def set_min_abs(table, n):
    """ Remove the players fewer than n ABs
    Input:
        table (str): SQL string representing the table we will query from.
    Output:
        results (str): SQL query to execute.
    """
    results = "SELECT * FROM ("+table+") WHERE AB>="+str(n)
    return results

def load_from_filepath(conn, batting_filepath, people_filepath):
    """ Load baseball data in the two files as tables into an in-memory SQLite database
    Input:
        conn (sqlite3.Connection) : Connection object corresponding to the database; used to perform SQL commands.
        batting_filepath (str) : absolute/relative path to Batting.csv file
        people_filepath (str) : absolute/relative path to People.csv file
    Output:
        None
    """

    c = conn.cursor()

    # open the csv file in Microsoft Excel (or any text editor) to determine the names and types of each column
    c.execute('''CREATE TABLE batting_stats (
                    playerID TEXT,
                    yearID INTEGER,
                    stint INTEGER,
                    teamID TEXT,
                    lgID TEXT,
                    G INTEGER,
                    AB INTEGER,
                    R INTEGER,
                    H INTEGER,
                    D INTEGER,
                    T INTEGER,
                    HR INTEGER,
                    RBI INTEGER,
                    SB INTEGER,
                    CS INTEGER,
                    BB INTEGER,
                    SO INTEGER,
                    IBB INTEGER,
                    HBP INTEGER,
                    SH INTEGER,
                    SF INTEGER,
                    GIDP INTEGER
                    )''')
    c.execute('''CREATE TABLE people (
                    playerID TEXT PRIMARY KEY,
                    birthYear INTEGER,
                    birthMonth INTEGER,
                    birthDay INTEGER,
                    birthCountry TEXT,
                    birthState TEXT,
                    birthCity TEXT,
                    deathYear INTEGER,
                    deathMonth INTEGER,
                    deathDay INTEGER,
                    deathCountry TEXT,
                    deathState TEXT,
                    deathCity TEXT,
                    nameFirst TEXT,
                    nameLast TEXT,
                    nameGiven TEXT,
                    weight INTEGER,
                    height INTEGER,
                    bats TEXT,
                    throws TEXT,
                    debut TEXT,
                    finalGame TEXT,
                    retroID TEXT,
                    bbrefID TEXT
                    )''')

    #collect the data for each row, then insert into the database
    with open(batting_filepath,'r') as fin:
        info = csv.DictReader(fin)
        to_insert = [(row['playerID'], row['yearID'], row['stint'],
                      row['teamID'], row['lgID'], row['G'], row['AB'],
                      row['R'], row['H'], row['2B'], row['3B'], row['HR'],
                      row['RBI'], row['SB'], row['CS'], row['BB'],
                      row['SO'], row['IBB'], row['HBP'], row['SH'],
                      row['SF'], row['GIDP']) for row in info]

    c.executemany("""INSERT INTO batting_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?)""", to_insert)

    with open(people_filepath,'r') as fin:
        info = csv.DictReader(fin)
        to_insert = [(row['playerID'], row['birthYear'], row['birthMonth'],
                      row['birthDay'], row['birthCountry'], row['birthState'],
                      row['birthCity'], row['deathYear'], row['deathMonth'],
                      row['deathDay'], row['deathCountry'], row['deathState'],
                      row['deathCity'], row['nameFirst'], row['nameLast'],
                      row['nameGiven'], row['weight'], row['height'], row['bats'],
                      row['throws'], row['debut'], row['finalGame'], row['retroID'],
                      row['bbrefID']) for row in info]

    c.executemany("""INSERT INTO people VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?)""", to_insert)

    conn.commit()

    print("Databases loaded!")


# Now we can call that function to create the databases we will use throughout the rest of this tutorial!

def load_batting_data():
    conn = sqlite3.connect("baseball.db")
    conn.text_factory = str

    #if you did not move you folders to the same folder as this tutorial,
    #you can change these to the correct filepath for your system
    load_from_filepath(conn, 'baseballdatabank-2019.2/baseballdatabank-2019.2/core/Batting.csv', 'baseballdatabank-2019.2/baseballdatabank-2019.2/core/People.csv')
    return conn

conn = load_batting_data()

def get_season_stats(table, n):
    """ Returns query that retrieves the season statistics for year n.
    Input:
        table (str): SQL string representing the table we will query from.
        n: The year we want statistics from.
    Output:
        results (str): SQL query to execute.
    """
    results = "SELECT * FROM ("+table+") WHERE yearID="+str(n)
    return results

def get_season_players(table, n):
     results = "SELECT playerID FROM ("+table+") WHERE yearID="+str(n)
     return results

def get_subset_seasons(table, lo, hi):
    """ Retrieves the statistics from the input table for years n where: lo <= n < hi.
    Input:
        table (str): SQL string representing the table we will query from.
        lo: The lower year bound (inclusive)
        hi: The upper year bound (exclusive)
    Output:
        results (str): SQL query to execute.
    """
    results = "SELECT * FROM ("+table+") WHERE yearID>="+str(lo)+" AND yearID<"+str(hi)
    return results


# In[3]:


def display_results(c):
    # Function for displaying the result of a query as a Dataframe
    # No need to worry about this
    df = pd.DataFrame(c.fetchall())
    df.columns = [i[0] for i in c.description]
    display(df)


# In[4]:


def get_features(table, lo, hi, year):
    avgTable = "SELECT AVG(H * 1.0 /AB) as ave, playerID as pid FROM ("+ table +") WHERE yearID>="+str(lo)+" AND yearID<"+str(hi) + " GROUP BY playerID"
    playerIDs = "SELECT a.ave, a.pid FROM (" + avgTable + ") as a WHERE a.pid in (" + get_season_players(table, year) + ")"
    return playerIDs

def get_previous_year_BA(table, year):
    prevYear = "SELECT H*1.0/AB as BA, playerID as pid FROM ("+ table +") WHERE yearID="+str(year - 1)+" and playerID in (" + get_season_players(table, year)+")"
    return prevYear

def get_two_features(table, lo, year):
    tf = "SELECT py.pid as pid, py.BA as PrevBA, "+ str(year) +" as yr, a.ave as AVGBA FROM (" + get_previous_year_BA(table, year) + ") as py, (" + get_features(table,lo,year,year)+") as a WHERE py.pid = a.pid AND a.ave = a.ave AND py.BA = py.BA"
    return tf

def get_more_features(table, lo, year):
    tf = "SELECT * FROM (" + get_two_features(table, lo, year) + ") as A INNER JOIN (" + table + ") as B on B.playerId = A.pid AND B.yearID = A.yr"
    return tf

def get_relevant_features(table, lo, year):
    tf = "SELECT H*1.0/AB as BA, AVGBA, PrevBA, yearID - BirthYear as age, SO*1.0/AB as SORate, BB*1.0/AB as WalkRate FROM (" + get_more_features(table, lo, year) + ") as A, People WHERE pid = People.playerID"
    return tf


# In[ ]:




# c = conn.cursor().execute(get_season_stats("batting_stats", 2018))
# c = conn.cursor().execute(get_features("batting_stats", 2000, 2017, 2017))
# display_results(c)
c = conn.cursor().execute(get_relevant_features(set_min_abs("batting_stats", 100), 2000, 2018))
# display_results(c)

df = pd.DataFrame(c.fetchall())
df.columns = [i for i in c.description]
x = df.to_numpy()
for i in range(2010, 2018):
    c = conn.cursor().execute(get_relevant_features(set_min_abs("batting_stats", 100), 2000, i))
    x = np.concatenate((x, pd.DataFrame(c.fetchall()).to_numpy()))

X = x[:,1:]
Y = x[:,1]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# xgb.DMatrix(x)



# In[ ]:


start_time = time.time()
# fit model no training data
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
elapsed_time = time.time() - start_time
print(elapsed_time)


with open('XGBModel','rb') as f:
    pickle.dump(model, f)

# In[ ]:


y_pred = model.predict(X_test)
predictions = [value for value in y_pred]
# print(y_pred)
# print(y_test)
# print(len(predictions))
c = 0
for i in range(len(predictions)):
    c += round(abs(y_test[i]-predictions[i])/y_test[i] * 100, 3)
print(c/len(predictions))

rms = math.sqrt(sk.metrics.mean_squared_error(y_test, predictions))
print(rms)


# In[ ]:





# In[ ]:
