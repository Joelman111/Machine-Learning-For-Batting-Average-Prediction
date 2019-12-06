import argparse
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


def set_min_abs(table, n):
    """ Remove the players fewer than n ABs
    Input:
        table (str): SQL string representing the table we will query from.
    Output:
        results (str): SQL query to execute.
    """
    results = "SELECT * FROM ("+table+") WHERE AB>="+str(n)
    return results
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

parser = argparse.ArgumentParser(description='Predict a players batting average for a given year')
parser.add_argument("Player name", nargs=1, type=str,
                    help="enter a player name (First Last)")
parser.add_argument("Year", nargs=1, type=int, default=2018,
                    help="Enter a year valid for player")

args = parser.parse_args()

names = vars(args)["Player name"][0].split(" ")
firstName = names[0]
lastName = names[1]

year = vars(args)["Year"][0]

conn = sqlite3.connect("baseball.db")

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

def get_relevant_features(table, lo, year, firstName, lastName):
    tf = "SELECT H*1.0/AB as BA, AVGBA, PrevBA, yearID - BirthYear as age, SO*1.0/AB as SORate, BB*1.0/AB as WalkRate FROM (" + get_more_features(table, lo, year) + ") as A, People WHERE pid = People.playerID AND People.nameFirst = \"" + firstName + "\" AND People.nameLast LIKE \"" + lastName + "\""
    return tf

model = pickle.load(open("XGBModel", "rb"))

c = conn.cursor().execute(get_relevant_features(set_min_abs("batting_stats", 100), year - 2, year, firstName, lastName))

df = pd.DataFrame(c.fetchall())
# df.columns = [i for i in c.description]
x = df.to_numpy()
X = x[:,1:]
Y = x[:,1]

y_pred = model.predict(X)

predictions = [value for value in y_pred]
print("Predicted: " + str(predictions[0]))
print("Actual: " + str(Y[0]))
print("Error: " + str(round(abs(Y[0]-predictions[0])/Y[0] * 100, 3)) + "%")
