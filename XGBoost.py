import numpy as np
import sklearn as sk
import Xgboost as xgb
import pandas
import sqlite3
import csv
# import matplotlib.pyplot as plt



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
    conn = sqlite3.connect(":memory:")
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


def get_previous_batting_avg(table, lo, hi, year):
    results = "(SELECT AVG(AB/H) FROM ("+ table +") WHERE yearID>="+str(lo)+" AND yearID<"+str(hi) + "AND playerID in " + get_season_stats(table, year) + ".playerID) as prevAvg"
