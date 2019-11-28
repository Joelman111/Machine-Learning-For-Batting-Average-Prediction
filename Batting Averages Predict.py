import csv
import math
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from IPython.display import display
from sklearn.linear_model import LinearRegression
import random
import statistics



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


def load_batting_data():
    conn = sqlite3.connect(":memory:")
    conn.text_factory = str
    
    #if you did not move you folders to the same folder as this tutorial, 
    #you can change these to the correct filepath for your system
    load_from_filepath(conn, 'baseballdatabank-2019.2/baseballdatabank-2019.2/core/Batting.csv', 'baseballdatabank-2019.2/baseballdatabank-2019.2/core/People.csv')
    return conn

conn = load_batting_data()


def display_results(c):
    # Function for displaying the result of a query as a Dataframe
    # No need to worry about this
    df = pd.DataFrame(c.fetchall())
    df.columns = [i[0] for i in c.description]
    display(df)

def combine_stint():
	results = """SELECT *, bs.lgID FROM (SELECT playerID, yearID, SUM(AB) AS AB, SUM(H) AS H, SUM(BB) AS BB, SUM(SO) AS SO FROM batting_stats GROUP BY playerID, yearID) as input LEFT JOIN (SELECT * FROM batting_stats WHERE stint = 1) as bs ON input.playerID = bs.playerID AND input.yearID = bs.yearID"""
	return results


def extract_player_info(table):
    """ Returns a table with the real player names attached to the input table.
    Input:
        table (str): SQL string representing the table we will query from.
    Output:
        results (str): Query string that creates a table with names attached to stats
    """
    results = """CREATE TABLE revisedStats AS SELECT input.playerID as playerID, yearID-people.birthYear as age, yearID, CASE input.lgID WHEN 'NL' THEN 1 ELSE 0 END AS NL, CASE input.lgID WHEN 'AL' THEN 1 ELSE 0 END AS AL, AB,
    	IFNULL(ROUND(CAST(input.H AS float) / CAST(input.AB AS float), 3),0) AS BA, IFNULL(ROUND(CAST(input.BB AS float) / CAST(input.AB AS float), 3),0) AS BB, IFNULL(ROUND(CAST(input.SO AS float) / CAST(input.AB AS float), 3),0) AS SO 
    	FROM ("""+table+""") AS input LEFT JOIN people ON input.playerID = people.playerID WHERE yearID>=1989"""
    return results

def get_prev_year_stats():
    """ Remove the players fewer than n ABs
    Input:
        table (str): SQL string representing the table we will query from.
    Output:
        results (str): SQL query to execute.
    """
    results = """SELECT curr.yearID as yearID, curr.age as age, curr.NL as NL, curr.AL as AL, past.AB as lastAB, past.BA as lastBA, past.BB as lastBB, past.SO as lastSO, curr.BA as BA FROM revisedStats curr INNER JOIN revisedStats past ON curr.playerID = past.playerID AND curr.yearID = past.yearID+1 WHERE curr.AB>=200"""
    return results

#get seasons 1989-2018
basic_info = extract_player_info(combine_stint())
conn.cursor().execute(basic_info)

#attach player names to result
q2 = get_prev_year_stats()
c = conn.cursor().execute(q2)

df = pd.DataFrame(c.fetchall())
df.columns = [i[0] for i in c.description]
data = df[["yearID","age","NL","AL","lastAB","lastBA","lastBB","lastSO","BA"]].values.tolist()

"""
random.shuffle(data)
n = len(data)
split = int(n*.8)
features = [[v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]] for v in data]
BAs = [[v[8]] for v in data]

train_features = np.array(features[:split])
train_output = np.array(BAs[:split])

test_features = np.array(features[split:])
test_output = np.array(BAs[split:])


model = LinearRegression(fit_intercept=True, normalize=False)

# Now we fit it with the data
model.fit(train_features, train_output)
print(model.score(train_features, train_output))
print(model.intercept_, model.coef_)

pred_ba = model.predict(test_features)


"""
errors = []
for i in range(100):
	random.shuffle(data)
	n = len(data)
	split = int(n*.8)
	features = [[v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]] for v in data]
	BAs = [[v[8]] for v in data]

	train_features = np.array(features[:split])
	train_output = np.array(BAs[:split])

	test_features = np.array(features[split:])
	test_output = np.array(BAs[split:])


	model = LinearRegression(fit_intercept=True, normalize=False)

	# Now we fit it with the data
	model.fit(train_features, train_output)

	pred_ba = model.predict(test_features)

	errors.append((abs(pred_ba-test_output)/test_output *100).mean())

print(statistics.mean(errors))
print(statistics.stdev(errors))
print(min(errors))
print(max(errors))

"""
# Get the real batting average using the functions we made
c = conn.cursor().execute(get_mean_ba_per_year(set_min_abs(get_ba(get_season_stats("batting_stats", 2018)), 200)))

df = pd.DataFrame(c.fetchall())
df.columns = [i[0] for i in c.description]
real_ba = df[["BA"]].values.tolist()[0][0]

print("The model predicted: ",str(pred_ba))
print("The real mean batting average was: ", str(real_ba))
print("This is a difference of: ", str(abs(real_ba-pred_ba)), ", which is an error of",str(round(abs(real_ba-pred_ba)/real_ba * 100, 3)), "%")


# 4.58% error. Hmm, not too bad considering the huge variance in the data we had to begin with. What could we do to reduce the error in the model? One thing we can try to restricting our data to the later years, which will help remove some of the variance we can see from the earlier days of baseball - we can justify this decision to ignore old statistics because the game has advanced so much since that time and the stats were not quite as well-kept back then. Let's try selecting data only from the last 20 years. We can use our `get_subset_seasons` function we created back in Part 1.

# In[13]:


#compute mean batting averages for each year 1998-2017 (min 200 ABs)
c = conn.cursor().execute(get_mean_ba_per_year(set_min_abs(get_ba(get_subset_seasons("batting_stats", 1998, 2018)), 200)))

df = pd.DataFrame(c.fetchall())
df.columns = [i[0] for i in c.description]
data = df[["yearID","BA"]].values.tolist()
years = [v[0] for v in data]
BAs = [v[1] for v in data]

#convert to numpy arrays (vectors)
x = np.array(years).reshape(-1, 1)
y = np.array(BAs)

model = LinearRegression(fit_intercept=True, normalize=False)

# Now we fit it with the data
model.fit(x, y)

# Predict for 2018
x1 = np.array([[2018]])
pred_ba = model.predict(x1)[0]

# Get the real batting average using the functions we made
c = conn.cursor().execute(get_mean_ba_per_year(set_min_abs(get_ba(get_season_stats("batting_stats", 2018)), 200)))

df = pd.DataFrame(c.fetchall())
df.columns = [i[0] for i in c.description]
real_ba = df[["BA"]].values.tolist()[0][0]

print("The model predicted: ",str(pred_ba))
print("The real mean batting average was: ", str(real_ba))
print("This is a difference of: ", str(abs(real_ba-pred_ba)), ", which is an error of",str(round(abs(real_ba-pred_ba)/real_ba * 100, 3)), "%")


# ### Nonlinear Regression
# By reducing the time frame we are looking at, we have greatly reduced the error in our model. And this makes sense as you can see the slight downward trend occurring over the past 20 years. Is there any way we can do even better than this? Another thing we can try is using a polynomial model, which is like linear regression except we expand the feature vector to include the original feature raised to the ith power for i=1 to d (where d is what's known as a hyperparameter, we get to choose it so that it best fits our needs). The implementation is very similar to the linear regression model: first we create our model, then we fit the data, and finally, we predict. There are only differences is that we expand our feature vector to have more than one feature. It's as simple as that!

# In[14]:


d = 10

#compute mean batting averages for each year 1998-2017 (min 200 ABs)
c = conn.cursor().execute(get_mean_ba_per_year(set_min_abs(get_ba(get_subset_seasons("batting_stats", 1998, 2018)), 200)))

df = pd.DataFrame(c.fetchall())
df.columns = [i[0] for i in c.description]
data = df[["yearID","BA"]].values
years = data[:,0]
BAs = data[:,1]

x = np.array([years**i for i in range(1,d)]).T
y = np.array(BAs)

#set up the model, we will normalize here so as to not overflow (high degree polynomials get large quickly)
model = LinearRegression(fit_intercept=True, normalize=True)
model.fit(x, y)

# Predict for 2018
x1 = np.array([np.array(2018.0)**i for i in range(1,d)]).T.reshape(1, -1)
pred_ba = model.predict(x1)[0]

# Get the real batting average using the functions we made
c = conn.cursor().execute(get_mean_ba_per_year(set_min_abs(get_ba(get_season_stats("batting_stats", 2018)), 200)))

df = pd.DataFrame(c.fetchall())
df.columns = [i[0] for i in c.description]
real_ba = df[["BA"]].values.tolist()[0][0]

print("The model predicted: ",str(pred_ba))
print("The real mean batting average was: ", str(real_ba))
print("This is a difference of: ", str(abs(real_ba-pred_ba)), ", which is an error of",str(round(abs(real_ba-pred_ba)/real_ba * 100, 3)), "%")


# Wow! It looks like this method really provided an effective model for predicting next season's mean batting average. By choosing a nonlinear model, we are more accurately able to account for the curves we see in the data, and therefore, we can better estimate the future outcomes based solely on the year. We've come very far since our first attempt at estimating the mean batting average in the 2018 season!

# ## Conclusion
# 
# This tutorial went through the process of loading a baseball statistics database, manipulating and creating statistics, and applying machine learning techniques using Scikit-Learn to solve interesting problems.
# 
# However, this is just a brief glimpse into the use of statistics and machine learning techniques in sports; a good starting point. Obviously, in the real world, data scientists and statisticians attempt to answer more meaningful questions that have a more direct impact to teams such as "how will Player X perform next season?" and "what set of statisitics are useful features in predicting a team's success?". These questions carry a lot of weight and are literally million-dollar problems in today's sporting world. 
# 
# If these challenges interest you, check out the other statistics available in the database folder you downloaded earlier. Additionally, [Baseball-Reference](https://www.baseball-reference.com/) has tons of MLB data and contains many, many more baseball stats. If you are interested in learning more about machine learning applied to sports, check out the articles [here](http://www.baseballdatascience.com/category/machine-learning/) as they are quite fascinating. Finally, if you want to learn more about the Scikit-Learn library, you can find more tutorials and documentation [here](https://scikit-learn.org/stable/documentation.html).
"""