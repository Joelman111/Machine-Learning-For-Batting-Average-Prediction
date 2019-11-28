


######### MAKE SURE TO INSTALL ALL OF THESE LIBRARIES ###########



import csv
import sqlite3
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LinearRegression
import random
import statistics

def load_batting_data():
	""" Create an in-memory SQLite database and load the baseball data
    Input:
        None
    Output:
        conn (sqlite3.Connection) : Connection object corresponding to the database; used to perform SQL commands.
    """

    # Create a new database that will be stored in memory
    # The conn object is the representation of the database in the code
    conn = sqlite3.connect(":memory:")
    conn.text_factory = str
    
    # Load the batting stats and personal info into the connection
    load_from_filepath(conn, 'baseballdatabank-2019.2/baseballdatabank-2019.2/core/Batting.csv', 'baseballdatabank-2019.2/baseballdatabank-2019.2/core/People.csv')
    return conn

def load_from_filepath(conn, batting_filepath, people_filepath):
    """ Load baseball data in the two files as tables into the given connection
    Input:
        conn (sqlite3.Connection) : Connection object corresponding to the database; used to perform SQL commands.
        batting_filepath (str) : absolute/relative path to Batting.csv file
        people_filepath (str) : absolute/relative path to People.csv file
    Output:
        None
    """
    
    # This is cursor object you will use to interact with the database
    # Cursors will hold the results of querying the database
    c = conn.cursor()

    # 'execute' executes the given SQL command (a string with SQL syntax)
    # This command creates a table named "batting_stats" with the given columns, each with the specified type
    # you can open the csv file in Microsoft Excel (or any text editor) to determine the names and types of each column
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
    # Same as above, but creates a table named "people" with different columns
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
    
    # Open the batting stats file
    with open(batting_filepath,'r') as fin:
    	# Read all the info into a dictionary
        info = csv.DictReader(fin)
        # For each row, load the columns. to_insert will be a list of tuples, which contain the column values of a particular row
        to_insert = [(row['playerID'], row['yearID'], row['stint'], 
                      row['teamID'], row['lgID'], row['G'], row['AB'],
                      row['R'], row['H'], row['2B'], row['3B'], row['HR'],
                      row['RBI'], row['SB'], row['CS'], row['BB'],
                      row['SO'], row['IBB'], row['HBP'], row['SH'], 
                      row['SF'], row['GIDP']) for row in info]
        
    # 'executemany' allows you to execute a list of commands at once
    # Insert each row into the batting_stats table we created
    c.executemany("""INSERT INTO batting_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                  ?, ?, ?, ?, ?, ?, ?, ?, ?)""", to_insert)
    
    # Same as above but for the people table
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
    
    # 'commit' commits the changes to the database. Must be called after altering the database
    conn.commit()
    
    print("Databases loaded!")


def display_results(c):
    """ Prints out the database or query results held in the given cursor
    Input:
        conn (sqlite3.Cursor) : Cursor object that contains the query results
    Output:
        None
    """

    # No need to worry about what this is actually doing
    df = pd.DataFrame(c.fetchall())
    df.columns = [i[0] for i in c.description]
    display(df)


def combine_stint():
	""" Return the query string that will combine the stints of all players into just a single stint for each season.
		Adds together season totals for all stints, keeps only the league the player played in to start the season.
    Input:
        None
    Output:
        None
    """
    # 'Select' gets the requested columns. * means get all columns, 'bs.lgID' means 'in table bs, get columns lgID'
    # 'From' says the Select statement applies to this table. Here, the first 'From' refers to the table generated in the parentheses (a subquery)
    # 'As' means I rename the object to the name that follows this As statement. E.g. the table generated from the subquery in parentheses is renamed to 'input'
    # 'Group By' means split the players into groups based on them having the same playerID and yearID
    # 'Left Join' means combine the columns of two seperate tables where the playerID and yearID match in tables
    # Results of this query will be a table containing, the playerID, the yearID, the lgID of the player's first stint, the sum of the player's ABs across all of his stints that season, the sum of the player's hits across all of his stints that season, ... for all relevant stats
	results = """SELECT *, bs.lgID FROM (SELECT playerID, yearID, SUM(AB) AS AB, SUM(H) AS H, SUM(BB) AS BB, SUM(SO) AS SO FROM batting_stats GROUP BY playerID, yearID) AS input LEFT JOIN (SELECT * FROM batting_stats WHERE stint = 1) as bs ON input.playerID = bs.playerID AND input.yearID = bs.yearID"""
	return results


def extract_player_info(table):
    """ Returns the query string that will create a table called 'revisedStats' that has the playerID, age, yearID, a binary column for NL, a binary column for AL, BA, walk rate, and strikeout rate for each player
    Input:
        table (str): SQL string representing the table we will query from.
    Output:
        results (str): Query string that creates a table with all the stats we will need
    """

    # Create a new table containing the stats we care about for the model (calculating the age, the AL/NL binary value, the batting average, the walk rate, the strikeout rate)
    # Only include years from 1989 to current year
    results = """CREATE TABLE revisedStats AS SELECT input.playerID as playerID, yearID-people.birthYear as age, yearID, CASE input.lgID WHEN 'NL' THEN 1 ELSE 0 END AS NL, CASE input.lgID WHEN 'AL' THEN 1 ELSE 0 END AS AL, AB,
    	IFNULL(ROUND(CAST(input.H AS float) / CAST(input.AB AS float), 3),0) AS BA, IFNULL(ROUND(CAST(input.BB AS float) / CAST(input.AB AS float), 3),0) AS BB, IFNULL(ROUND(CAST(input.SO AS float) / CAST(input.AB AS float), 3),0) AS SO 
    	FROM ("""+table+""") AS input LEFT JOIN people ON input.playerID = people.playerID WHERE yearID>=1989"""
    return results

def get_prev_year_stats():
    """ Returns the query string that will create a table of the current year, player's current age, player's current league, player's previous ABs, player's previous BA, player's previous walk rate, player's previous strikeout rate, and player's BA for this season (what we want to predict)
    Input:
        table (str): SQL string representing the table we will query from.
    Output:
        results (str): SQL query to execute.
    """

    # Use a left join to combine the revised stats table with itself (so we can get last year's stats and this year's stats together)
    # Left join on the playerID to match up the players, and also on the year and the previous year to get the stats of this player in the previous season
    # Only include players who ended up with more than 200 ABs in the prediction year
    results = """SELECT curr.yearID AS yearID, curr.age AS age, curr.NL AS NL, curr.AL AS AL, past.AB AS lastAB, past.BA AS lastBA, past.BB AS lastBB, past.SO AS lastSO, curr.BA AS BA FROM revisedStats curr INNER JOIN revisedStats past ON curr.playerID = past.playerID AND curr.yearID = past.yearID+1 WHERE curr.AB>=200"""
    return results


# Get a new loaded database
conn = load_batting_data()

# Get query string to create 'revisedStats' table as defined above
basic_info = extract_player_info(combine_stint())
conn.cursor().execute(basic_info)

# Combine previous year's stats with prediction year to get all features and labels
q2 = get_prev_year_stats()
c = conn.cursor().execute(q2)

# Convert the SQLite database to a pandas dataframe so that we can work with it in sci-kitlearn
# Really just massages the data into a different form of the same stats
df = pd.DataFrame(c.fetchall())
df.columns = [i[0] for i in c.description]
data = df[["yearID","age","NL","AL","lastAB","lastBA","lastBB","lastSO","BA"]].values.tolist()


# Code for training a single LinearRegression model. Uncomment code below and comment the for loop after to run this


"""

# Randomly shuffle the data (I might get rid of this since it is kinda unrealistic since you won't be using data from more recent year to predict outcomes from previous years)
random.shuffle(data)
n = len(data)

# Index where we will split data into a training set and a test set (training set being 80% of the data)
splitIdx = int(n*.8)

# Features
features = [[v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]] for v in data]

# Prediction labels
BAs = [[v[8]] for v in data]

# Get the training data and create a numpy array
train_features = np.array(features[:split])
train_output = np.array(BAs[:split])

# Get the test data and create a numpy array
test_features = np.array(features[split:])
test_output = np.array(BAs[split:])

# Create a new LinearRegression model
model = LinearRegression(fit_intercept=True, normalize=False)

# Now we fit it with the data
model.fit(train_features, train_output)

# Prints internal variables of the trained model (gives insight into what is happening)
print(model.score(train_features, train_output))
print(model.intercept_, model.coef_)

# Create an array of predictions
pred_ba = model.predict(test_features)

# Prints the average error of the model on the test set
print((abs(pred_ba-test_output)/test_output *100).mean())
"""

# Code for training the model many different times with different training data.

# Keep track of average errors to get statistics about the model
errors = []
for i in range(100):

	# Same as training code above
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
	model.fit(train_features, train_output)

	pred_ba = model.predict(test_features)

	errors.append((abs(pred_ba-test_output)/test_output *100).mean())

# Print mean and standard deviation of average error for over 100 models
print(statistics.mean(errors))
print(statistics.stdev(errors))

# Print min/max average error of the 100 models trained in the loop
print(min(errors))
print(max(errors))




######## IGNORE BELOW THIS LINE #########
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