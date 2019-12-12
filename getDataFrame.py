import pandas as pd
import pickle


batting_stats = pd.read_csv('baseballdatabank-2019.2/baseballdatabank-2019.2/core/Batting.csv')
person_info = pd.read_csv('baseballdatabank-2019.2/baseballdatabank-2019.2/core/People.csv')

with open("batting_stats.df", 'wb') as f:
	pickle.dump(batting_stats, f)

with open("person_info.df", 'wb') as f:
	pickle.dump(person_info, f)

