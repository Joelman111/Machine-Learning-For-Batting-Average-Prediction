import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))
with open('batting_stats.df', 'rb') as f:
	batting_stats = pickle.load(f)

with open('person_info.df', 'rb') as f:
	person_info = pickle.load(f)

aggF = {}
for k in batting_stats.dtypes.index:
	if (np.float64 == batting_stats.dtypes[k] or batting_stats.dtypes[k]==np.int64) and k!='yearID' and k!='stint':
		aggF[k] = 'sum'
	elif k!='yearID' and k!='playerID':
		aggF[k] = 'first'

#for col in batting_stats.columns:
#	print(col)
#print(batting_stats)
stints_combined = batting_stats.groupby(['playerID', 'yearID']).agg(aggF).reset_index()
person_info = person_info[['playerID', 'birthYear']].set_index('playerID')
with_birthYear = stints_combined.join(person_info, on='playerID')
with_birthYear['age'] = with_birthYear['yearID'] - with_birthYear['birthYear']
with_birthYear['NL'] = (with_birthYear['lgID'] == 'NL').astype(np.int64)
with_birthYear['BA'] = (with_birthYear['H']/with_birthYear['AB'])
with_birthYear['BB'] = (with_birthYear['BB']/with_birthYear['AB'])
with_birthYear['SO'] = (with_birthYear['SO']/with_birthYear['AB'])
with_birthYear = with_birthYear.replace([np.inf, -np.inf], 1).fillna(0)
with_birthYear = with_birthYear[['playerID', 'yearID','age','NL','AB','BA','BB','SO']]

with_birthYear[['yearID','age','NL','AB','BA','BB','SO']] = scaler.fit_transform(with_birthYear[['yearID','age','NL','AB','BA','BB','SO']])

data = []
players = with_birthYear.groupby('playerID')
for n, g in players:
	for i in range(0,len(g)-3):
		if i+3<len(g):
			v = g.iloc[i:i+3].to_numpy()[:,1:].tolist()
			l = g.iloc[i+3].to_numpy()[5].tolist()
			data.append([v,l])
		else:
			break

with open("lstm_data", 'wb') as f:
	pickle.dump(data, f)