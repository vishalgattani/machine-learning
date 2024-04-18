from linear_regression import MyLinearRegression

from split_dataset import train_test_split as my_test_train_split

import pandas as pd
import numpy as np



data = pd.read_csv("datasets/Fifa_23_Players_Data.csv")
df = data[['Attacking Work Rate',
       'Defensive Work Rate', 'Pace Total', 'Shooting Total', 'Passing Total',
       'Dribbling Total', 'Defending Total', 'Physicality Total', 'Crossing',
       'Finishing', 'Heading Accuracy', 'Short Passing', 'Volleys',
       'Dribbling', 'Curve', 'Freekick Accuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'Sprint Speed', 'Agility', 'Reactions', 'Balance',
       'Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'Standing Tackle', 'Sliding Tackle',
       'Goalkeeper Diving', 'Goalkeeper Handling', ' GoalkeeperKicking',
       'Goalkeeper Positioning', 'Goalkeeper Reflexes', 'ST Rating',
       'LW Rating', 'LF Rating', 'CF Rating', 'RF Rating', 'RW Rating',
       'CAM Rating', 'LM Rating', 'CM Rating', 'RM Rating', 'LWB Rating',
       'CDM Rating', 'RWB Rating', 'LB Rating', 'CB Rating', 'RB Rating',
       'GK Rating','Overall']]

print(df.describe())