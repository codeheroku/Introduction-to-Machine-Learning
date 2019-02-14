import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File

##Step 2: Select Features

##Step 3: Create a column in DF which combines all selected features

##Step 4: Create count matrix from this new combined column

##Step 5: Compute the Cosine Similarity based on the count_matrix

movie_user_likes = "Avatar"

## Step 6: Get index of this movie from its title

## Step 7: Get a list of similar movies in descending order of similarity score


## Step 8: Print titles of first 50 movies