import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import zipfile
from IPython.display import clear_output

# -------------------------------------------------------------------------- Recommend Function ----------------------------------------------------------------------------------------------

def get_recommends(book = ""):

  distance, recommended_books = model.kneighbors(df_user_title.loc[book, :].values.reshape(1, -1))
  recommended = df_user_title.index[recommended_books][0][1:]

  return print(' First Book = ',recommended[0],'\n',
  				'Second Book = ',recommended[1],'\n',
  				'Third Book = ',recommended[2],'\n',
  				'Forth Book = ',recommended[3])

# ------------------------------------------------------------------------- Importing Files -------------------------------------------------------------------------------------------------

!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# ------------------------------------------------------------------------- CSV to DataFrame ------------------------------------------------------------------------------------------------

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# ------------------------------------------------------------------------- Data Management  ------------------------------------------------------------------------------------------------

relevant_users = df_ratings['user'].value_counts() > 200
relevant_users = relevant_users[relevant_users == True].index

df_ratings = df_ratings[df_ratings['user'].isin(relevant_users)]

df_ratings_books = df_ratings.merge(df_books, on = 'isbn')

relevant_books = df_ratings_books['isbn'].value_counts() > 100
relevant_books = relevant_books[relevant_books == True].index

df_ratings_books = df_ratings_books[df_ratings_books['isbn'].isin(relevant_books)]

df_ratings_books.drop_duplicates(['user','isbn'], inplace=True)

df_user_title = df_ratings_books.pivot_table(columns = 'user', index = 'title', values = 'rating')
df_user_title.fillna(0, inplace = True)

df_user_title_sparse = csr_matrix(df_user_title)

model = NearestNeighbors(metric="cosine",algorithm="brute", p=2)
model.fit(df_user_title_sparse)

# -------------------------------------------------------------------------- Recommendations ------------------------------------------------------------------------------------------------

get_recommends(book = "Where the Heart Is (Oprah's Book Club (Paperback))")