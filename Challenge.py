import pandas as pd
import numpy as np
import json
import re 
from sqlalchemy import create_engine
from config import db_password
import time

file_dir='C:/Users/briordan/Class/Movies_ETL/'
def ETL(wiki,kaggle,rating):
    # Pull the Json
    with open(f'{file_dir}{wiki}', mode='r') as file:
        wiki_movies_raw = json.load(file)
    # Make the json file a DataFrame
    wiki_movies_df = pd.DataFrame(wiki_movies_raw)
    # Find all movies that have a director in 'Director' or 'Directed By' and there is a 'imdb_link'
    try: 
        wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]
    except:
        wiki_movies = [movie for movie in wiki_movies_raw
               if 'imdb_link' in movie
                   and 'No. of episodes' not in movie]
    # Make this into a DataFrame
    wiki_movies_df = pd.DataFrame(wiki_movies)
    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        # combine alternate titles into one list
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
            'Hangul','Hebrew','Hepburn','Japanese','Literally',
            'Mandarin','McCune–Reischauer','Original title','Polish',
            'Revised Romanization','Romanized','Russian',
            'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles
        
        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
        try:
            change_column_name('Adaptation by', 'Writer(s)')
        except: 
            pass
        try:
            change_column_name('Country of origin', 'Country')
        except:
            pass
        try:
            change_column_name('Directed by', 'Director')
        except:
            pass
        try:
            change_column_name('Distributed by', 'Distributor')
        except:
            pass
        try:
            change_column_name('Edited by', 'Editor(s)')
        except:
            pass
        try:
            change_column_name('Length', 'Running time')
        except:
            pass
        try:
            change_column_name('Original release', 'Release date')
        except:
            pass
        try:
            change_column_name('Music by', 'Composer(s)')
        except:
            pass
        try:
            change_column_name('Produced by', 'Producer(s)')
        except:
            pass
        try:
            change_column_name('Producer', 'Producer(s)')
        except:
            pass
        try:
            change_column_name('Productioncompanies ', 'Production company(s)')
        except:
            pass
        try:
            change_column_name('Productioncompany ', 'Production company(s)')
        except:
            pass
        try:
            change_column_name('Released', 'Release Date')
        except:
            pass
        try:
            change_column_name('Release Date', 'Release date')
        except:
            pass
        try:
            change_column_name('Screen story by', 'Writer(s)')
        except:
            pass
        try:
            change_column_name('Screenplay by', 'Writer(s)')
        except:
            pass
        try:
            change_column_name('Story by', 'Writer(s)')
        except:
            pass
        try:
            change_column_name('Theme music composer', 'Composer(s)')
        except:
            pass
        try:
            change_column_name('Written by', 'Writer(s)')
        except:
            pass  
    
        return movie
    
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    # Create a DataFrame from the function
    wiki_movies_df = pd.DataFrame(clean_movies)
    # Find the IMDB_if out of the link
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    # remove all duplicates fo new column
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
    # only show the columns with a 90% fill rate
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() 
                        < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
    # Drop all the NaN info in the 'Box Office' column
    box_office = wiki_movies_df['Box office'].dropna()
    # remove spaces from Box Office
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
    # Regular expression to find the money in the box office
    form_one = r'\$\d+\.?\d*\s*[mb]illion'
    # Regular expression to find the normal xxx,xxx,xxx numbers
    form_two = r'\$\d{1,3}(?:,\d{3})+'
    # Create variables to capture form one and two
    matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)
    box_office[~matches_form_one & ~matches_form_two]
    # For box office ranges, we replace the hyphen with a dollar sign
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    # Make sure the last step did not change any of the form 1 data
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
    # Chagne the misspelled Millon as Million
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    # Exctract all the box office that match the egular expressions
    box_office.str.extract(f'({form_one}|{form_two})')
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan
    
        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
        
            # remove dollar sign and " million"
            s= re.sub(r'\$|\s|[a-zA-Z]','', s)
            # convert to float and multiply by a million
            value = float(s) * 10**6
            # return value
            return value
    
        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
        
            # remove dollar sign and " billion"
            s = re.sub(r'\$|\s|[a-zA-Z]','', s)
            # convert to float and multiply by a billion
            value = float(s) * 10**9
            # return value
            return value
    
        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

            # remove dollar sign and commas
            s = re.sub(r'\$|,','', s)
            # convert to float
            value = float(s)
            # return value
            return value
    
        # otherwise, return NaN
        else:
            return np.nan
        
    # extract the values from box_office
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    
    wiki_movies_df.drop('Box office', axis=1, inplace=True)
    
    try: 
        budget = wiki_movies_df['Budget'].dropna()
        # converts lists t strings
        budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
        # removes any values between a dollar sign and hyphen
        budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
        # using the Box office data, see what is left over
        matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
        matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
        # remove the citation bracket [] inside the budget
        budget = budget.str.replace(r'\[\d+\]\s*', '')
        # Same info as the box office data, but just replace the word with budget.  this is called Parse
        wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
        wiki_movies_df.drop('Budget', axis=1, inplace=True)
    except:
        pass
    try: 
        # For release date, make a variable that holds the non-null values converting lists to strings
        release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
        # find all the different forms of release dates
        date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
        date_form_two = r'\d{4}.[01]\d.[123]\d'
        date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
        date_form_four = r'\d{4}'
        # extract the dates
        # Make the new columns with the dates
        wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(
            f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
    except:
        pass
    try:
        # Do the same thing for Running time
        running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
        # extract the values
        running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
        # Turn the strings to numeric and use'coerce' to change the empty cells to NaN
        running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
        wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
        wiki_movies_df.drop('Running time', axis=1, inplace=True)
    except:
        pass

    
    kaggle_metadata = pd.read_csv(f'{file_dir}{kaggle}')
    # Keep rows where the adult column is False
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] =='False'].drop('adult', axis='columns')
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
    # convert all the numberics
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    # Change release date to date time
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])
   
    
    # Print out a list of columns to identify which ones are redundant.
    # Use Inner join to show all rows.
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
    # get the index for that row in order to drop it
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') 
                                     & (movies_df['release_date_kaggle'] < '1965-01-01')].index)
    # Change the lists in Language to tuples so the value_counts will work
    movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)
    # Drop the title_wiki, release_date_wiki, Language and Production
    movies_df.drop(columns=['title_wiki', 'release_date_wiki', 'Language', 'Production company(s)'], inplace=True)
    # Create function that fills in the missing date for a column pair and then drops redundant column
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column], axis=1)
        df.drop(columns=wiki_column, inplace=True)
    # Run the function
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
    movies_df
    # Check to see if any columns only have one value
    for col in movies_df.columns:
        lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
        value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
        num_values = len(value_counts)
        if num_values == 1:
            bad_columns = col
            movies_df[bad_columns].value_counts(dropna=False)
    # Reorder the data in the following way
    # Identifying information (IDs, titles, URLs, etc.)
    # Quantitative facts (runtime, budget, revenue, etc.)
    # Qualitative facts (genres, languages, country, etc.)
    # Business data (production companies, distributors, etc.)
    # People (producers, director, cast, writers, etc.)
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]
    # Rename the columns to be consistant
    movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)
    
    ratings = pd.read_csv(f'{file_dir}{rating}')
    # Assign it to the timestamp column
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    # Include the rating counts for each movie
    # Groupby the movield and rating to get a count for each group
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()
    # rename the userID column to count
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                .rename({'userId':'count'}, axis=1) 
    # Pivot this data so that the movieID is the index, the columns will be rating values and the rows will be the counts
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                .rename({'userId':'count'}, axis=1) \
                .pivot(index='movieId',columns='rating', values='count')
    # Make the columns easier to read
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
    # Merge the data sets.  Use a left merge so the movies_df stays in tact
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
    # Fill in missing rating values in with NaN
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)
    
    # Give the link to the postres
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
    # Create the engine
    engine = create_engine(db_string)
    # Move the movies table to sql
    movies_df.to_sql(name='movies', con=engine, if_exists='replace')
    # create a variable for the number of rows imported
    rows_imported = 0
    # get the start_time from time.time()
    start_time = time.time()
    for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):

        # print out the range of rows that are being imported
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='append')

        # increment the number of rows imported by the chunksize
        rows_imported += len(data)
    
        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')

ETL('wikipedia.movies.json','movies_metadata.csv','ratings.csv')