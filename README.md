# Movies_ETL

# Goals
 - Create an ETL that will take the Wikipedia data, kaggle data and ratings and transform it into usable data for coders.  

# Process
 - We removed duplicate columns.
 - Transformed the times into Unix format to be consistant. 
 - Transformed the dollar quntities into a consistant format to be able to run analysis.
 - Any column that is missing 90% of the data will be deleted.
 - Any missing data will be filled with NaN.
 
# Assumptions
 1. We are assuming that there will be either a 'Director' or 'Directed by' column.
 2. 
