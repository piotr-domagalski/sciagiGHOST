import pandas as pd

### 1. Creating, Reading and Writing ###
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})

pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')

wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.shape
wine_reviews.head()

### 2. Indexing, Selecting & Assigning ###
reviews.country
reviews['country']

#po indeksie
reviews.iloc[0]          # zerowy    rzad
reviews.iloc[:, 0]       #wszystkie  rzedy zerowej kolumny
reviews.iloc[2:4, 0]     #2 i 3      rzad  zerowej kolumny
reviews.iloc[[1,2,5], 0] #wybrane    rzedy zerowej kolumny
reviews.iloc[-5:]        #ostatnie 5 rzedow 

#po nazwach kolumn/wierszy
reviews.loc[0, 'country']
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
reviews.loc[:, 'points':'taster_twitter_handle'] # w przypadku loc operator : wybiera włącznie z ostatnim elementem, nie jak w pythonie

#warunkowe
reviews.country == 'Italy' #porównanie serii z wartością zwraca serię booleanów
reviews[reviews.country == 'Italy'] #którą można użyć do indeksowania
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
#warunkowe wbudowane
reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.isnull()]
reviews.loc[reviews.price.notnull()]
#są inne - TODO

#modyfikacje
reviews.set_index("title") #zmień kolumnę indeksową z poprzedniej na kolumnę "title". teraz:
reviews.loc["Nicosia 2013 Vulkà Bianco (Etna)", :]

reviews['critic'] = 'everyone' #przypisanie stałej
reviews['critic']

reviews['index_backwards'] = range(len(reviews), 0, -1) #przypisanie iterable
reviews['index_backwards']

### 3. Summary Functions and Maps ###
reviews
reviews.head()
reviews.describe()
#różne describe'y dla zmiennych liczbowych i stringów
reviews.points.describe()
reviews.taster_name.describe()

reviews.points.mean()
reviews.taster_name.value_counts()

#map - dla serii
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)

#apply - dla DataFrame'ów
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns') #axis='columns' i axis=1 wywołuje funkcję dla każdego wiersza
                                             #axis='index'   i axis=0                      każdej kolumny
#map i apply zwracają nowy obiekt, nie modyfikują oryginalny

#wbudowane mapy
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean #typowe operacje pythonowe - arytmetyczne, warunkowe
reviews.country + " - " + reviews.region_1 #konkatenacja stringów

### 4. Grouping and Sorting ###
#zwraca obiekt DataFrameGroupBy - każda grupa to jakby DataFrame zawierający elementy oryginalnego pasujące do danej grupy
reviews.groupby('points').points.count()
reviews.groupby('points').price.min()
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
reviews.groupby(['country']).price.agg([len, min, max])

#wielowarstwowe indeksy 
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed
countries_reviewed.reset_index() #
countries_reviewed.sort_values(by='len', ascending=False)
countries_reviewed.sort_values(by=['country', 'len'], ascending=False)
countries_reviewd.sort_index()

### 5. Data Types and Missing Values ###
reviews.price.dtype
reviews.dtypes #zwraca serię której indeksy to nazwy kolumn a wartości to dtype danej kolumny
reviews.points.astype('float64') #konwersja
reviews.index.dtype #indeksy też mają typ

reviews.region_2.fillna('Unknown')

### 6. Renaming and Combining ###
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")

reviews.rename(columns={'points': 'score'}) #zmienia nazwę kolumny 'points' na 'score'
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube]) #dokleja na dole

left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK') #dokleja z boku
