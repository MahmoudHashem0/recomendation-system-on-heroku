# import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class content_RcmSys():
    def __init__(self, popularity=True, genre= True, year= True):
        self.popularity= popularity
        self.genre= genre
        self.year= year
    
    def fit(self, df, item, user):
        self.df = df
        self.item= item
        self.user = user
        
        if self.year: self.__year_score(istrain=True)
        if self.popularity: self.__popularity_score(istrain=True)
        if self.genre: self.__genre_score(istrain=True)



    def __popularity_score(self, istrain, candidate_similar_items=None): #  item_score=None, tracks_similar_not_target=None,
        if istrain:
            self.track_popularity = self.df[~ self.df.track_id.duplicated()][[self.item, 'track_popularity']].set_index(self.item)
        else:
            popularity_score = self.track_popularity.loc[candidate_similar_items]['track_popularity']
            popularity_score.name = 'popularity_score'
            return popularity_score
#             item_score = (item_score * (popularity_score.to_numpy() + 1))
#             return item_score

        
    def __genre_score(self, istrain, candidate_similar_items=None , target_user=None):
        if istrain:
            self.track_genre = self.df[~ self.df.track_id.duplicated()][[self.item, 'track_genre']].set_index(self.item)
            self.track_genres_ratios_users = self.df.groupby(self.user).apply(lambda X:X['track_genre'].value_counts() / X[self.user].count())
        else:
            # candidate_items_genre
            candidate_items_genre = self.track_genre.loc[candidate_similar_items]
            track_genres_ratios_target = self.track_genres_ratios_users[target_user]
            genre_score = candidate_items_genre['track_genre'].apply(lambda row:track_genres_ratios_target.get(row, 0))
            genre_score.name = 'genre_score'
            return genre_score
#             item_score['track_rank'] = item_score['track_rank'] * (item_score['genre_score']+1)
#             return item_score
    def __year_score(self, istrain, candidate_similar_items=None , target_user=None):
        if istrain:
            self.year_scaler = MinMaxScaler()
            self.year_scaler.fit(self.df[['year']])
            self.track_year = self.df[~ self.df.track_id.duplicated()][[self.item, 'year']].set_index(self.item)
        else:
            # candidate_items_genre
            candidate_items_year = self.track_year.loc[candidate_similar_items]
            year_score = self.year_scaler.transform(candidate_items_year)
            year_score = pd.Series(year_score.squeeze(), name = "modernity_score", index=candidate_similar_items)
            return year_score
    

    
    def predict(self, target_user):       
        candidate_similar_items = self.track_genre.index
        
        popularity_score = self.__popularity_score(False, candidate_similar_items) if self.popularity else 0
        genre_score = self.__genre_score(False, candidate_similar_items, target_user) if self.genre else 0
        year_score = self.__year_score(False, candidate_similar_items, target_user) if self.year else 0

        
        total_score = (genre_score+1) * (popularity_score+1) * (year_score+1)
        

        total_score.name = 'total_score'
        item_score = pd.DataFrame(total_score)

        try: item_score = pd.concat([popularity_score, item_score], axis=1)
        except: pass
        
        try: item_score = pd.concat([genre_score, item_score], axis=1)
        except: pass

        try: item_score = pd.concat([year_score, item_score], axis=1)
        except: pass

        # item_score = pd.concat([interactions_score, item_score], axis=1)
        
        return item_score.sort_values(by='total_score', ascending=False)


# df = pd.read_csv("spotify_full_dataset_cleaned_feature_engineering.csv")

# ob = content_RcmSys()
# ob.fit(df, 'track_id', 'playlist_pid')
# pickle.dump(ob, open('content_model.pkl','wb'))
# model = pickle.load(open('content_model.pkl','rb'))
# target_user = 10
# item_rank = model.predict(target_user)

# print(item_rank)
