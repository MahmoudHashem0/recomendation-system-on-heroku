import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

class RcmSys():
    def __init__(self, popularity=True, genre= True):
        self.popularity= popularity
        self.genre= genre
    
    def fit(self, file_path, item, user):
        self.df = pd.read_csv(file_path)
        self.item= item
        self.user = user
        
        item_ = self.item + "_"
        self.df[item_] = self.df[self.item]
        self.user_item_matrix = pd.pivot_table(data= self.df, index=self.user, columns=self.item, values = item_, aggfunc=pd.Series.count, fill_value=0 )

        user_cosine_similarity = cosine_similarity(self.user_item_matrix) # Calculate similarity between users
        np.fill_diagonal(user_cosine_similarity, 0) # Set diagonal to zero not to affect our calculations.
        self.user_cosine_similarity = pd.DataFrame(user_cosine_similarity)
        
        if self.popularity: self.__popularity_score(istrain=True)
        if self.genre: self.__genre_score(istrain=True)


    def __interactions_score(self, target_user, user_similarity_threshold, n):
        users_similarlity_scores = self.user_cosine_similarity.iloc[target_user] # get the similarity score between target user and others
        similar_users_scores = users_similarlity_scores[users_similarlity_scores > user_similarity_threshold].sort_values(ascending=False)[:n]
        target_user_tracks = self.user_item_matrix.iloc[target_user][self.user_item_matrix.iloc[target_user] > 0].index
        
        similar_users_track_matrix = self.user_item_matrix.iloc[similar_users_scores.index,:]
        similar_users_track_matrix = similar_users_track_matrix[similar_users_track_matrix > 0].dropna(axis=1, how='all')
        
        tracks_similar_not_target = similar_users_track_matrix.drop(columns=target_user_tracks, errors='ignore')
        
        interactions_score = tracks_similar_not_target.apply(lambda col:self.__weighted_avg(col, similar_users_scores))
        interactions_score.name = "interactions_score"
        
        return interactions_score


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
    

    
    def predict(self, target_user, user_similarity_threshold, n):
        interactions_score = self.__interactions_score(target_user, user_similarity_threshold, n)
        candidate_similar_items = interactions_score.index
        
        popularity_score = self.__popularity_score(False, candidate_similar_items) if self.popularity else 0
        genre_score = self.__genre_score(False, candidate_similar_items, target_user) if self.genre else 0
        

        total_score = interactions_score * (genre_score+1) * (popularity_score+1)
        total_score.name = "total_score"
        item_score = pd.DataFrame(total_score)
        
        try: item_score = pd.concat([popularity_score, item_score], axis=1)
        except: pass
        
        try: item_score = pd.concat([genre_score, item_score], axis=1)
        except: pass

        item_score = pd.concat([interactions_score, item_score], axis=1)
        return item_score.sort_values(by='total_score', ascending=False)

        
    def __weighted_avg(self, col, weights):
        numerator = (col * weights).sum()
        denominator = weights[~ col.isnull()].sum()
        w_avg = numerator / denominator
        return w_avg


