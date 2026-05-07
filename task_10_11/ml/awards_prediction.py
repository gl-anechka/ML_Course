import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from typing import Dict, List, Set


class OptimizedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keyword_frequency_threshold: int = 58):
        self.keyword_frequency_threshold = keyword_frequency_threshold
        self.keywords_dict: Dict[str, int] = {}
        self.unique_genres: Set[str] = set()
        self.unique_directors: Set[str] = set()
        self.unique_locations: Set[str] = set()
        self.frequent_keywords: Set[str] = set()
        self.categorical_features: List[str] = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']

    def fit(self, X: pd.DataFrame, y=None) -> 'OptimizedTransformer':
        self._collect_unique_values(X)
        self._count_keyword_frequencies(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()

        # gender -> числовой код
        for cat_feature in self.categorical_features:
            if cat_feature in X_transformed.columns:
                X_transformed[cat_feature] = (
                    X_transformed[cat_feature]
                    .fillna('unknown')
                    .astype('category')
                    .cat.codes
                    .astype(np.int16)
                )

        all_features = {}

        self._transform_list_features(X_transformed, all_features, 'genres', self.unique_genres, prefix='genre')
        self._transform_list_features(X_transformed, all_features, 'directors', self.unique_directors, prefix='director')
        self._transform_list_features(X_transformed, all_features, 'filming_locations', self.unique_locations, prefix='location')
        self._transform_list_features(X_transformed, all_features, 'keywords', self.frequent_keywords, prefix='keyword')

        one_hot_df = pd.DataFrame(all_features, index=X_transformed.index, dtype=np.uint8)

        X_transformed = X_transformed.drop(
            columns=['genres', 'directors', 'filming_locations', 'keywords'],
            errors='ignore'
        )

        result = pd.concat([X_transformed, one_hot_df], axis=1)

        # на всякий случай добиваем пропуски в числах
        for col in result.columns:
            if result[col].dtype.kind in 'biufc':
                result[col] = result[col].fillna(0)

        return result

    def _collect_unique_values(self, X: pd.DataFrame) -> None:
        if 'genres' in X.columns:
            self.unique_genres = set(
                genre for genres in X['genres']
                if isinstance(genres, list)
                for genre in genres
            )

        if 'directors' in X.columns:
            self.unique_directors = set(
                director for directors in X['directors']
                if isinstance(directors, list)
                for director in directors
            )

        if 'filming_locations' in X.columns:
            self.unique_locations = set(
                location for locations in X['filming_locations']
                if isinstance(locations, list)
                for location in locations
            )

    def _count_keyword_frequencies(self, X: pd.DataFrame) -> None:
        if 'keywords' in X.columns:
            all_keywords = [
                word for words in X['keywords']
                if isinstance(words, list)
                for word in words
            ]

            from collections import Counter
            keyword_counter = Counter(all_keywords)

            self.keywords_dict = {
                k: v for k, v in sorted(
                    keyword_counter.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
            }

            self.frequent_keywords = {
                k for k, v in self.keywords_dict.items()
                if v >= self.keyword_frequency_threshold
            }

    def _transform_list_features(
        self,
        X: pd.DataFrame,
        features_dict: Dict[str, List[int]],
        feature_name: str,
        unique_values: Set[str],
        prefix: str,
    ) -> None:
        if feature_name not in X.columns:
            return

        safe_unique_values = sorted(unique_values)

        for value in safe_unique_values:
            col_name = f'{prefix}__{value}'
            features_dict[col_name] = [
                1 if isinstance(row[feature_name], list) and value in row[feature_name] else 0
                for _, row in X.iterrows()
            ]


def train_model_and_predict(train_file: str, test_file: str) -> np.ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    y_train = df_train["awards"].to_numpy(dtype=np.float32)
    X_train = df_train.drop(columns=["awards"])

    pipeline = Pipeline([
        ('transformer', OptimizedTransformer(keyword_frequency_threshold=58)),
        ('model', HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(df_test)

    return np.asarray(predictions, dtype=np.float64)
