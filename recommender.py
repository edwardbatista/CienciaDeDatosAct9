
import pandas as pd
import ast
from itertools import chain
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors

class KNNCollabRecommender:
    def __init__(self, metric: str = "cosine", algorithm: str = "brute"):
        self.metric = metric
        self.algorithm = algorithm
        self.nn = None
        self.pivot = None
        self.cust_prods = None
        self.mlb = None

    def fit_from_transactions(self, df: pd.DataFrame) -> None:
        if "Customer_Name" not in df.columns or "Product" not in df.columns:
            raise ValueError("El DataFrame debe contener las columnas 'Customer_Name' y 'Product'.")
        if isinstance(df["Product"].iloc[0], str):
            df = df.copy()
            df["Product"] = df["Product"].apply(ast.literal_eval)
        self.cust_prods = (
            df.groupby("Customer_Name")["Product"]
              .apply(lambda lsts: set(chain.from_iterable(lsts)))
        )
        self.mlb = MultiLabelBinarizer(sparse_output=False)
        X = self.mlb.fit_transform(self.cust_prods.values)
        self.pivot = pd.DataFrame(X, index=self.cust_prods.index, columns=self.mlb.classes_)
        self.nn = NearestNeighbors(metric=self.metric, algorithm=self.algorithm)
        self.nn.fit(self.pivot)

    def recommend(self, customer: str, top_k: int = 5, n_neighbors: int = 6) -> pd.Series:
        if self.nn is None or self.pivot is None:
            raise RuntimeError("Debes llamar fit_from_transactions(df) antes de recomendar.")
        if customer not in self.pivot.index:
            raise ValueError(f"Cliente '{customer}' no existe en la matriz.")
        query_df = self.pivot.loc[[customer]]  # preservar nombres de columnas
        k = min(len(self.pivot), max(2, n_neighbors))
        distances, indices = self.nn.kneighbors(query_df, n_neighbors=k)
        neighbors = self.pivot.iloc[indices[0]].index.tolist()
        neighbors = [n for n in neighbors if n != customer]
        user_items = self.cust_prods.loc[customer]
        counts = Counter(chain.from_iterable(self.cust_prods.loc[n] for n in neighbors))
        ranked = [(item, freq) for item, freq in counts.items() if item not in user_items]
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]
        return pd.Series({item: int(freq) for item, freq in ranked})
