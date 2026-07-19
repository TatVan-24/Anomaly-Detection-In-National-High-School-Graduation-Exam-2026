import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import warnings

class ClusterInvestigator:
    def __init__(self, pattern3_df: pd.DataFrame):
        self.df = pattern3_df.copy()

    def investigate_province(self, province_code: str, n_clusters: int = 2) -> pd.DataFrame:
        province_outliers = self.df[self.df['Province_Code'] == province_code].copy()
        
        if len(province_outliers) == 0:
            print(f"Not found anomaly in Province {province_code}")
            return None
        province_outliers['SBD_seq'] = province_outliers['SBD'].astype(str).str.zfill(8).str[2:].astype(int)
        X_1d = province_outliers[['SBD_seq']].values
        
        actual_clusters = min(n_clusters, len(X_1d))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
            province_outliers['Cluster'] = kmeans.fit_predict(X_1d)
        
        return province_outliers