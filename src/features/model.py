import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination=0.001, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=300, 
            contamination=contamination, 
            random_state=random_state, 
            n_jobs=-1 
        )
        self.feature_cols = ['F1_Personal_Std', 'F2_Z_Range', 'F3_Micro_Z', 'F4_Macro_Z']

    def fit_predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        self.df = df_features.copy()

        X_scaled = self.scaler.fit_transform(self.df[self.feature_cols])
        df_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)
        self.df['anomaly_label'] = self.model.fit_predict(df_scaled)
        self.df['anomaly_score'] = self.model.decision_function(df_scaled)
        for col in self.feature_cols:
            self.df[f'{col}_scaled'] = df_scaled[col]
        return self.df

    def extract_blacklist(self) -> pd.DataFrame:
        blacklist = self.df[self.df['anomaly_label'] == -1].sort_values(by='anomaly_score').copy()
        threshold = 2.5 

        def classify_pattern(row):
            if row['F4_Macro_Z_scaled'] > 3.0: 
                return "Pattern 3: Organized fraud (Exam cluster/Provice)"

            elif row['F3_Micro_Z_scaled'] > threshold:
                    return "Pattern 2: Personal fraud"

            elif row['F1_Personal_Std_scaled'] > threshold and row['F2_Z_Range_scaled'] > threshold:
                    return "Pattern 1: ...."
                
            else:
                return "Pattern 0: Mixed anomaly"
            
        blacklist['Insight_Pattern'] = blacklist.apply(classify_pattern, axis=1)
        return blacklist