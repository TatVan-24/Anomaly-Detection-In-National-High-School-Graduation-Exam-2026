import pandas as pd
import numpy as np
import warnings

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        try:
            self.df = df.copy()
            
            self.subjects = [
                'Toán', 'Văn', 'Lý', 'Hóa', 'Sinh', 'Sử', 
                'Địa', 'GD Kinh tế - Pháp luật', 'Tin học', 
                'Công nghệ', 'Ngoại ngữ'
            ]
            
            self.available_subjects = [col for col in self.subjects if col in self.df.columns]
        except Exception as e:
            print(f"Error during initializing FeatureEngineer: {e}")
            raise

    def extract_features(self) -> pd.DataFrame:
        try:
            self.df['SBD'] = self.df['SBD'].astype(str).str.zfill(8)
            self.df = self.df.sort_values('SBD').reset_index(drop=True)
            self.df['Province_Code'] = self.df['SBD'].str[:2]
        except Exception as e:
                print(f"Error at SBD preparation process: {e}")
                raise

        try:
            print("[1/4] Calculating internal feature F1 - Personal std...")
            self.df['F1_Personal_Std'] = self.df[self.available_subjects].std(axis=1, skipna=True)
            self.df['F1_Personal_Std'] = self.df['F1_Personal_Std'].fillna(0) 
        except Exception as e:
            print(f"Error during calculating F1: {e}")
            raise        

        try:
            print("[2/4] Calculating internal feature F2: - Z-score range (Max-Min Z-score Range)...")
            z_scores = pd.DataFrame()
            for subj in self.available_subjects:
                mean_val = self.df[subj].mean(skipna=True)
                std_val = self.df[subj].std(skipna=True)
                if std_val > 0:
                    z_scores[f'Z_{subj}'] = (self.df[subj] - mean_val) / std_val
                else:
                    z_scores[f'Z_{subj}'] = np.nan
                        
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                z_max = z_scores.max(axis=1, skipna=True)
                z_min = z_scores.min(axis=1, skipna=True)
                self.df['F2_Z_Range'] = z_max - z_min
                    
            self.df['F2_Z_Range'] = self.df['F2_Z_Range'].fillna(0)
        except Exception as e:
            print(f"Error during calculating F2: {e}")
            raise

        try:
            print("[3/4] Calculation F3: Space fature - Z-score Exam room (Micro-Anomaly)...")
            self.df['Avg_Score'] = self.df[self.available_subjects].mean(axis=1, skipna=True)
                
            window_size = 24
                
            self.df['Window_Mean'] = self.df.groupby('Province_Code')['Avg_Score'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean()
                )
            self.df['Window_Std'] = self.df.groupby('Province_Code')['Avg_Score'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).std()
                )
                
            self.df['Window_Std'] = self.df['Window_Std'].replace(0, 1e-9).fillna(1)
            self.df['F3_Micro_Z'] = (self.df['Avg_Score'] - self.df['Window_Mean']) / self.df['Window_Std']
        except Exception as e:
            print(f"Error during calculating F3: {e}")
            raise

        try:
            print("[4/4] Calculation F4: Space feature - Z-score Exam cluster/Provice (Macro-Anomaly)...")
            province_stats = self.df.groupby('Province_Code')['Window_Mean'].agg(['mean', 'std']).reset_index()
            province_stats.rename(columns={'mean': 'Province_Mean', 'std': 'Province_Std'}, inplace=True)
                
            self.df = self.df.merge(province_stats, on='Province_Code', how='left')
            self.df['Province_Std'] = self.df['Province_Std'].replace(0, 1e-9).fillna(1)
                
            self.df['F4_Macro_Z'] = (self.df['Window_Mean'] - self.df['Province_Mean']) / self.df['Province_Std']
        except Exception as e:
            print(f"Error during calculating F4: {e}")
            raise

        try:
            print("Finish!")
            feature_cols = ['F1_Personal_Std', 'F2_Z_Range', 'F3_Micro_Z', 'F4_Macro_Z']
                
            self.df_features = self.df[['SBD'] + feature_cols].copy()
            self.df_features[feature_cols] = self.df_features[feature_cols].fillna(0)
                
            print(f"Matrix X is ready: {self.df_features.shape[0]:,} student, {self.df_features.shape[1]-1} feature (0 NaN).")
            return self.df_features
        except Exception as e:
            print(f"Error during exporting Output X: {e}")
            raise
                
        except Exception as ex:
            print(f"Extreeme error in extract_features function: {ex}")
            raise