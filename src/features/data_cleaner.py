import pandas as pd
import numpy as np
import os

class DataCleaner:
    def __init__(self, data_path:str): 
        self.data_path = data_path
        self.df = None
        self.subjects = ['Toán', 'Văn', 'Lý', 'Hóa', 'Sinh', 'Sử', 'Địa', 'GD Kinh tế - Pháp luật', 'Tin học', 'Công nghệ', 'Ngoại ngữ']

    def load_data(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"File {self.data_path} not found!")

            self.df = pd.read_csv(self.data_path, dtype={'SBD':str})
            return self.df

        except pd.errors.EmptyDataError:
            print(f"Empty CSV")
            raise

        except Exception as ex:
            print(f"Serious Error During Loading Data:{ex}")
            raise
    
    def score_under_1(self) ->pd.DataFrame:
        try:
            if self.df is None:
                raise ValueError("Empty DataFrame đang trống!")

            missing_cols = [col for col in self.subjects if col not in self.df.columns]
            if missing_cols:
                raise KeyError(f'File is missing the column subject:{missing_cols}')

            under_1_score = (self.df[self.subjects] <= 1.0).any(axis=1)
            self.df['score_under_1'] = under_1_score
            print(f'Eplore {under_1_score.sum()} students under 1 point!')
            return self.df

        except KeyError as ke:
            print(f"Error caused by input data: {ke}")
            raise
        except Exception as e:
            print(f"Errors during point calculation: {e}")
            raise            

    def split_by_block(self) -> dict:
        try: 
            if self.df is None:
                raise ValueError('Empty dataframe!')

            if 'score_under_1' not in self.df.columns:
                print('Not execute score_under_1 function yet, Auto ignore the socre under 1!')
                df_clean = self.df.copy()

            else:
                df_clean = self.df[self.df['score_under_1'] == False].copy()

            # blocks = {
            #     'A00': ['Toán', 'Lý', 'Hóa'],
            #     'A01': ['Toán', 'Lý', 'Ngoại ngữ'],
            #     'B00': ['Toán', 'Hóa', 'Sinh'],
            #     'C00': ['Văn', 'Sử', 'Địa'],
            #     'D01': ['Toán', 'Văn', 'Ngoại ngữ']
            # }

            blocks = {
            # ===== Khối A =====
            "A00": ["Toán", "Lý", "Hóa"],
            "A01": ["Toán", "Lý", "Ngoại ngữ"],
            "A02": ["Toán", "Lý", "Sinh"],
            "A03": ["Toán", "Lý", "Sử"],
            "A04": ["Toán", "Lý", "Địa"],
            "A05": ["Toán", "Hóa", "Sử"],
            "A06": ["Toán", "Hóa", "Địa"],
            "A07": ["Toán", "Sử", "Địa"],
            "A08": ["Toán", "Sử", "GD Kinh tế - Pháp luật"],
            "A09": ["Toán", "Địa", "GD Kinh tế - Pháp luật"],
            "A10": ["Toán", "Lý", "GD Kinh tế - Pháp luật"],
            "A11": ["Toán", "Hóa", "GD Kinh tế - Pháp luật"],

            # ===== Khối B =====
            "B00": ["Toán", "Hóa", "Sinh"],
            "B01": ["Toán", "Sinh", "Lý"],
            "B02": ["Toán", "Sinh", "Địa"],
            "B03": ["Toán", "Sinh", "Văn"],
            "B04": ["Toán", "Sinh", "GD Kinh tế - Pháp luật"],
            "B05": ["Toán", "Sinh", "Ngoại ngữ"],

            # ===== Khối C =====
            "C00": ["Văn", "Sử", "Địa"],
            "C01": ["Văn", "Toán", "Lý"],
            "C02": ["Văn", "Toán", "Hóa"],
            "C03": ["Văn", "Toán", "Sử"],
            "C04": ["Văn", "Toán", "Địa"],
            "C05": ["Văn", "Lý", "Hóa"],
            "C06": ["Văn", "Lý", "Sinh"],
            "C07": ["Văn", "Lý", "Sử"],
            "C08": ["Văn", "Hóa", "Sinh"],
            "C09": ["Văn", "Lý", "GD Kinh tế - Pháp luật"],
            "C10": ["Văn", "Hóa", "GD Kinh tế - Pháp luật"],
            "C11": ["Văn", "Sinh", "GD Kinh tế - Pháp luật"],

            # ===== Khối D =====
            "D01": ["Toán", "Văn", "Ngoại ngữ"],
            "D07": ["Toán", "Hóa", "Ngoại ngữ"],
            "D08": ["Toán", "Sinh", "Ngoại ngữ"],
            "D09": ["Toán", "Sử", "Ngoại ngữ"],
            "D10": ["Toán", "Địa", "Ngoại ngữ"],
            "D84": ["Toán", "GD Kinh tế - Pháp luật", "Ngoại ngữ"],
            "D90": ["Toán", "Khoa học tự nhiên", "Ngoại ngữ"],  # Không dùng với CSV hiện tại
            }

            splitted_data = {}
            for block_name, block_subjects in blocks.items():
                missing_subjects = [subj for subj in block_subjects if subj not in df_clean.columns]
                if missing_subjects:
                    print(f"Ignore {block_name} since not exist in CSV file: {missing_subjects}")
                    continue
                valid_mask = df_clean[block_subjects].notna().all(axis=1)
                df_block = df_clean[valid_mask].copy()            
                df_block[f'Total_{block_name}'] = df_block[block_subjects].sum(axis=1)
                splitted_data[block_name] = df_block
                print(f"Block {block_name}: extracted {len(df_block)} valid students")
            return splitted_data
            # for block_name, block_subjects in blocks.items():
            #     valid_mask = df_clean[block_subjects].notna().all(axis=1)
            #     df_block = df_clean[valid_mask].copy()            

            #     df_block[f'Total_{block_name}'] = df_block[block_subjects].sum(axis=1)
            #     splitted_data[block_name] = df_block

            #     print(f"Block {block_name}: extracts {len(df_block)} valid students")

            # return splitted_data

        except KeyError as ke:
            print(f"Block calculation error: {ke}")
            raise
        except Exception as e:
            print(f"Unknown error during split data: {e}")
            raise