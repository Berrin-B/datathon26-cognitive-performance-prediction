import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        self.medians = {}
        self.modes = {}
        self.stres_map = {}
        self.global_stres_median = None
        self.outlier_limits = {}
        self.country_map = {
            'Spain': 'Ispanya', 'South Korea': 'Guney Kore',
            'Sweden': 'Isvec', 'Mexico': 'Meksika', 'Netherlands': 'Hollanda'
        }

    def basic_clean(self, df):
        """Sütun isimleri ve string formatlarını düzeltir."""
        df = df.copy()
        if 'ulke' in df.columns:
            df['ulke'] = df['ulke'].replace(self.country_map)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df = df.replace(['nan', 'NaN', '?', 'None', '', 'null', 'NULL'], np.nan)
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        return df

    def fit(self, train_df):
        """Eğitim verisinden istatistikleri öğrenir."""
        temp_df = self.basic_clean(train_df)
        numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.medians[col] = temp_df[col].median()
        if 'kronotip' in temp_df.columns:
            self.modes['kronotip'] = temp_df['kronotip'].mode()[0]
        if 'stres_skoru' in temp_df.columns and 'ruh_sagligi_durumu' in temp_df.columns:
            self.stres_map = temp_df.groupby('ruh_sagligi_durumu')['stres_skoru'].median().to_dict()
            self.global_stres_median = temp_df['stres_skoru'].median()
        
        outlier_cols = ['vucut_kitle_indeksi', 'uyku_oncesi_kafein_mg', 'gunluk_adim_sayisi', 
                        'sekerleme_suresi_dk', 'gunluk_calisma_saati', 'uykuya_dalma_suresi_dk']
        for col in outlier_cols:
            if col in temp_df.columns:
                self.outlier_limits[col] = (temp_df[col].quantile(0.01), temp_df[col].quantile(0.99))

    def transform_for_eda(self, df):
        """EDA için veriyi temizler ve doldurur ama encoding yapmaz (Okunabilir bırakır)."""
        df = self.basic_clean(df)
        df['meslek'] = df['meslek'].fillna('Bilinmiyor')
        df['ruh_sagligi_durumu'] = df['ruh_sagligi_durumu'].fillna('Bilinmiyor')
        if 'kronotip' in df.columns:
            df['kronotip'] = df['kronotip'].fillna(self.modes['kronotip'])
        
        # Sayısal dolgular
        for col, val in self.medians.items():
            if col in df.columns:
                fill_val = 0 if col == 'uyku_oncesi_kafein_mg' else val
                df[col] = df[col].fillna(fill_val)
        
        if 'stres_skoru' in df.columns:
            df['stres_skoru'] = df['stres_skoru'].fillna(df['ruh_sagligi_durumu'].map(self.stres_map))
            df['stres_skoru'] = df['stres_skoru'].fillna(self.global_stres_median)
            
        for col, (low, high) in self.outlier_limits.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=low, upper=high)
        return df

    def encode_data(self, df):
        """EDA'sı bitmiş veriyi modele hazır hale getirmek için encode eder."""
        df = df.copy()
        binary_mapping = {
            'cinsiyet': {'Erkek': 0, 'Kadin': 1},
            'mevsim': {'Sonbahar-Kis': 0, 'Ilkbahar-Yaz': 1},
            'gun_tipi': {'Hafta ici': 0, 'Hafta sonu': 1}
        }
        for col, mapping in binary_mapping.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        ohe_cols = ['meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu']
        ohe_cols = [c for c in ohe_cols if c in df.columns]
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
        return df