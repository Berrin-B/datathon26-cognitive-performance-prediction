import pandas as pd
import numpy as np

def clean_and_format(df):
    """Sütun isimlendirmeleri, tip dönüşümleri ve string temizliği yapar."""
    df = df.copy()
    
    # 1. ID sütununu düşür (Eğer varsa)
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)

    # 2. Ülke isimlerindeki tutarsızlıkları düzelt (Map stratejin)
    country_map = {
        'Spain': 'Ispanya',
        'South Korea': 'Guney Kore',
        'Sweden': 'Isvec',
        'Mexico': 'Meksika', 
        'Netherlands': 'Hollanda'
    }
    if 'ulke' in df.columns:
        df['ulke'] = df['ulke'].replace(country_map)

    # 3. String temizliği (leading/trailing whitespace)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # 4. Gizli NaN değerlerini gerçek NaN ile değiştir
    df = df.replace(['nan', 'NaN', '?', 'None', '', 'null', 'NULL'], np.nan)
    
    return df

def handle_missing_values(df):
    """Sütun bazlı özel eksik veri doldurma stratejilerini uygular."""
    df = df.copy()
    
    # Kategorik dolgular
    if 'meslek' in df.columns:
        df['meslek'] = df['meslek'].fillna('Bilinmiyor')
    
    if 'kronotip' in df.columns:
        # Mode her zaman bir Seri döner, o yüzden [0] alıyoruz
        df['kronotip'] = df['kronotip'].fillna(df['kronotip'].mode()[0])
        
    if 'ruh_sagligi_durumu' in df.columns:
        df['ruh_sagligi_durumu'] = df['ruh_sagligi_durumu'].fillna('Bilinmiyor')

    # Sayısal dolgular
    if 'vucut_kitle_indeksi' in df.columns:
        df['vucut_kitle_indeksi'] = df['vucut_kitle_indeksi'].fillna(df['vucut_kitle_indeksi'].median())
        
    if 'uyku_oncesi_kafein_mg' in df.columns:
        df['uyku_oncesi_kafein_mg'] = df['uyku_oncesi_kafein_mg'].fillna(df['uyku_oncesi_kafein_mg'].median())

    # Stres skoru için gruplandırılmış median dolgusu
    if 'stres_skoru' in df.columns and 'ruh_sagligi_durumu' in df.columns:
        df['stres_skoru'] = df.groupby('ruh_sagligi_durumu')['stres_skoru'].transform(lambda x: x.fillna(x.median()))
        # Eğer hala dolmayan varsa (tüm grubun nan olması durumu) genel median ile doldur
        df['stres_skoru'] = df['stres_skoru'].fillna(df['stres_skoru'].median())

    return df

def correct_outliers(df):
    """Belirlenen kritik sütunlarda %1-%99 Capping uygular."""
    df = df.copy()
    outlier_cols = ['vucut_kitle_indeksi', 'uyku_oncesi_kafein_mg', 'gunluk_adim_sayisi', 'sekerleme_suresi_dk']
    
    for col in outlier_cols:
        if col in df.columns:
            lower_limit = df[col].quantile(0.01)
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
    return df

def encode_features(df):
    """Binary ve One-Hot Encoding işlemlerini uygular."""
    df = df.copy()
    
    # 1. Binary Mapping
    binary_cols = {
        'cinsiyet': {'Erkek': 0, 'Kadin': 1},
        'mevsim': {'Sonbahar-Kis': 0, 'Ilkbahar-Yaz': 1},
        'gun_tipi': {'Hafta ici': 0, 'Hafta sonu': 1}
    }
    
    for col, mapping in binary_cols.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # 2. One-Hot Encoding
    ohe_cols = ['meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu']
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    
    return df

def pipeline(df):
    """Tüm preprocessing adımlarını sırayla çalıştırır."""
    df = clean_and_format(df)
    df = handle_missing_values(df)
    df = correct_outliers(df)
    df = encode_features(df)
    
    # Son bir kontrol: Kalan sayısal eksikleri median ile kapat (Güvenlik önlemi)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df