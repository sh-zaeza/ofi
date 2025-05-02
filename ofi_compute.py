"""
Order Flow Imbalance (OFI) Computation Module
Implements features from 'Cross-Impact of Order Flow Imbalance in Equity Markets' (Boudoukh et al.)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def compute_level_ofi(df: pd.DataFrame, level: int, interval: str = '1min') -> pd.Series:
    """
    Computes OFI for a specific order book level using price/quantity dynamics
    
    Args:
        df: DataFrame with price/quantity columns (P1_b, Q1_b, etc.)
        level: Order book level to compute (1=best, 2=next, etc.)
        interval: Resampling interval for aggregation
        
    Returns:
        pd.Series: OFI time series for specified level
    """
    df = df.set_index('timestamp').sort_index()
    
    # Bid-side OFI calculation
    # Price up: New quote (current quantity)
    # Price down: Cancelation (-previous quantity)
    # Same price: Quantity change
    df['of_b'] = np.select(
        [
            df[f'P{level}_b'] > df[f'P{level}_b'].shift(),
            df[f'P{level}_b'] < df[f'P{level}_b'].shift(),
        ],
        [
            df[f'Q{level}_b'],
            -df[f'Q{level}_b'].shift(),
        ],
        default=df[f'Q{level}_b'] - df[f'Q{level}_b'].shift()
    )
    
    # Ask-side OFI calculation (symmetric logic)
    # Price down: New quote (current quantity)
    # Price up: Cancelation (-previous quantity)
    # Same price: Quantity change
    df['of_a'] = np.select(
        [
            df[f'P{level}_a'] < df[f'P{level}_a'].shift(),
            df[f'P{level}_a'] > df[f'P{level}_a'].shift(),
        ],
        [
            df[f'Q{level}_a'],
            -df[f'Q{level}_a'].shift(),
        ],
        default=df[f'Q{level}_a'] - df[f'Q{level}_a'].shift()
    )
    
    # OFI = Bid flow - Ask flow
    df['delta_ofi'] = df['of_b'] - df['of_a']
    return df['delta_ofi'].resample(interval).sum().rename(f'ofi{level}')

def compute_best_level_ofi(df: pd.DataFrame, **kw) -> pd.Series:
    """Shorthand for level=1 (best bid/ask) OFI"""
    return compute_level_ofi(df, level=1, **kw)

def compute_multilevel_ofi(df: pd.DataFrame, M: int = 10, **kw) -> pd.DataFrame:
    """Aggregates OFI across multiple order book levels (1-M)"""
    return pd.concat(
        [compute_level_ofi(df, level=m, **kw) for m in range(1, M+1)],
        axis=1
    )

def compute_scaled_multilevel_ofi(df: pd.DataFrame, M: int = 10, **kw) -> pd.DataFrame:
    """Z-score normalized OFI values for cross-level comparison"""
    ofis = compute_multilevel_ofi(df, M, **kw)
    return (ofis - ofis.mean()) / (ofis.std() + 1e-9)  # Prevent division by zero

def compute_integrated_ofi(ofis: pd.DataFrame) -> pd.Series:
    """
    PCA-weighted combination of multi-level OFI (Section 3.2 of paper)
    Returns: Single time series capturing dominant OFI pattern
    """
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(ofis.fillna(0))
    weights = pca.components_[0]
    weights /= np.sum(np.abs(weights))  # Signed weight normalization
    return pd.Series(ofis.values.dot(weights), index=ofis.index, name='ofiI')

def compute_cross_asset_ofi(ofi_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Placeholder for cross-asset OFI analysis 
    (Implementation requires Lasso regression as in Section 4.3 of paper)
    """
    return pd.DataFrame(ofi_dict).sort_index()