"""
Main OFI Processing Pipeline
1. Loads and preprocesses order book data
2. Computes all OFI features
3. Exports results to CSV
"""

import pandas as pd
from ofi_compute import (
    compute_best_level_ofi,
    compute_multilevel_ofi,
    compute_scaled_multilevel_ofi,
    compute_integrated_ofi,
    compute_cross_asset_ofi
)

def main(input_csv: str = 'first_25000_rows.csv', interval: str = '1min') -> None:
    # Load and preprocess raw order book data
    df = pd.read_csv(input_csv, parse_dates=['ts_recv']).rename(columns={'ts_recv': 'timestamp'})
    
    # Map raw columns to standardized Pn_b/a (price) and Qn_b/a (quantity) format
    rename_map = {}
    for col in df.columns:
        if 'bid_px_' in col:
            level = int(col.split('_')[-1]) + 1
            rename_map[col] = f'P{level}_b'
        elif 'bid_sz_' in col:
            level = int(col.split('_')[-1]) + 1
            rename_map[col] = f'Q{level}_b'
        elif 'ask_px_' in col:
            level = int(col.split('_')[-1]) + 1
            rename_map[col] = f'P{level}_a'
        elif 'ask_sz_' in col:
            level = int(col.split('_')[-1]) + 1
            rename_map[col] = f'Q{level}_a'
    
    processed_df = df.rename(columns=rename_map)
    results = {}

    # Process each symbol independently
    for symbol, group in processed_df.groupby('symbol'):
        # Compute OFI features
        best_ofi = compute_best_level_ofi(group, interval=interval)
        multi_ofi = compute_multilevel_ofi(group, interval=interval)
        scaled_ofi = compute_scaled_multilevel_ofi(group, interval=interval)
        integrated_ofi = compute_integrated_ofi(scaled_ofi)
        
        # Combine and export results
        output = pd.concat([best_ofi, multi_ofi, integrated_ofi], axis=1)
        output.columns = ['best (ofi1)'] + [f'ofi{m}' for m in range(1, multi_ofi.shape[1]+1)] + ['ofiI']
        output.to_csv(f'data/ofi_{symbol}.csv')
        results[symbol] = integrated_ofi

    # Generate cross-asset matrix (placeholder implementation)
    cross_asset_df = compute_cross_asset_ofi(results)
    cross_asset_df.to_csv('data/cross_asset_ofi.csv')

if __name__ == '__main__':
    main()