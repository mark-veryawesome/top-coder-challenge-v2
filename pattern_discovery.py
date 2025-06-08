#!/usr/bin/env python3
"""
Deep Pattern Discovery - Focus on finding the exact business logic
"""

import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def deep_pattern_analysis():
    # Load data
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    rows = []
    for i, case in enumerate(cases):
        input_data = case['input']
        rows.append({
            'case_id': i + 1,
            'trip_duration_days': input_data['trip_duration_days'],
            'miles_traveled': input_data['miles_traveled'], 
            'total_receipts_amount': input_data['total_receipts_amount'],
            'reimbursement': case['expected_output']
        })
    
    df = pd.DataFrame(rows)
    
    # Add all possible derived features
    df['daily_mileage'] = df['miles_traveled'] / df['trip_duration_days']
    df['cost_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['reimbursement_per_day'] = df['reimbursement'] / df['trip_duration_days']
    df['receipt_ratio'] = df['total_receipts_amount'] / df['reimbursement']
    df['mileage_per_dollar'] = df['miles_traveled'] / (df['total_receipts_amount'] + 1)  # +1 to avoid division by zero
    
    print("DEEP PATTERN DISCOVERY")
    print("=" * 60)
    
    # Create a very simple decision tree to understand the logic
    X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].values
    y = df['reimbursement'].values
    
    # Simple tree - max depth 6 to see clear patterns
    tree = DecisionTreeRegressor(max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=42)
    tree.fit(X, y)
    
    print(f"Decision Tree R²: {tree.score(X, y):.3f}")
    print("\nDecision Tree Rules:")
    feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    tree_rules = export_text(tree, feature_names=feature_names, max_depth=6)
    print(tree_rules)
    
    # Look at specific case patterns
    print("\n" + "=" * 60)
    print("SPECIFIC CASE ANALYSIS")
    print("=" * 60)
    
    # Analyze the problem cases that are failing
    problem_cases = [
        (4, 69, 2321.49, 322.00),   # Case 152
        (1, 1082, 1809.49, 446.94), # Case 996  
        (8, 795, 1645.99, 644.69),  # Case 684
        (5, 516, 1878.49, 669.85),  # Case 711
        (8, 482, 1411.49, 631.81)   # Case 548
    ]
    
    print("\nProblem Case Analysis:")
    for days, miles, receipts, expected in problem_cases:
        # Find similar cases in the data
        similar = df[
            (abs(df['trip_duration_days'] - days) <= 1) &
            (abs(df['miles_traveled'] - miles) <= 100) &
            (abs(df['total_receipts_amount'] - receipts) <= 300)
        ]
        
        print(f"\nCase: {days} days, {miles} miles, ${receipts:.2f} → Expected: ${expected:.2f}")
        print(f"  Cost per day: ${receipts/days:.2f}")
        print(f"  Daily mileage: {miles/days:.1f}")
        print(f"  Expected per day: ${expected/days:.2f}")
        
        if len(similar) > 0:
            print(f"  Similar cases found: {len(similar)}")
            for _, sim in similar.head(3).iterrows():
                print(f"    {sim['trip_duration_days']} days, {sim['miles_traveled']} miles, ${sim['total_receipts_amount']:.2f} → ${sim['reimbursement']:.2f}")
        else:
            print("  No similar cases found")
    
    # Look for receipt-based patterns
    print("\n" + "=" * 60)
    print("RECEIPT PATTERN ANALYSIS")
    print("=" * 60)
    
    # Group by spending level and see patterns
    df['spending_category'] = pd.cut(df['cost_per_day'], 
                                   bins=[0, 50, 100, 200, 400, float('inf')],
                                   labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    spending_analysis = df.groupby('spending_category').agg({
        'reimbursement_per_day': ['count', 'mean', 'std'],
        'receipt_ratio': 'mean'
    }).round(2)
    print("Spending Category Analysis:")
    print(spending_analysis)
    
    # Look for mileage patterns  
    print("\n" + "=" * 60)
    print("MILEAGE PATTERN ANALYSIS")
    print("=" * 60)
    
    df['mileage_category'] = pd.cut(df['daily_mileage'],
                                   bins=[0, 50, 100, 200, 400, float('inf')],
                                   labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    mileage_analysis = df.groupby('mileage_category').agg({
        'reimbursement_per_day': ['count', 'mean', 'std']
    }).round(2)
    print("Mileage Category Analysis:")
    print(mileage_analysis)
    
    # Cross-tabulation analysis
    print("\n" + "=" * 60)
    print("CROSS-PATTERN ANALYSIS")
    print("=" * 60)
    
    cross_analysis = pd.crosstab(df['spending_category'], df['mileage_category'], 
                               values=df['reimbursement_per_day'], aggfunc='mean').round(2)
    print("Average Reimbursement per Day by Spending vs Mileage:")
    print(cross_analysis)
    
    return df, tree

def find_exact_formula_patterns(df):
    """Try to reverse engineer the exact formula"""
    print("\n" + "=" * 60)
    print("EXACT FORMULA REVERSE ENGINEERING")
    print("=" * 60)
    
    # Test if there are any perfect mathematical relationships
    print("\n1. Testing Simple Mathematical Relationships:")
    
    # Test various potential formulas
    formulas = [
        ("days * 100 + miles * 0.5 + receipts * 0.3", lambda r: r['trip_duration_days'] * 100 + r['miles_traveled'] * 0.5 + r['total_receipts_amount'] * 0.3),
        ("days * 80 + miles * 0.4", lambda r: r['trip_duration_days'] * 80 + r['miles_traveled'] * 0.4),
        ("days * 120 + sqrt(miles) * 20", lambda r: r['trip_duration_days'] * 120 + (r['miles_traveled'] ** 0.5) * 20),
        ("Complex: base + efficiency + receipts", complex_formula)
    ]
    
    for name, formula_func in formulas:
        try:
            df['predicted'] = df.apply(formula_func, axis=1)
            mae = abs(df['reimbursement'] - df['predicted']).mean()
            max_error = abs(df['reimbursement'] - df['predicted']).max()
            r2 = np.corrcoef(df['reimbursement'], df['predicted'])[0,1] ** 2
            
            print(f"  {name}:")
            print(f"    MAE: ${mae:.2f}")
            print(f"    Max Error: ${max_error:.2f}")
            print(f"    R²: {r2:.3f}")
            
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    # Look for segmented patterns
    print("\n2. Segmented Pattern Analysis:")
    
    # Short trips (1-3 days)
    short_trips = df[df['trip_duration_days'] <= 3]
    if len(short_trips) > 0:
        print(f"\nShort trips (≤3 days): {len(short_trips)} cases")
        analyze_segment(short_trips, "Short")
    
    # Medium trips (4-7 days)  
    medium_trips = df[(df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 7)]
    if len(medium_trips) > 0:
        print(f"\nMedium trips (4-7 days): {len(medium_trips)} cases")
        analyze_segment(medium_trips, "Medium")
    
    # Long trips (8+ days)
    long_trips = df[df['trip_duration_days'] >= 8]
    if len(long_trips) > 0:
        print(f"\nLong trips (≥8 days): {len(long_trips)} cases")
        analyze_segment(long_trips, "Long")

def complex_formula(row):
    """A more complex formula attempt"""
    days = row['trip_duration_days']
    miles = row['miles_traveled']
    receipts = row['total_receipts_amount']
    
    # Base rate varies by trip length
    if days <= 3:
        base_rate = 200
    elif days <= 7:
        base_rate = 150
    else:
        base_rate = 100
    
    base = days * base_rate
    
    # Mileage component
    if miles <= 100:
        mileage_comp = miles * 0.6
    elif miles <= 500:
        mileage_comp = 60 + (miles - 100) * 0.4
    else:
        mileage_comp = 220 + (miles - 500) * 0.2
    
    # Receipt component  
    if receipts <= 500:
        receipt_comp = receipts * 0.3
    elif receipts <= 1500:
        receipt_comp = 150 + (receipts - 500) * 0.2
    else:
        receipt_comp = 350 + (receipts - 1500) * 0.1
    
    return base + mileage_comp + receipt_comp

def analyze_segment(segment_df, segment_name):
    """Analyze a specific segment for patterns"""
    print(f"  {segment_name} segment analysis:")
    print(f"    Average reimbursement per day: ${segment_df['reimbursement_per_day'].mean():.2f}")
    print(f"    Average cost per day: ${segment_df['cost_per_day'].mean():.2f}")
    print(f"    Average daily mileage: {segment_df['daily_mileage'].mean():.1f}")
    print(f"    Average receipt ratio: {segment_df['receipt_ratio'].mean():.2f}")
    
    # Look for correlations within the segment
    corr_with_reimbursement = segment_df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement']].corr()['reimbursement']
    print(f"    Correlations with reimbursement:")
    for var, corr in corr_with_reimbursement.items():
        if var != 'reimbursement':
            print(f"      {var}: {corr:.3f}")

if __name__ == "__main__":
    df, tree = deep_pattern_analysis()
    find_exact_formula_patterns(df)
    
    # Save the tree for implementation
    with open('decision_tree_analysis.txt', 'w') as f:
        feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        tree_rules = export_text(tree, feature_names=feature_names, max_depth=6)
        f.write(tree_rules)