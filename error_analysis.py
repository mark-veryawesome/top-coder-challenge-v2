#!/usr/bin/env python3
"""
Error Analysis - Understanding why the linear model fails
Focus on the high-error cases to discover the real business logic
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def analyze_high_error_cases():
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
            'actual_reimbursement': case['expected_output']
        })
    
    df = pd.DataFrame(rows)
    
    # Calculate linear model predictions
    X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].values
    y = df['actual_reimbursement'].values
    
    model = LinearRegression().fit(X, y)
    df['linear_prediction'] = model.predict(X)
    df['error'] = np.abs(df['actual_reimbursement'] - df['linear_prediction'])
    
    # Add derived features
    df['cost_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['daily_mileage'] = df['miles_traveled'] / df['trip_duration_days']
    df['actual_per_day'] = df['actual_reimbursement'] / df['trip_duration_days']
    
    print("HIGH ERROR CASE ANALYSIS")
    print("=" * 50)
    
    # Look at the top error cases mentioned in eval.sh output
    high_error_cases = [152, 996, 684, 711, 548]
    
    print("\nSpecific High-Error Cases:")
    for case_id in high_error_cases:
        case = df[df['case_id'] == case_id].iloc[0]
        print(f"\nCase {case_id}:")
        print(f"  Input: {case['trip_duration_days']} days, {case['miles_traveled']} miles, ${case['total_receipts_amount']:.2f}")
        print(f"  Cost per day: ${case['cost_per_day']:.2f}")
        print(f"  Daily mileage: {case['daily_mileage']:.1f}")
        print(f"  Expected: ${case['actual_reimbursement']:.2f} (${case['actual_per_day']:.2f}/day)")
        print(f"  Linear model: ${case['linear_prediction']:.2f}")
        print(f"  Error: ${case['error']:.2f}")
    
    # Pattern analysis of high-error cases
    print("\n" + "=" * 50)
    print("PATTERN ANALYSIS OF HIGH-ERROR CASES")
    print("=" * 50)
    
    # High receipt cases
    high_receipts = df[df['total_receipts_amount'] > 1500]
    print(f"\nHigh Receipt Cases (>$1500): {len(high_receipts)}")
    print(f"  Average actual per day: ${high_receipts['actual_per_day'].mean():.2f}")
    print(f"  Average cost per day: ${high_receipts['cost_per_day'].mean():.2f}")
    print(f"  Linear model avg error: ${high_receipts['error'].mean():.2f}")
    
    # Low cost per day but high receipts (efficiency bonus cases)
    weird_cases = df[(df['cost_per_day'] > 200) & (df['actual_per_day'] < 100)]
    print(f"\nHigh spending but low reimbursement/day: {len(weird_cases)}")
    if len(weird_cases) > 0:
        print(f"  Sample cases:")
        for _, case in weird_cases.head(3).iterrows():
            print(f"    {case['trip_duration_days']} days, ${case['cost_per_day']:.0f}/day spending → ${case['actual_per_day']:.0f}/day reimbursement")
    
    # Very short high-mileage trips
    short_high_mile = df[(df['trip_duration_days'] <= 2) & (df['miles_traveled'] > 500)]
    print(f"\nShort high-mileage trips (≤2 days, >500 miles): {len(short_high_mile)}")
    if len(short_high_mile) > 0:
        print(f"  Average reimbursement per day: ${short_high_mile['actual_per_day'].mean():.2f}")
        print(f"  Average daily mileage: {short_high_mile['daily_mileage'].mean():.1f}")
    
    # Discover potential caps or thresholds
    print(f"\n" + "=" * 50)
    print("THRESHOLD DISCOVERY")
    print("=" * 50)
    
    # Look for receipt caps
    print("\nReceipt Coverage Analysis:")
    df['receipt_coverage'] = df['total_receipts_amount'] / df['actual_reimbursement']
    
    receipt_bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]
    for low, high in receipt_bins:
        subset = df[(df['total_receipts_amount'] >= low) & (df['total_receipts_amount'] < high)]
        if len(subset) > 0:
            avg_coverage = subset['receipt_coverage'].mean()
            print(f"  ${low}-{high}: {avg_coverage:.2f} coverage ratio (n={len(subset)})")
    
    # Look for mileage thresholds
    print("\nMileage Rate Analysis:")
    df['mileage_rate'] = df['actual_reimbursement'] / df['miles_traveled']
    
    mile_bins = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 1500)]
    for low, high in mile_bins:
        subset = df[(df['miles_traveled'] >= low) & (df['miles_traveled'] < high)]
        if len(subset) > 0:
            avg_rate = subset['mileage_rate'].mean()
            print(f"  {low}-{high} miles: ${avg_rate:.3f}/mile (n={len(subset)})")
    
    # Look for trip length effects
    print("\nTrip Length Per-Day Rates:")
    for days in sorted(df['trip_duration_days'].unique()):
        subset = df[df['trip_duration_days'] == days]
        avg_per_day = subset['actual_per_day'].mean()
        print(f"  {days} days: ${avg_per_day:.2f}/day (n={len(subset)})")
    
    return df

def discover_nonlinear_patterns(df):
    """Look for complex business rules that explain the high errors"""
    print(f"\n" + "=" * 50)
    print("NONLINEAR PATTERN DISCOVERY")
    print("=" * 50)
    
    # Pattern 1: Receipt penalty for high spending
    print("\n1. Receipt Penalty Analysis:")
    high_spend_cases = df[df['cost_per_day'] > 300]
    if len(high_spend_cases) > 0:
        print(f"   High spending cases (>$300/day): {len(high_spend_cases)}")
        print(f"   Average reimbursement per day: ${high_spend_cases['actual_per_day'].mean():.2f}")
        print(f"   Average receipt coverage: {high_spend_cases['receipt_coverage'].mean():.2f}")
    
    # Pattern 2: Efficiency bonus discovery
    print("\n2. Efficiency Bonus Analysis:")
    df['efficiency'] = df['miles_traveled'] / df['trip_duration_days']
    
    # Very high efficiency cases
    ultra_efficient = df[df['efficiency'] > 400]
    if len(ultra_efficient) > 0:
        print(f"   Ultra-efficient cases (>400 mi/day): {len(ultra_efficient)}")
        print(f"   Average reimbursement per day: ${ultra_efficient['actual_per_day'].mean():.2f}")
        print(f"   Average spending per day: ${ultra_efficient['cost_per_day'].mean():.2f}")
    
    # Pattern 3: Base rate discovery
    print("\n3. Base Rate Discovery:")
    minimal_cases = df[(df['total_receipts_amount'] < 100) & (df['miles_traveled'] < 100)]
    if len(minimal_cases) > 0:
        print(f"   Minimal expense cases (<$100 receipts, <100 miles): {len(minimal_cases)}")
        print(f"   Average reimbursement per day: ${minimal_cases['actual_per_day'].mean():.2f}")
        
    # Pattern 4: Look for multi-tier system
    print("\n4. Multi-Tier System Analysis:")
    
    # Categorize trips
    df['trip_category'] = 'unknown'
    df.loc[(df['trip_duration_days'] <= 3), 'trip_category'] = 'short'
    df.loc[(df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 7), 'trip_category'] = 'medium'
    df.loc[(df['trip_duration_days'] >= 8), 'trip_category'] = 'long'
    
    for category in ['short', 'medium', 'long']:
        subset = df[df['trip_category'] == category]
        if len(subset) > 0:
            print(f"   {category.capitalize()} trips:")
            print(f"     Average per day: ${subset['actual_per_day'].mean():.2f}")
            print(f"     Receipt coverage: {subset['receipt_coverage'].mean():.2f}")
            print(f"     Count: {len(subset)}")

if __name__ == "__main__":
    df = analyze_high_error_cases()
    discover_nonlinear_patterns(df)
    
    print(f"\n" + "=" * 50)
    print("KEY INSIGHTS FOR BETTER MODEL")
    print("=" * 50)
    print("1. Linear model fails because the system has complex non-linear rules")
    print("2. High spending cases get penalized (not linear coverage)")
    print("3. Very efficient trips get major bonuses")
    print("4. Different rules for short vs medium vs long trips")
    print("5. Need to implement rule-based logic, not just linear regression")