#!/usr/bin/env python3
"""
Analyze the specific failure cases to improve the model
"""

import json
import numpy as np
import pandas as pd

def analyze_failure_cases():
    # Load data
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Convert to DataFrame
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
    
    # Add derived features
    df['cost_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['daily_mileage'] = df['miles_traveled'] / df['trip_duration_days']
    df['actual_per_day'] = df['actual_reimbursement'] / df['trip_duration_days']
    
    print("FAILURE CASE ANALYSIS")
    print("=" * 60)
    
    # Analyze the specific failure cases
    failure_cases = [684, 152, 996, 548, 367]
    
    print("\nSpecific Failure Cases:")
    for case_id in failure_cases:
        case = df[df['case_id'] == case_id].iloc[0]
        print(f"\nCase {case_id}:")
        print(f"  Input: {case['trip_duration_days']} days, {case['miles_traveled']} miles, ${case['total_receipts_amount']:.2f}")
        print(f"  Expected: ${case['actual_reimbursement']:.2f} (${case['actual_per_day']:.2f}/day)")
        print(f"  Cost per day: ${case['cost_per_day']:.2f}")
        print(f"  Daily mileage: {case['daily_mileage']:.1f}")
        
        # Find very similar cases
        similar = df[
            (df['trip_duration_days'] == case['trip_duration_days']) &
            (abs(df['cost_per_day'] - case['cost_per_day']) < 50) &
            (abs(df['daily_mileage'] - case['daily_mileage']) < 50)
        ]
        
        if len(similar) > 1:
            print(f"  Very similar cases:")
            for _, sim in similar.iterrows():
                if sim['case_id'] != case_id:
                    print(f"    Case {sim['case_id']}: {sim['trip_duration_days']} days, {sim['miles_traveled']} miles, ${sim['total_receipts_amount']:.2f} → ${sim['actual_reimbursement']:.2f}")
    
    print(f"\n" + "=" * 60)
    print("PATTERN ANALYSIS OF FAILURE CASES")
    print("=" * 60)
    
    # Pattern 1: High spending penalty cases
    print("\n1. High Spending Cases Analysis:")
    high_spending = df[df['cost_per_day'] > 400]
    low_reimbursement_high_spending = high_spending[high_spending['actual_per_day'] < 150]
    
    print(f"   High spending (>$400/day): {len(high_spending)} cases")
    print(f"   High spending BUT low reimbursement (<$150/day): {len(low_reimbursement_high_spending)} cases")
    
    if len(low_reimbursement_high_spending) > 0:
        print(f"   These cases get severely penalized:")
        for _, case in low_reimbursement_high_spending.head(5).iterrows():
            efficiency = case['daily_mileage']
            print(f"     Case {case['case_id']}: {case['trip_duration_days']} days, ${case['cost_per_day']:.0f}/day, {efficiency:.0f} mi/day → ${case['actual_per_day']:.0f}/day")
    
    # Pattern 2: Efficiency vs spending analysis
    print("\n2. Efficiency vs Spending Analysis:")
    
    # Low efficiency + high spending = major penalty
    low_eff_high_spend = df[(df['daily_mileage'] < 50) & (df['cost_per_day'] > 300)]
    print(f"   Low efficiency (<50 mi/day) + High spending (>$300/day): {len(low_eff_high_spend)} cases")
    if len(low_eff_high_spend) > 0:
        avg_per_day = low_eff_high_spend['actual_per_day'].mean()
        print(f"   Average reimbursement per day: ${avg_per_day:.2f}")
    
    # High efficiency + high spending = still good
    high_eff_high_spend = df[(df['daily_mileage'] > 200) & (df['cost_per_day'] > 300)]
    print(f"   High efficiency (>200 mi/day) + High spending (>$300/day): {len(high_eff_high_spend)} cases")
    if len(high_eff_high_spend) > 0:
        avg_per_day = high_eff_high_spend['actual_per_day'].mean()
        print(f"   Average reimbursement per day: ${avg_per_day:.2f}")
    
    # Pattern 3: Look for receipt penalty patterns
    print("\n3. Receipt Penalty Patterns:")
    
    # Very high receipts but low reimbursement
    high_receipts = df[df['total_receipts_amount'] > 2000]
    penalized_high_receipts = high_receipts[high_receipts['actual_reimbursement'] < 1000]
    
    print(f"   Very high receipts (>$2000): {len(high_receipts)} cases")
    print(f"   High receipts BUT low reimbursement (<$1000): {len(penalized_high_receipts)} cases")
    
    if len(penalized_high_receipts) > 0:
        print(f"   These get penalized:")
        for _, case in penalized_high_receipts.head(3).iterrows():
            print(f"     Case {case['case_id']}: ${case['total_receipts_amount']:.0f} receipts, {case['daily_mileage']:.0f} mi/day → ${case['actual_reimbursement']:.0f}")
    
    return df

def find_penalty_rules(df):
    """Find the exact penalty rules that explain the failure cases"""
    print(f"\n" + "=" * 60)
    print("PENALTY RULE DISCOVERY")
    print("=" * 60)
    
    # Rule 1: Waste penalty (high spending, low mileage)
    print("\n1. Waste Penalty Rule:")
    df['efficiency_spending_ratio'] = df['daily_mileage'] / (df['cost_per_day'] + 1)  # +1 to avoid division by zero
    
    # Find cases with very low efficiency/spending ratio
    waste_cases = df[df['efficiency_spending_ratio'] < 0.2]  # Less than 0.2 miles per dollar per day
    print(f"   Low efficiency/spending ratio (<0.2): {len(waste_cases)} cases")
    if len(waste_cases) > 0:
        avg_per_day = waste_cases['actual_per_day'].mean()
        normal_avg = df['actual_per_day'].mean()
        print(f"   Average reimbursement per day: ${avg_per_day:.2f} (vs normal ${normal_avg:.2f})")
        
        print(f"   Sample waste cases:")
        for _, case in waste_cases.head(3).iterrows():
            print(f"     Case {case['case_id']}: {case['daily_mileage']:.0f} mi/day, ${case['cost_per_day']:.0f}/day → ${case['actual_per_day']:.0f}/day")
    
    # Rule 2: Long trip high spending penalty
    print("\n2. Long Trip High Spending Penalty:")
    long_trips = df[df['trip_duration_days'] >= 8]
    long_high_spend = long_trips[long_trips['cost_per_day'] > 200]
    long_low_reimburse = long_high_spend[long_high_spend['actual_per_day'] < 100]
    
    print(f"   Long trips (≥8 days) with high spending (>$200/day): {len(long_high_spend)} cases")
    print(f"   Of those, getting low reimbursement (<$100/day): {len(long_low_reimburse)} cases")
    
    if len(long_low_reimburse) > 0:
        print(f"   Sample penalized long trips:")
        for _, case in long_low_reimburse.head(3).iterrows():
            print(f"     Case {case['case_id']}: {case['trip_duration_days']} days, ${case['cost_per_day']:.0f}/day → ${case['actual_per_day']:.0f}/day")

def suggest_improvements():
    print(f"\n" + "=" * 60)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 60)
    
    print("\n1. Add Waste Penalty Rule:")
    print("   - When daily_mileage / cost_per_day < 0.2, apply major penalty")
    print("   - This catches cases like Case 152 (17 mi/day, $580/day)")
    
    print("\n2. Enhance Long Trip Penalties:")
    print("   - Long trips (≥8 days) with high spending get reduced per-day rates")
    print("   - Current decision tree may be too generous for these cases")
    
    print("\n3. Add Receipt Reasonableness Check:")
    print("   - Very high receipts with low productivity should be capped")
    print("   - Current tree may not capture all receipt penalty scenarios")
    
    print("\n4. Consider Ensemble Approach:")
    print("   - Combine decision tree with penalty adjustments")
    print("   - Use tree as base, then apply specific penalty rules")

if __name__ == "__main__":
    df = analyze_failure_cases()
    find_penalty_rules(df)
    suggest_improvements()