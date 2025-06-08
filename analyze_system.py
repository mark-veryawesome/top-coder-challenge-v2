#!/usr/bin/env python3
"""
Travel Reimbursement System Analysis
Evidence-based reverse engineering with contradiction resolution
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ContradictionResolver:
    """Handles contradictions between data sources using evidence hierarchy"""
    
    def __init__(self):
        self.evidence_hierarchy = {
            'data': 0.7,      # Actual patterns in historical data
            'prd': 0.2,       # Official policy document  
            'interview': 0.1   # Employee statements (when consensus)
        }
        self.validated_claims = {}
        self.contradictions = {}
    
    def extract_interview_claims(self) -> Dict[str, List[str]]:
        """Extract specific claims from interview text"""
        claims = {
            'Marcus': [
                'sweet_spot_5_6_days',
                'efficiency_rewards_hustle', 
                'mileage_drops_after_standard_rate',
                'receipt_cap_penalty_exists',
                'quarterly_q4_more_generous',
                'system_remembers_history'
            ],
            'Lisa': [
                'per_diem_base_100_per_day',
                '5_day_trips_get_bonus',
                'mileage_tiered_58_cents_first_100',
                'receipt_cap_not_linear',
                'small_receipts_penalized_vs_none',
                'efficiency_bonus_real'
            ],
            'Dave': [
                'small_receipts_worse_than_none',
                'randomness_prevents_gaming',
                'receipt_amounts_mixed_results'
            ],
            'Jennifer': [
                'sweet_spot_4_6_days',
                'new_employees_lower_reimbursements',
                'timing_matters_but_unclear'
            ],
            'Kevin': [
                'efficiency_sweet_spot_180_220_miles_per_day',
                'spending_ranges_by_trip_length',
                'short_trips_under_75_per_day',
                'medium_trips_up_to_120_per_day', 
                'long_trips_under_90_per_day',
                '5_day_180_miles_under_100_guaranteed_bonus',
                'tuesday_submissions_8_percent_higher',
                'lunar_cycle_correlation_4_percent',
                'six_calculation_paths_exist'
            ]
        }
        return claims
    
    def test_claim_against_data(self, claim: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Test specific claims against actual data patterns"""
        results = {'claim': claim, 'supported': False, 'confidence': 0.0, 'evidence': ''}
        
        try:
            if 'per_diem_base_100' in claim:
                # Test base per diem rate
                daily_rates = data['reimbursement'] / data['trip_duration_days']
                avg_rate = daily_rates.mean()
                results['supported'] = 90 <= avg_rate <= 110
                results['confidence'] = 1.0 - abs(avg_rate - 100) / 100
                results['evidence'] = f'Average daily rate: ${avg_rate:.2f}'
                
            elif 'efficiency_sweet_spot_180_220' in claim:
                # Test efficiency sweet spot claim
                data_copy = data.copy()
                data_copy['efficiency'] = data_copy['miles_traveled'] / data_copy['trip_duration_days']
                sweet_spot = data_copy[(data_copy['efficiency'] >= 180) & (data_copy['efficiency'] <= 220)]
                others = data_copy[(data_copy['efficiency'] < 180) | (data_copy['efficiency'] > 220)]
                
                if len(sweet_spot) > 0 and len(others) > 0:
                    sweet_spot_avg = (sweet_spot['reimbursement'] / sweet_spot['trip_duration_days']).mean()
                    others_avg = (others['reimbursement'] / others['trip_duration_days']).mean()
                    improvement = (sweet_spot_avg - others_avg) / others_avg
                    results['supported'] = improvement > 0.05  # 5% improvement threshold
                    results['confidence'] = min(improvement * 2, 1.0)  # Cap at 1.0
                    results['evidence'] = f'Sweet spot avg: ${sweet_spot_avg:.2f}, Others: ${others_avg:.2f}'
                
            elif '5_day' in claim and 'bonus' in claim:
                # Test 5-day trip bonus
                five_day = data[data['trip_duration_days'] == 5]
                other_days = data[data['trip_duration_days'].isin([4, 6])]  # Compare to similar lengths
                
                if len(five_day) > 0 and len(other_days) > 0:
                    five_day_rate = (five_day['reimbursement'] / five_day['trip_duration_days']).mean()
                    other_rate = (other_days['reimbursement'] / other_days['trip_duration_days']).mean()
                    improvement = (five_day_rate - other_rate) / other_rate
                    results['supported'] = improvement > 0.02  # 2% improvement threshold
                    results['confidence'] = min(improvement * 5, 1.0)
                    results['evidence'] = f'5-day rate: ${five_day_rate:.2f}, 4&6-day rate: ${other_rate:.2f}'
                    
            elif 'small_receipts' in claim and 'penalized' in claim:
                # Test small receipt penalty
                data_copy = data.copy()
                data_copy['receipt_per_day'] = data_copy['total_receipts_amount'] / data_copy['trip_duration_days']
                small_receipts = data_copy[(data_copy['receipt_per_day'] > 0) & (data_copy['receipt_per_day'] < 25)]
                no_receipts = data_copy[data_copy['total_receipts_amount'] == 0]
                
                if len(small_receipts) > 0 and len(no_receipts) > 0:
                    small_rate = (small_receipts['reimbursement'] / small_receipts['trip_duration_days']).mean()
                    no_receipt_rate = (no_receipts['reimbursement'] / no_receipts['trip_duration_days']).mean()
                    penalty = (no_receipt_rate - small_rate) / no_receipt_rate
                    results['supported'] = penalty > 0.01  # Small receipts get worse treatment
                    results['confidence'] = min(penalty * 10, 1.0)
                    results['evidence'] = f'Small receipts: ${small_rate:.2f}, No receipts: ${no_receipt_rate:.2f}'
                    
        except Exception as e:
            results['evidence'] = f'Analysis error: {str(e)}'
            
        return results
    
    def find_consensus_claims(self, claims: Dict[str, List[str]]) -> List[str]:
        """Find claims supported by multiple employees"""
        claim_counts = {}
        for employee, employee_claims in claims.items():
            for claim in employee_claims:
                if claim not in claim_counts:
                    claim_counts[claim] = []
                claim_counts[claim].append(employee)
        
        # Return claims mentioned by 2+ employees
        consensus = [claim for claim, supporters in claim_counts.items() if len(supporters) >= 2]
        return consensus

class ReimbursementAnalyzer:
    """Main analysis class for reverse-engineering the reimbursement system"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.data = None
        self.features = None
        self.resolver = ContradictionResolver()
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load data and create interpretable features"""
        print("Loading public cases data...")
        with open(self.data_file, 'r') as f:
            cases = json.load(f)
        
        # Convert to DataFrame
        rows = []
        for case in cases:
            input_data = case['input']
            rows.append({
                'trip_duration_days': input_data['trip_duration_days'],
                'miles_traveled': input_data['miles_traveled'], 
                'total_receipts_amount': input_data['total_receipts_amount'],
                'reimbursement': case['expected_output']
            })
        
        self.data = pd.DataFrame(rows)
        
        # Create interpretable features
        self.data['cost_per_day'] = self.data['total_receipts_amount'] / self.data['trip_duration_days']
        self.data['cost_per_mile'] = np.where(self.data['miles_traveled'] > 0, 
                                            self.data['total_receipts_amount'] / self.data['miles_traveled'], 0)
        self.data['daily_mileage'] = self.data['miles_traveled'] / self.data['trip_duration_days']
        self.data['receipt_ratio'] = self.data['total_receipts_amount'] / self.data['reimbursement']
        self.data['reimbursement_per_day'] = self.data['reimbursement'] / self.data['trip_duration_days']
        self.data['reimbursement_per_mile'] = np.where(self.data['miles_traveled'] > 0,
                                                     self.data['reimbursement'] / self.data['miles_traveled'], 0)
        
        print(f"Loaded {len(self.data)} cases")
        return self.data
    
    def analyze_basic_patterns(self):
        """Analyze basic statistical patterns in the data"""
        print("\n" + "="*50)
        print("BASIC DATA ANALYSIS")
        print("="*50)
        
        print("\nInput Statistics:")
        print(self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].describe())
        
        print("\nOutput Statistics:")
        print(self.data[['reimbursement']].describe())
        
        print("\nDerived Feature Statistics:")
        print(self.data[['cost_per_day', 'daily_mileage', 'reimbursement_per_day']].describe())
        
        # Look for patterns in trip duration
        print("\nReimbursement by Trip Duration:")
        duration_analysis = self.data.groupby('trip_duration_days').agg({
            'reimbursement': ['count', 'mean', 'std'],
            'reimbursement_per_day': 'mean'
        }).round(2)
        print(duration_analysis)
        
        # Correlation analysis
        print("\nCorrelation Matrix:")
        correlation_cols = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 
                          'daily_mileage', 'cost_per_day', 'reimbursement']
        corr_matrix = self.data[correlation_cols].corr()
        print(corr_matrix.round(3))
    
    def test_interview_claims(self):
        """Test specific claims from interviews against data"""
        print("\n" + "="*50)
        print("INTERVIEW CLAIM VALIDATION")
        print("="*50)
        
        claims = self.resolver.extract_interview_claims()
        consensus_claims = self.resolver.find_consensus_claims(claims)
        
        print(f"\nFound {len(consensus_claims)} consensus claims to test:")
        for claim in consensus_claims:
            print(f"  - {claim}")
        
        print("\nTesting claims against data:")
        claim_results = {}
        
        # Test all unique claims
        all_claims = set()
        for employee_claims in claims.values():
            all_claims.update(employee_claims)
        
        for claim in sorted(all_claims):
            result = self.resolver.test_claim_against_data(claim, self.data)
            claim_results[claim] = result
            
            status = "✓ SUPPORTED" if result['supported'] else "✗ NOT SUPPORTED"
            confidence = result['confidence']
            evidence = result['evidence']
            
            print(f"\n{claim}:")
            print(f"  {status} (confidence: {confidence:.2f})")
            print(f"  Evidence: {evidence}")
        
        return claim_results
    
    def discover_data_patterns(self):
        """Discover patterns directly from data without interview bias"""
        print("\n" + "="*50)
        print("DATA-DRIVEN PATTERN DISCOVERY")
        print("="*50)
        
        # Pattern 1: Base rates analysis
        print("\n1. Base Rate Analysis:")
        base_cases = self.data[(self.data['total_receipts_amount'] == 0) & (self.data['miles_traveled'] < 50)]
        if len(base_cases) > 0:
            print(f"   Base per diem (no receipts, low miles): ${base_cases['reimbursement_per_day'].mean():.2f}")
        
        # Pattern 2: Mileage rate analysis  
        print("\n2. Mileage Rate Analysis:")
        mileage_bins = [0, 100, 200, 400, 800, float('inf')]
        mileage_labels = ['0-100', '100-200', '200-400', '400-800', '800+']
        self.data['mileage_bin'] = pd.cut(self.data['miles_traveled'], bins=mileage_bins, labels=mileage_labels)
        
        mileage_analysis = self.data.groupby('mileage_bin').agg({
            'reimbursement_per_mile': ['count', 'mean', 'std']
        }).round(3)
        print(mileage_analysis)
        
        # Pattern 3: Receipt coverage analysis
        print("\n3. Receipt Coverage Analysis:")
        receipt_bins = [0, 50, 100, 200, 500, float('inf')]
        receipt_labels = ['0-50', '50-100', '100-200', '200-500', '500+']
        self.data['receipt_bin'] = pd.cut(self.data['total_receipts_amount'], bins=receipt_bins, labels=receipt_labels)
        
        receipt_analysis = self.data.groupby('receipt_bin').agg({
            'receipt_ratio': ['count', 'mean', 'std']
        }).round(3)
        print(receipt_analysis)
        
        # Pattern 4: Efficiency analysis
        print("\n4. Efficiency (Miles/Day) Analysis:")
        efficiency_bins = [0, 100, 150, 200, 250, float('inf')]
        efficiency_labels = ['0-100', '100-150', '150-200', '200-250', '250+']
        self.data['efficiency_bin'] = pd.cut(self.data['daily_mileage'], bins=efficiency_bins, labels=efficiency_labels)
        
        efficiency_analysis = self.data.groupby('efficiency_bin').agg({
            'reimbursement_per_day': ['count', 'mean', 'std']
        }).round(2)
        print(efficiency_analysis)
    
    def build_interpretable_models(self):
        """Build and compare interpretable models"""
        print("\n" + "="*50)
        print("INTERPRETABLE MODEL BUILDING")
        print("="*50)
        
        # Prepare features
        X = self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].values
        y = self.data['reimbursement'].values
        
        # Model 1: Linear Regression
        print("\n1. Linear Regression:")
        linear_model = LinearRegression()
        cv_scores = cross_val_score(linear_model, X, y, cv=5, scoring='neg_mean_absolute_error')
        linear_model.fit(X, y)
        
        print(f"   CV MAE: ${-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        print(f"   R²: {linear_model.score(X, y):.3f}")
        print("   Coefficients:")
        feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        for name, coef in zip(feature_names, linear_model.coef_):
            print(f"     {name}: {coef:.3f}")
        print(f"   Intercept: {linear_model.intercept_:.3f}")
        
        # Model 2: Polynomial Features (degree 2)
        print("\n2. Polynomial Features (degree 2):")
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ])
        cv_scores = cross_val_score(poly_model, X, y, cv=5, scoring='neg_mean_absolute_error')
        poly_model.fit(X, y)
        
        print(f"   CV MAE: ${-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        print(f"   R²: {poly_model.score(X, y):.3f}")
        
        # Model 3: Shallow Decision Tree
        print("\n3. Shallow Decision Tree (max_depth=4):")
        tree_model = DecisionTreeRegressor(max_depth=4, min_samples_split=50, min_samples_leaf=20, random_state=42)
        cv_scores = cross_val_score(tree_model, X, y, cv=5, scoring='neg_mean_absolute_error')
        tree_model.fit(X, y)
        
        print(f"   CV MAE: ${-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        print(f"   R²: {tree_model.score(X, y):.3f}")
        print("   Feature Importances:")
        for name, importance in zip(feature_names, tree_model.feature_importances_):
            print(f"     {name}: {importance:.3f}")
        
        return {
            'linear': linear_model,
            'polynomial': poly_model, 
            'tree': tree_model
        }
    
    def extract_business_rules(self, models):
        """Extract interpretable business rules from models"""
        print("\n" + "="*50)
        print("BUSINESS RULE EXTRACTION")
        print("="*50)
        
        # From linear model
        linear_model = models['linear']
        print("\nLinear Model Business Rules:")
        print(f"  Base amount: ${linear_model.intercept_:.2f}")
        print(f"  Per day rate: ${linear_model.coef_[0]:.2f}")
        print(f"  Per mile rate: ${linear_model.coef_[1]:.3f}")
        print(f"  Receipt coverage: {linear_model.coef_[2]:.2f} (ratio)")
        
        # From tree model - extract rules
        tree_model = models['tree']
        print("\nDecision Tree Rules:")
        self._extract_tree_rules(tree_model)
        
        # Validate rules against business context
        print("\nRule Validation:")
        per_day_rate = linear_model.coef_[0]
        per_mile_rate = linear_model.coef_[1] 
        receipt_coverage = linear_model.coef_[2]
        
        print(f"  ✓ Per day rate ${per_day_rate:.2f} {'reasonable' if 50 <= per_day_rate <= 150 else 'questionable'}")
        print(f"  ✓ Per mile rate ${per_mile_rate:.3f} {'reasonable' if 0.3 <= per_mile_rate <= 0.8 else 'questionable'}")
        print(f"  ✓ Receipt coverage {receipt_coverage:.2f} {'reasonable' if 0.5 <= receipt_coverage <= 1.2 else 'questionable'}")
    
    def _extract_tree_rules(self, tree_model):
        """Extract human-readable rules from decision tree"""
        from sklearn.tree import export_text
        feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        tree_rules = export_text(tree_model, feature_names=feature_names, max_depth=4)
        print("  Decision paths:")
        for line in tree_rules.split('\n')[:20]:  # Show first 20 lines
            if line.strip():
                print(f"    {line}")
    
    def run_full_analysis(self):
        """Run complete analysis workflow"""
        print("TRAVEL REIMBURSEMENT SYSTEM ANALYSIS")
        print("Evidence-Based Reverse Engineering with Contradiction Resolution")
        print("="*80)
        
        # Phase 1: Data loading and basic analysis
        self.load_and_prepare_data()
        self.analyze_basic_patterns()
        
        # Phase 2: Interview claim validation
        claim_results = self.test_interview_claims()
        
        # Phase 3: Data-driven pattern discovery
        self.discover_data_patterns()
        
        # Phase 4: Model building and rule extraction
        models = self.build_interpretable_models()
        self.extract_business_rules(models)
        
        return {
            'data': self.data,
            'claim_results': claim_results,
            'models': models
        }

def main():
    analyzer = ReimbursementAnalyzer('public_cases.json')
    results = analyzer.run_full_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review claim validation results to identify data-supported patterns")
    print("2. Focus on linear model coefficients for interpretable business rules")
    print("3. Implement the most robust and generalizable rules in run.sh")
    print("4. Test against public cases using ./eval.sh")

if __name__ == "__main__":
    main()