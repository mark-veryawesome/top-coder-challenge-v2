#!/usr/bin/env python3
"""
Advanced Reimbursement Analysis
Focus on data-validated patterns and robust generalization
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedReimbursementAnalyzer:
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.data = None
        self.models = {}
        
    def load_data(self):
        """Load and preprocess data"""
        with open(self.data_file, 'r') as f:
            cases = json.load(f)
        
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
        
        # Add derived features based on data insights
        self.data['daily_mileage'] = self.data['miles_traveled'] / self.data['trip_duration_days']
        self.data['cost_per_day'] = self.data['total_receipts_amount'] / self.data['trip_duration_days']
        self.data['reimbursement_per_day'] = self.data['reimbursement'] / self.data['trip_duration_days']
        
        return self.data
    
    def analyze_key_patterns(self):
        """Focus on the most important data-driven patterns"""
        print("ADVANCED PATTERN ANALYSIS")
        print("="*50)
        
        # Pattern 1: The efficiency bonus is REAL - strong correlation
        print("\n1. EFFICIENCY ANALYSIS (Miles/Day):")
        efficiency_ranges = [(0, 100), (100, 150), (150, 200), (200, 250), (250, float('inf'))]
        
        for i, (low, high) in enumerate(efficiency_ranges):
            if high == float('inf'):
                subset = self.data[self.data['daily_mileage'] >= low]
                range_label = f"{low}+"
            else:
                subset = self.data[(self.data['daily_mileage'] >= low) & (self.data['daily_mileage'] < high)]
                range_label = f"{low}-{high}"
            
            if len(subset) > 0:
                avg_reimbursement_per_day = subset['reimbursement_per_day'].mean()
                count = len(subset)
                print(f"   {range_label} miles/day: ${avg_reimbursement_per_day:.2f}/day (n={count})")
        
        # Pattern 2: Receipt coverage varies dramatically
        print("\n2. RECEIPT COVERAGE ANALYSIS:")
        receipt_ranges = [(0, 0.1), (0.1, 500), (500, 1000), (1000, 1500), (1500, float('inf'))]
        
        for low, high in receipt_ranges:
            if high == float('inf'):
                subset = self.data[self.data['total_receipts_amount'] >= low]
                range_label = f"${low}+"
            else:
                subset = self.data[(self.data['total_receipts_amount'] >= low) & (self.data['total_receipts_amount'] < high)]
                range_label = f"${low}-{high}"
            
            if len(subset) > 0:
                avg_coverage = (subset['total_receipts_amount'] / subset['reimbursement']).mean()
                count = len(subset)
                print(f"   {range_label} receipts: {avg_coverage:.2f} coverage ratio (n={count})")
        
        # Pattern 3: Trip length matters but not linearly
        print("\n3. TRIP LENGTH ANALYSIS:")
        trip_ranges = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 15)]
        
        for low, high in trip_ranges:
            subset = self.data[(self.data['trip_duration_days'] >= low) & (self.data['trip_duration_days'] <= high)]
            if len(subset) > 0:
                avg_per_day = subset['reimbursement_per_day'].mean()
                count = len(subset)
                print(f"   {low}-{high} days: ${avg_per_day:.2f}/day (n={count})")
    
    def discover_interaction_effects(self):
        """Look for interaction effects between variables"""
        print("\n4. INTERACTION EFFECTS:")
        
        # High efficiency + low receipts
        high_eff_low_rec = self.data[(self.data['daily_mileage'] > 200) & (self.data['total_receipts_amount'] < 500)]
        if len(high_eff_low_rec) > 0:
            avg_rate = high_eff_low_rec['reimbursement_per_day'].mean()
            print(f"   High efficiency (>200 mi/day) + Low receipts (<$500): ${avg_rate:.2f}/day")
        
        # Medium efficiency + high receipts
        med_eff_high_rec = self.data[(self.data['daily_mileage'].between(100, 200)) & (self.data['total_receipts_amount'] > 1500)]
        if len(med_eff_high_rec) > 0:
            avg_rate = med_eff_high_rec['reimbursement_per_day'].mean()
            print(f"   Medium efficiency (100-200 mi/day) + High receipts (>$1500): ${avg_rate:.2f}/day")
        
        # Short trips vs long trips with similar spending
        short_trips = self.data[(self.data['trip_duration_days'] <= 3) & (self.data['cost_per_day'].between(100, 200))]
        long_trips = self.data[(self.data['trip_duration_days'] >= 10) & (self.data['cost_per_day'].between(100, 200))]
        
        if len(short_trips) > 0 and len(long_trips) > 0:
            short_rate = short_trips['reimbursement_per_day'].mean()
            long_rate = long_trips['reimbursement_per_day'].mean()
            print(f"   Short trips (≤3 days) at $100-200/day: ${short_rate:.2f}/day")
            print(f"   Long trips (≥10 days) at $100-200/day: ${long_rate:.2f}/day")
    
    def build_robust_models(self):
        """Build models optimized for generalization"""
        print("\n" + "="*50)
        print("ROBUST MODEL DEVELOPMENT")
        print("="*50)
        
        # Prepare features
        feature_sets = {
            'basic': ['trip_duration_days', 'miles_traveled', 'total_receipts_amount'],
            'enhanced': ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'daily_mileage', 'cost_per_day'],
            'interaction': ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'daily_mileage', 'cost_per_day']
        }
        
        y = self.data['reimbursement'].values
        
        results = {}
        
        for feature_name, feature_list in feature_sets.items():
            print(f"\n{feature_name.upper()} FEATURES: {feature_list}")
            
            if feature_name == 'interaction':
                # Add interaction terms manually for better control
                X_base = self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].values
                X_derived = self.data[['daily_mileage', 'cost_per_day']].values
                
                # Add selected interactions based on domain knowledge
                efficiency_cost_interaction = (self.data['daily_mileage'] * self.data['cost_per_day']).values.reshape(-1, 1)
                miles_duration_interaction = (self.data['miles_traveled'] * self.data['trip_duration_days']).values.reshape(-1, 1)
                
                X = np.hstack([X_base, X_derived, efficiency_cost_interaction, miles_duration_interaction])
                feature_names = feature_list + ['efficiency_cost_interaction', 'miles_duration_interaction']
            else:
                X = self.data[feature_list].values
                feature_names = feature_list
            
            # Test multiple model types
            models_to_test = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=10.0),
                'lasso': Lasso(alpha=5.0),
                'tree': DecisionTreeRegressor(max_depth=6, min_samples_split=30, min_samples_leaf=15, random_state=42)
            }
            
            feature_results = {}
            
            for model_name, model in models_to_test.items():
                # 10-fold cross-validation for robust estimation
                cv_scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')
                
                # Fit on full data to get coefficients
                model.fit(X, y)
                
                feature_results[model_name] = {
                    'cv_mae': -cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'r2': model.score(X, y),
                    'model': model
                }
                
                print(f"   {model_name}: MAE=${-cv_scores.mean():.2f}±{cv_scores.std():.2f}, R²={model.score(X, y):.3f}")
                
                # Show coefficients for linear models
                if hasattr(model, 'coef_'):
                    print(f"     Coefficients:")
                    for name, coef in zip(feature_names, model.coef_):
                        print(f"       {name}: {coef:.4f}")
                    if hasattr(model, 'intercept_'):
                        print(f"       intercept: {model.intercept_:.4f}")
            
            results[feature_name] = feature_results
        
        self.models = results
        return results
    
    def validate_best_model(self):
        """Validate the best performing model"""
        print("\n" + "="*50)
        print("BEST MODEL VALIDATION")
        print("="*50)
        
        # Find best model across all feature sets
        best_mae = float('inf')
        best_model_info = None
        
        for feature_set, models in self.models.items():
            for model_name, model_info in models.items():
                if model_info['cv_mae'] < best_mae:
                    best_mae = model_info['cv_mae']
                    best_model_info = (feature_set, model_name, model_info)
        
        if best_model_info:
            feature_set, model_name, model_info = best_model_info
            print(f"Best model: {model_name} with {feature_set} features")
            print(f"CV MAE: ${model_info['cv_mae']:.2f} ± {model_info['cv_std']:.2f}")
            print(f"R²: {model_info['r2']:.3f}")
            
            # Additional validation: residual analysis
            model = model_info['model']
            
            if feature_set == 'basic':
                X = self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].values
            elif feature_set == 'enhanced':
                X = self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'daily_mileage', 'cost_per_day']].values
            else:  # interaction
                X_base = self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].values
                X_derived = self.data[['daily_mileage', 'cost_per_day']].values
                efficiency_cost_interaction = (self.data['daily_mileage'] * self.data['cost_per_day']).values.reshape(-1, 1)
                miles_duration_interaction = (self.data['miles_traveled'] * self.data['trip_duration_days']).values.reshape(-1, 1)
                X = np.hstack([X_base, X_derived, efficiency_cost_interaction, miles_duration_interaction])
            
            y = self.data['reimbursement'].values
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            print(f"\nResidual Analysis:")
            print(f"  Mean residual: ${residuals.mean():.2f}")
            print(f"  Residual std: ${residuals.std():.2f}")
            print(f"  Max absolute error: ${np.abs(residuals).max():.2f}")
            
            # Look for patterns in residuals
            large_errors = np.abs(residuals) > 200
            if np.any(large_errors):
                print(f"  Cases with large errors (>$200): {np.sum(large_errors)}")
                error_data = self.data[large_errors].copy()
                error_data['prediction'] = y_pred[large_errors]
                error_data['actual'] = y[large_errors]
                error_data['error'] = residuals[large_errors]
                
                print("  Sample large error cases:")
                print(error_data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 
                                'actual', 'prediction', 'error']].head())
        
        return best_model_info
    
    def generate_business_formula(self, best_model_info):
        """Generate implementable business formula"""
        print("\n" + "="*50)
        print("BUSINESS FORMULA GENERATION")
        print("="*50)
        
        feature_set, model_name, model_info = best_model_info
        model = model_info['model']
        
        if hasattr(model, 'coef_') and feature_set == 'basic':
            # Simple linear model we can implement directly
            coeffs = model.coef_
            intercept = model.intercept_
            
            print("IMPLEMENTABLE FORMULA:")
            print(f"reimbursement = {intercept:.2f}")
            print(f"               + {coeffs[0]:.3f} * trip_duration_days")
            print(f"               + {coeffs[1]:.3f} * miles_traveled") 
            print(f"               + {coeffs[2]:.3f} * total_receipts_amount")
            
            print(f"\nBUSINESS INTERPRETATION:")
            print(f"  Base reimbursement: ${intercept:.2f}")
            print(f"  Per day rate: ${coeffs[0]:.2f}")
            print(f"  Per mile rate: ${coeffs[1]:.3f}")
            print(f"  Receipt coverage: {coeffs[2]:.1%}")
            
            return {
                'intercept': intercept,
                'per_day': coeffs[0],
                'per_mile': coeffs[1], 
                'receipt_coverage': coeffs[2]
            }
        
        else:
            print("Best model is too complex for direct implementation.")
            print("Consider using the basic linear model with good performance.")
            
            # Fall back to basic linear model
            X_basic = self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].values
            y = self.data['reimbursement'].values
            basic_model = LinearRegression().fit(X_basic, y)
            
            coeffs = basic_model.coef_
            intercept = basic_model.intercept_
            
            print(f"\nFALLBACK LINEAR FORMULA:")
            print(f"reimbursement = {intercept:.2f}")
            print(f"               + {coeffs[0]:.3f} * trip_duration_days")
            print(f"               + {coeffs[1]:.3f} * miles_traveled") 
            print(f"               + {coeffs[2]:.3f} * total_receipts_amount")
            
            return {
                'intercept': intercept,
                'per_day': coeffs[0],
                'per_mile': coeffs[1], 
                'receipt_coverage': coeffs[2]
            }
    
    def run_analysis(self):
        """Run complete advanced analysis"""
        print("ADVANCED REIMBURSEMENT SYSTEM ANALYSIS")
        print("="*80)
        
        self.load_data()
        self.analyze_key_patterns()
        self.discover_interaction_effects()
        model_results = self.build_robust_models()
        best_model = self.validate_best_model()
        formula = self.generate_business_formula(best_model)
        
        return {
            'data': self.data,
            'models': model_results,
            'best_model': best_model,
            'formula': formula
        }

if __name__ == "__main__":
    analyzer = AdvancedReimbursementAnalyzer('public_cases.json')
    results = analyzer.run_analysis()