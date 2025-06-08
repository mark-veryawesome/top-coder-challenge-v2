# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reverse-engineering challenge for a legacy travel reimbursement system. The goal is to recreate a 60-year-old black box system's behavior by analyzing 1,000 historical input/output examples and employee interviews.

**Core Task**: Build a script that takes 3 inputs (trip_duration_days, miles_traveled, total_receipts_amount) and outputs a single reimbursement amount that matches the legacy system's calculations.

## Essential Commands

### Development and Testing
- `chmod +x run.sh` - Make your implementation executable
- `./eval.sh` - Test implementation against 1,000 public cases with expected outputs
- `./generate_results.sh` - Generate final results for 5,000 private cases (for submission)

### Implementation Requirements
- Copy `run.sh.template` to `run.sh` and implement your logic
- Script must output only a single number (reimbursement amount)
- Must run in under 5 seconds per test case
- No external dependencies (network, databases)

## Data Structure

### Input Format
- `trip_duration_days`: integer (number of days traveling)
- `miles_traveled`: integer (total miles traveled)  
- `total_receipts_amount`: float (total dollar amount of receipts)

### Output Format
- Single float rounded to 2 decimal places (reimbursement amount)

### Test Data Files
- `public_cases.json`: 1,000 cases with inputs and expected outputs for development
- `private_cases.json`: 5,000 cases with inputs only (for final evaluation)

## Key Business Logic Insights from Interviews

### Critical Patterns Identified
1. **Efficiency Bonuses**: High miles-per-day ratios are rewarded (sweet spot: 180-220 miles/day)
2. **Trip Duration Sweet Spot**: 5-day trips consistently get bonuses; 4-6 day range is optimal
3. **Spending Thresholds**: Optimal spending varies by trip length:
   - Short trips: <$75/day
   - Medium trips (4-6 days): up to $120/day  
   - Long trips: <$90/day
4. **Small Receipt Penalty**: Very low receipt amounts often get penalized vs. no receipts
5. **Receipt Caps**: Diminishing returns on high receipt amounts, with complex non-linear scaling

### Calculation Complexity
- At least 6 different calculation paths based on trip characteristics
- Interaction effects between factors (not just individual inputs)
- Threshold effects that trigger bonuses/penalties
- Possible randomization/noise to prevent gaming

### Implementation Strategy
1. Start with base per diem (~$100/day) and mileage rates (~$0.58/mile for first 100 miles)
2. Apply efficiency bonuses for optimal miles/day ratios
3. Implement spending threshold logic with trip-length dependencies  
4. Add special handling for 5-day trips and other duration-based bonuses
5. Handle receipt penalties for very low amounts
6. Test against public cases and refine based on error patterns

## Development Notes

- The system appears to have intentional complexity to prevent gaming
- Employee interviews reveal conflicting theories but consistent patterns around efficiency and spending thresholds
- Focus on the mathematical relationships rather than trying to implement every anecdotal theory
- Use `eval.sh` iteratively to identify which patterns matter most for accuracy