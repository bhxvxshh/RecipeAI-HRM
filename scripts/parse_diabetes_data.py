"""
Parse diabetes patient data into health profiles for recipe generation.

Data from UCI Machine Learning Repository:
70 patients with glucose, insulin, meals, and exercise logs.
"""

import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import json

# Code mappings from Data-Codes file
CODES = {
    33: 'regular_insulin',
    34: 'nph_insulin', 
    35: 'ultralente_insulin',
    48: 'glucose_unspecified',
    57: 'glucose_unspecified_2',
    58: 'glucose_pre_breakfast',
    59: 'glucose_post_breakfast',
    60: 'glucose_pre_lunch',
    61: 'glucose_post_lunch',
    62: 'glucose_pre_supper',
    63: 'glucose_post_supper',
    64: 'glucose_pre_snack',
    65: 'hypoglycemic_symptoms',
    66: 'typical_meal',
    67: 'large_meal',
    68: 'small_meal',
    69: 'typical_exercise',
    70: 'more_exercise',
    71: 'less_exercise',
    72: 'special_event'
}

def parse_patient_file(filepath):
    """Parse a single patient data file."""
    records = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                date, time, code, value = parts
                records.append({
                    'date': date,
                    'time': time,
                    'code': int(code),
                    'code_name': CODES.get(int(code), 'unknown'),
                    'value': int(value)
                })
    
    return pd.DataFrame(records)

def extract_health_profile(patient_id, df):
    """Extract health metrics from patient logs."""
    
    # Calculate average glucose levels
    glucose_codes = [48, 57, 58, 59, 60, 61, 62, 63, 64]
    glucose_readings = df[df['code'].isin(glucose_codes)]
    avg_glucose = glucose_readings['value'].mean() if len(glucose_readings) > 0 else None
    
    # Calculate average insulin doses
    insulin_codes = [33, 34, 35]
    insulin_doses = df[df['code'].isin(insulin_codes)]
    avg_insulin = insulin_doses['value'].mean() if len(insulin_doses) > 0 else None
    
    # Meal patterns
    meal_codes = [66, 67, 68]
    meal_count = len(df[df['code'].isin(meal_codes)])
    
    # Exercise patterns
    exercise_codes = [69, 70, 71]
    exercise_count = len(df[df['code'].isin(exercise_codes)])
    
    # Activity level based on exercise frequency
    days_tracked = len(df['date'].unique())
    exercise_per_day = exercise_count / days_tracked if days_tracked > 0 else 0
    
    if exercise_per_day >= 1.0:
        activity_level = 'very_active'
    elif exercise_per_day >= 0.5:
        activity_level = 'active'
    elif exercise_per_day >= 0.2:
        activity_level = 'moderate'
    else:
        activity_level = 'light'
    
    # Glucose control status
    if avg_glucose and avg_glucose > 180:
        glucose_control = 'poor'
    elif avg_glucose and avg_glucose > 140:
        glucose_control = 'fair'
    elif avg_glucose and avg_glucose >= 70:
        glucose_control = 'good'
    else:
        glucose_control = 'unknown'
    
    profile = {
        'patient_id': patient_id,
        'days_tracked': days_tracked,
        'avg_glucose_mg_dl': round(avg_glucose, 1) if avg_glucose else None,
        'avg_insulin_units': round(avg_insulin, 1) if avg_insulin else None,
        'meals_logged': meal_count,
        'exercise_events': exercise_count,
        'activity_level': activity_level,
        'glucose_control': glucose_control,
        'has_diabetes': True,
        'dietary_restrictions': ['diabetic_friendly', 'low_sugar', 'complex_carbs'],
        
        # Estimated demographics (since not in data)
        # Type 1 diabetes typically diagnosed younger, Type 2 older
        'estimated_age': 45 if avg_insulin and avg_insulin > 20 else 35,
        'estimated_weight_kg': 75,  # Average
        'estimated_height_cm': 170,  # Average
        'gender': 'unknown',
        'health_goal': 'maintain',  # Diabetics need stable weight
        
        # Nutritional targets for diabetics
        'target_carbs_percent': 45,  # Lower than standard 50-55%
        'target_fiber_g': 30,  # High fiber for blood sugar control
        'target_sugar_g': 25,  # Low added sugar
    }
    
    return profile

def main():
    """Process all 70 patient files."""
    
    data_dir = '/home/bhavesh/MajorB/RecipeAI/data/health_datasets/Diabetes-Data'
    output_file = '/home/bhavesh/MajorB/RecipeAI/data/health_profiles_diabetes.csv'
    
    all_profiles = []
    
    print("Processing 70 diabetes patient files...")
    
    for i in range(1, 71):
        filename = f"data-{i:02d}"
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                df = parse_patient_file(filepath)
                profile = extract_health_profile(f"diabetes_{i:03d}", df)
                all_profiles.append(profile)
                
                if i % 10 == 0:
                    print(f"  Processed {i}/70 patients...")
                    
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
    
    # Create DataFrame
    profiles_df = pd.DataFrame(all_profiles)
    
    # Save to CSV
    profiles_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Extracted {len(profiles_df)} health profiles")
    print(f"✓ Saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("HEALTH PROFILE SUMMARY")
    print("="*60)
    print(f"Total patients: {len(profiles_df)}")
    print(f"\nGlucose Control:")
    print(profiles_df['glucose_control'].value_counts())
    print(f"\nActivity Levels:")
    print(profiles_df['activity_level'].value_counts())
    print(f"\nAverage glucose: {profiles_df['avg_glucose_mg_dl'].mean():.1f} mg/dL")
    print(f"Average insulin: {profiles_df['avg_insulin_units'].mean():.1f} units")
    print(f"Average days tracked: {profiles_df['days_tracked'].mean():.1f}")
    
    # Show sample profiles
    print("\n" + "="*60)
    print("SAMPLE PROFILES (first 3)")
    print("="*60)
    print(profiles_df[['patient_id', 'avg_glucose_mg_dl', 'avg_insulin_units', 
                       'activity_level', 'glucose_control']].head(3).to_string(index=False))
    
    return profiles_df

if __name__ == "__main__":
    profiles = main()
