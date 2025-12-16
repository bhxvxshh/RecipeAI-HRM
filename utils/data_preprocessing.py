"""
Data preprocessing utilities for USDA FoodData Central
Converts raw food data into ingredient-level nutrient vectors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

def load_usda_data(data_dir=config.DATA_DIR):
    """
    Load raw USDA data files
    
    Returns:
        food_df: DataFrame with food descriptions
        nutrient_df: DataFrame with nutrient values
    """
    food_path = Path(data_dir) / config.RAW_FOOD_FILE
    nutrient_path = Path(data_dir) / config.RAW_NUTRIENT_FILE
    
    print(f"Loading USDA data from {data_dir}...")
    food_df = pd.read_csv(food_path)
    nutrient_df = pd.read_csv(nutrient_path)
    
    print(f"Loaded {len(food_df)} foods and {len(nutrient_df)} nutrient entries")
    return food_df, nutrient_df


def extract_nutrients(food_df, nutrient_df, nutrient_ids=config.NUTRIENT_IDS):
    """
    Extract specific nutrients for each food item
    
    Args:
        food_df: Food descriptions
        nutrient_df: Nutrient values
        nutrient_ids: Dict mapping nutrient names to USDA IDs
        
    Returns:
        DataFrame with columns: [food_id, food_name, calories, protein, sodium, carbs, fat]
    """
    print("Extracting nutrients...")
    
    # Filter for required nutrients only
    nutrient_subset = nutrient_df[nutrient_df['nutrient_id'].isin(nutrient_ids.values())].copy()
    
    # Keep only necessary columns
    nutrient_subset = nutrient_subset[['fdc_id', 'nutrient_id', 'amount']]
    
    # Pivot to get one row per food
    nutrients_pivot = nutrient_subset.pivot_table(
        index='fdc_id',
        columns='nutrient_id',
        values='amount',
        aggfunc='first'
    )
    
    # Rename columns using the correct mapping
    id_to_name = {v: k for k, v in nutrient_ids.items()}
    nutrients_pivot = nutrients_pivot.rename(columns=id_to_name)
    
    # Merge with food names
    result = food_df[['fdc_id', 'description']].merge(
        nutrients_pivot,
        left_on='fdc_id',
        right_index=True,
        how='inner'
    )
    
    # Rename first two columns
    result = result.rename(columns={'fdc_id': 'food_id', 'description': 'food_name'})
    
    print(f"Extracted nutrients for {len(result)} foods")
    return result


def clean_ingredients(df, min_completeness=0.6):
    """
    Clean ingredient data:
    - Remove entries with missing nutrients
    - Remove duplicates
    - Filter out non-ingredient items
    
    Args:
        df: DataFrame with nutrient data
        min_completeness: Minimum fraction of non-null nutrients required
        
    Returns:
        Cleaned DataFrame
    """
    print("Cleaning ingredient data...")
    
    # Remove rows with too many missing values
    nutrient_cols = ['calories', 'protein', 'sodium', 'carbs', 'fat']
    df['completeness'] = df[nutrient_cols].notna().sum(axis=1) / len(nutrient_cols)
    df = df[df['completeness'] >= min_completeness]
    
    # Fill remaining NaNs with 0
    df_clean = df.copy()
    df_clean[nutrient_cols] = df_clean[nutrient_cols].fillna(0)
    
    # Remove negative values (data errors)
    for col in nutrient_cols:
        df_clean = df_clean[df_clean[col] >= 0]
    
    # Remove extreme outliers (likely data entry errors)
    df_clean = df_clean[df_clean['calories'] <= 900]  # Max 900 cal per 100g
    df_clean = df_clean[df_clean['protein'] <= 100]
    df_clean = df_clean[df_clean['sodium'] <= 5000]
    df_clean = df_clean[df_clean['carbs'] <= 100]
    df_clean = df_clean[df_clean['fat'] <= 100]
    
    # Remove entries with all zeros (likely invalid)
    df_clean = df_clean[(df_clean[nutrient_cols].sum(axis=1) > 0)]
    
    # Remove duplicates based on similar nutrient profiles
    df_clean = df_clean.drop_duplicates(subset=nutrient_cols, keep='first')
    
    print(f"After cleaning: {len(df_clean)} ingredients")
    return df_clean.drop('completeness', axis=1)


def select_ingredient_pool(df, pool_size=config.INGREDIENT_POOL_SIZE):
    """
    Select top N ingredients based on nutritional diversity and commonality
    
    Strategy:
    - Prioritize ingredients with balanced nutrient profiles
    - Ensure diversity across nutrient dimensions
    
    Args:
        df: Cleaned ingredient DataFrame
        pool_size: Number of ingredients to select
        
    Returns:
        DataFrame with top N ingredients
    """
    print(f"Selecting top {pool_size} ingredients...")
    
    # Calculate nutrient diversity score
    nutrient_cols = ['calories', 'protein', 'sodium', 'carbs', 'fat']
    
    # Normalize nutrients
    df_norm = df.copy()
    for col in nutrient_cols:
        df_norm[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
    
    # Diversity score = variance across nutrients
    norm_cols = [f'{col}_norm' for col in nutrient_cols]
    df['diversity_score'] = df_norm[norm_cols].var(axis=1)
    
    # Select top by diversity
    selected = df.nlargest(pool_size, 'diversity_score')
    
    print(f"Selected {len(selected)} diverse ingredients")
    return selected.drop('diversity_score', axis=1).reset_index(drop=True)


def normalize_nutrients(df, reference='per_100g'):
    """
    Normalize nutrient values to standard serving size
    
    Args:
        df: Ingredient DataFrame
        reference: Normalization reference (default: per_100g)
        
    Returns:
        DataFrame with normalized nutrients
    """
    # USDA data is already per 100g, so we just ensure consistency
    print("Nutrients are standardized per 100g")
    return df


def save_processed_data(df, output_path=None):
    """
    Save processed ingredient data
    
    Args:
        df: Processed DataFrame
        output_path: Output file path
    """
    if output_path is None:
        output_path = Path(config.DATA_DIR).parent / "ingredients_processed.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    # Print summary statistics
    print("\n=== Ingredient Pool Summary ===")
    print(f"Total ingredients: {len(df)}")
    print("\nNutrient ranges (per 100g):")
    nutrient_cols = ['calories', 'protein', 'sodium', 'carbs', 'fat']
    print(df[nutrient_cols].describe())


def load_processed_data(input_path=None):
    """
    Load processed ingredient data
    
    Args:
        input_path: Input file path
        
    Returns:
        DataFrame with processed ingredients
    """
    if input_path is None:
        input_path = Path(__file__).parent.parent / config.PROCESSED_INGREDIENT_FILE
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} processed ingredients")
    return df


def preprocess_pipeline():
    """
    Complete preprocessing pipeline from raw USDA data to processed ingredients
    
    Returns:
        DataFrame with processed ingredients
    """
    # Load raw data
    food_df, nutrient_df = load_usda_data()
    
    # Extract nutrients
    ingredients = extract_nutrients(food_df, nutrient_df)
    
    # Clean data
    ingredients = clean_ingredients(ingredients)
    
    # Select ingredient pool
    ingredients = select_ingredient_pool(ingredients)
    
    # Normalize
    ingredients = normalize_nutrients(ingredients)
    
    # Save
    save_processed_data(ingredients)
    
    return ingredients


if __name__ == "__main__":
    # Run preprocessing pipeline
    ingredients = preprocess_pipeline()
    print("\n✓ Preprocessing complete!")
