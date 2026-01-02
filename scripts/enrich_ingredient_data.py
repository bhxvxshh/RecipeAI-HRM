"""
Enrich ingredient database with categories, allergens, and dietary information.
This script adds metadata to help with filtering and practical recipe generation.
"""

import pandas as pd
import re

def categorize_ingredient(name):
    """Categorize ingredient based on its name."""
    name_lower = name.lower()
    
    # Meat & Poultry
    if any(word in name_lower for word in ['beef', 'pork', 'chicken', 'turkey', 'lamb', 'veal', 'duck', 'bacon', 'ham', 'sausage', 'meat']):
        return 'meat'
    
    # Fish & Seafood
    if any(word in name_lower for word in ['fish', 'salmon', 'tuna', 'shrimp', 'crab', 'lobster', 'seafood', 'anchovy', 'sardine']):
        return 'seafood'
    
    # Dairy & Eggs
    if any(word in name_lower for word in ['cheese', 'milk', 'yogurt', 'butter', 'cream', 'egg', 'whey', 'curd']):
        return 'dairy'
    
    # Vegetables
    if any(word in name_lower for word in ['lettuce', 'spinach', 'kale', 'broccoli', 'carrot', 'tomato', 'potato', 
                                            'onion', 'garlic', 'pepper', 'celery', 'cucumber', 'cabbage', 'squash',
                                            'zucchini', 'eggplant', 'mushroom', 'bean', 'pea', 'lentil', 'chickpea']):
        return 'vegetable'
    
    # Fruits
    if any(word in name_lower for word in ['apple', 'banana', 'orange', 'berry', 'grape', 'melon', 'peach', 'pear',
                                            'plum', 'cherry', 'mango', 'pineapple', 'lemon', 'lime', 'avocado',
                                            'strawberry', 'blueberry', 'raspberry', 'cranberry']):
        return 'fruit'
    
    # Grains & Flour
    if any(word in name_lower for word in ['flour', 'wheat', 'rice', 'oat', 'barley', 'rye', 'corn', 'quinoa', 
                                            'pasta', 'bread', 'cereal', 'grain', 'fonio', 'millet', 'sorghum']):
        return 'grain'
    
    # Nuts & Seeds
    if any(word in name_lower for word in ['nut', 'almond', 'walnut', 'pecan', 'cashew', 'peanut', 'pistachio',
                                            'hazelnut', 'macadamia', 'seed', 'sesame', 'sunflower', 'pumpkin']):
        return 'nuts_seeds'
    
    # Oils & Fats
    if any(word in name_lower for word in ['oil', 'fat', 'lard', 'shortening', 'margarine']):
        return 'oil_fat'
    
    # Sweeteners
    if any(word in name_lower for word in ['sugar', 'honey', 'syrup', 'molasses', 'agave', 'stevia']):
        return 'sweetener'
    
    # Herbs & Spices
    if any(word in name_lower for word in ['spice', 'herb', 'basil', 'oregano', 'thyme', 'rosemary', 'cilantro',
                                            'parsley', 'mint', 'sage', 'pepper', 'salt', 'cinnamon', 'cumin']):
        return 'herb_spice'
    
    # Legumes
    if any(word in name_lower for word in ['bean', 'lentil', 'chickpea', 'pea', 'soy', 'tofu', 'tempeh']):
        return 'legume'
    
    # Other/Condiments
    if any(word in name_lower for word in ['sauce', 'vinegar', 'pickle', 'olive', 'ketchup', 'mustard', 'mayo']):
        return 'condiment'
    
    return 'other'


def detect_allergens(name, category):
    """Detect common allergens in ingredient."""
    allergens = []
    name_lower = name.lower()
    
    # Dairy
    if category == 'dairy' or any(word in name_lower for word in ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'whey']):
        allergens.append('dairy')
    
    # Eggs
    if 'egg' in name_lower:
        allergens.append('eggs')
    
    # Gluten (wheat, rye, barley)
    if any(word in name_lower for word in ['wheat', 'rye', 'barley', 'flour']) and 'gluten' not in name_lower:
        # Exceptions for gluten-free flours
        if not any(word in name_lower for word in ['rice', 'corn', 'potato', 'cassava', 'almond', 'coconut', 
                                                     'chestnut', 'glutinous rice']):
            allergens.append('gluten')
    
    # Tree nuts
    if any(word in name_lower for word in ['almond', 'walnut', 'pecan', 'cashew', 'pistachio', 'hazelnut', 
                                            'macadamia', 'pine nut', 'brazil nut']):
        allergens.append('tree_nuts')
    
    # Peanuts (separate from tree nuts)
    if 'peanut' in name_lower:
        allergens.append('peanuts')
    
    # Fish & Shellfish
    if category == 'seafood':
        if any(word in name_lower for word in ['shrimp', 'crab', 'lobster', 'clam', 'oyster', 'mussel', 'scallop']):
            allergens.append('shellfish')
        else:
            allergens.append('fish')
    
    # Soy
    if any(word in name_lower for word in ['soy', 'tofu', 'tempeh', 'miso', 'edamame']):
        allergens.append('soy')
    
    # Sesame
    if 'sesame' in name_lower:
        allergens.append('sesame')
    
    return ','.join(allergens) if allergens else 'none'


def determine_dietary_tags(category, allergens, name):
    """Determine dietary compatibility tags."""
    tags = []
    name_lower = name.lower()
    allergen_list = allergens.split(',')
    
    # Vegan (no animal products)
    if category not in ['meat', 'seafood', 'dairy'] and 'eggs' not in allergen_list and 'honey' not in name_lower:
        tags.append('vegan')
    
    # Vegetarian (no meat/seafood, but dairy/eggs OK)
    if category not in ['meat', 'seafood']:
        tags.append('vegetarian')
    
    # Gluten-free
    if 'gluten' not in allergen_list:
        tags.append('gluten_free')
    
    # Dairy-free
    if 'dairy' not in allergen_list:
        tags.append('dairy_free')
    
    # Nut-free
    if 'tree_nuts' not in allergen_list and 'peanuts' not in allergen_list:
        tags.append('nut_free')
    
    # Paleo (no grains, legumes, dairy, processed foods)
    if category not in ['grain', 'legume', 'dairy'] and 'sugar' not in name_lower:
        tags.append('paleo')
    
    # Keto-friendly (low carb - <10g per 100g for simplicity)
    # Will be determined from actual carb content
    
    return ','.join(tags) if tags else 'none'


def estimate_cost_tier(name, category):
    """Estimate relative cost tier (low/medium/high)."""
    name_lower = name.lower()
    
    # Low cost (staples, common items)
    if any(word in name_lower for word in ['rice', 'flour', 'oil', 'sugar', 'potato', 'onion', 'carrot', 
                                            'egg', 'milk', 'pasta', 'bread', 'bean', 'lentil']):
        return 'low'
    
    # High cost (premium, specialty items)
    if any(word in name_lower for word in ['truffle', 'caviar', 'saffron', 'lobster', 'crab', 'salmon', 
                                            'pine nut', 'macadamia', 'pecan', 'organic', 'grass-fed']):
        return 'high'
    
    # Medium cost (everything else)
    return 'medium'


def enrich_ingredients():
    """Main function to enrich ingredient data."""
    print("Loading ingredients data...")
    df = pd.read_csv('data/ingredients_processed.csv')
    
    print(f"Processing {len(df)} ingredients...")
    
    # Add new columns
    df['category'] = df['food_name'].apply(categorize_ingredient)
    df['allergens'] = df.apply(lambda row: detect_allergens(row['food_name'], row['category']), axis=1)
    df['dietary_tags'] = df.apply(lambda row: determine_dietary_tags(row['category'], row['allergens'], row['food_name']), axis=1)
    df['cost_tier'] = df.apply(lambda row: estimate_cost_tier(row['food_name'], row['category']), axis=1)
    
    # Add keto-friendly tag based on carbs
    def add_keto_tag(row):
        tags = row['dietary_tags'].split(',')
        if row['carbs'] < 10:  # Low carb threshold
            if 'none' in tags:
                return 'keto'
            else:
                tags.append('keto')
                return ','.join(tags)
        return row['dietary_tags']
    
    df['dietary_tags'] = df.apply(add_keto_tag, axis=1)
    
    # Save enriched data
    output_file = 'data/ingredients_enriched.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved enriched data to {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("ENRICHMENT STATISTICS")
    print("="*60)
    print("\nCategory Distribution:")
    print(df['category'].value_counts())
    
    print("\nAllergen Distribution:")
    allergen_counts = {}
    for allergens in df['allergens']:
        for allergen in allergens.split(','):
            allergen = allergen.strip()
            if allergen and allergen != 'none':
                allergen_counts[allergen] = allergen_counts.get(allergen, 0) + 1
    for allergen, count in sorted(allergen_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {allergen}: {count}")
    
    print("\nDietary Tag Distribution:")
    tag_counts = {}
    for tags in df['dietary_tags']:
        for tag in tags.split(','):
            tag = tag.strip()
            if tag and tag != 'none':
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tag}: {count}")
    
    print("\nCost Tier Distribution:")
    print(df['cost_tier'].value_counts())
    
    print("\n" + "="*60)
    print("Sample enriched ingredients:")
    print("="*60)
    print(df[['food_name', 'category', 'allergens', 'dietary_tags', 'cost_tier']].head(10).to_string(index=False))
    
    return df


if __name__ == '__main__':
    enrich_ingredients()
