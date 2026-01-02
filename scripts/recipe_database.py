"""
Recipe Search & Retrieval System
Finds recipes by name (dosa, sandwich, pasta) from multiple datasets.
"""

import pandas as pd
import json
import re
from fuzzywuzzy import fuzz
from pathlib import Path


class RecipeDatabase:
    def __init__(self):
        """Load all recipe datasets."""
        print("Loading recipe databases...")
        
        # Epicurious (20k recipes with nutrition)
        self.epicurious = pd.read_csv('data/recipes/epi_r.csv')
        print(f"✓ Epicurious: {len(self.epicurious)} recipes")
        
        # Food.com (231k recipes with steps)
        self.foodcom = pd.read_csv('data/recipes/food_com/RAW_recipes.csv')
        print(f"✓ Food.com: {len(self.foodcom)} recipes")
        
        # RecipeNLG (2.2M recipes - load sample for now)
        print("  Loading RecipeNLG sample (this may take a moment)...")
        self.recipenlg = pd.read_csv(
            'data/recipes/recipenlg/RecipeNLG_dataset.csv',
            nrows=50000  # Load first 50k for speed
        )
        print(f"✓ RecipeNLG: {len(self.recipenlg)} recipes loaded")
        
        print(f"\n📚 Total: {len(self.epicurious) + len(self.foodcom) + len(self.recipenlg)} recipes available\n")
    
    def search_recipes(self, query, max_results=10, limit=None, dataset='all'):
        """
        Search for recipes by name.
        
        Args:
            query: Recipe name (e.g., "dosa", "sandwich", "pasta")
            max_results: Number of results to return
            limit: Alias for max_results (for backwards compatibility)
            dataset: 'epicurious', 'foodcom', 'recipenlg', or 'all'
        
        Returns:
            List of matching recipes with scores
        """
        # Support both max_results and limit parameter names
        if limit is not None:
            max_results = limit
        query_lower = query.lower()
        results = []
        
        # Search Epicurious
        if dataset in ['epicurious', 'all']:
            for idx, row in self.epicurious.iterrows():
                title = str(row['title']).lower()
                score = fuzz.partial_ratio(query_lower, title)
                
                if score > 60:  # Threshold
                    results.append({
                        'source': 'epicurious',
                        'name': row['title'],
                        'score': score,
                        'calories': row.get('calories', 0),
                        'protein': row.get('protein', 0),
                        'fat': row.get('fat', 0),
                        'sodium': row.get('sodium', 0),
                        'rating': row.get('rating', 0),
                        'data': row
                    })
        
        # Search Food.com
        if dataset in ['foodcom', 'all']:
            for idx, row in self.foodcom.iterrows():
                if idx > 10000:  # Limit search for speed
                    break
                
                name = str(row['name']).lower()
                score = fuzz.partial_ratio(query_lower, name)
                
                if score > 60:
                    # Parse nutrition (stored as string)
                    nutrition = eval(row['nutrition']) if pd.notna(row['nutrition']) else [0]*7
                    
                    results.append({
                        'source': 'foodcom',
                        'name': row['name'],
                        'score': score,
                        'calories': nutrition[0] if len(nutrition) > 0 else 0,
                        'protein': nutrition[4] if len(nutrition) > 4 else 0,
                        'fat': nutrition[1] if len(nutrition) > 1 else 0,
                        'sodium': nutrition[5] if len(nutrition) > 5 else 0,
                        'minutes': row.get('minutes', 0),
                        'n_steps': row.get('n_steps', 0),
                        'n_ingredients': row.get('n_ingredients', 0),
                        'ingredients': eval(row['ingredients']) if pd.notna(row['ingredients']) else [],
                        'steps': eval(row['steps']) if pd.notna(row['steps']) else [],
                        'data': row
                    })
        
        # Search RecipeNLG
        if dataset in ['recipenlg', 'all']:
            for idx, row in self.recipenlg.iterrows():
                if idx > 5000:  # Limit for speed
                    break
                
                title = str(row.get('title', '')).lower()
                score = fuzz.partial_ratio(query_lower, title)
                
                if score > 60:
                    results.append({
                        'source': 'recipenlg',
                        'name': row.get('title', 'Unknown'),
                        'score': score,
                        'ingredients': row.get('ner', '').split(',') if pd.notna(row.get('ner')) else [],
                        'directions': row.get('directions', ''),
                        'data': row
                    })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:max_results]
    
    def get_recipe_details(self, recipe):
        """Get full recipe details with ingredients and instructions."""
        source = recipe['source']
        data = recipe['data']
        
        if source == 'epicurious':
            # Need to load full_format_recipes.json for ingredients
            return {
                'name': recipe['name'],
                'source': 'Epicurious',
                'nutrition': {
                    'calories': recipe['calories'],
                    'protein': recipe['protein'],
                    'fat': recipe['fat'],
                    'sodium': recipe['sodium']
                },
                'rating': recipe['rating'],
                'ingredients': [],  # Would need to parse from JSON
                'instructions': []
            }
        
        elif source == 'foodcom':
            return {
                'name': recipe['name'],
                'source': 'Food.com',
                'cook_time': recipe['minutes'],
                'servings': 1,  # Default
                'difficulty': 'Medium' if recipe['n_steps'] > 8 else 'Easy',
                'nutrition': {
                    'calories': recipe['calories'],
                    'protein': recipe['protein'],
                    'fat': recipe['fat'],
                    'sodium': recipe['sodium']
                },
                'ingredients': recipe['ingredients'],
                'instructions': recipe['steps'],
                'tags': eval(data['tags']) if pd.notna(data['tags']) else []
            }
        
        elif source == 'recipenlg':
            return {
                'name': recipe['name'],
                'source': 'RecipeNLG',
                'ingredients': recipe['ingredients'],
                'instructions': recipe['directions'].split('\n') if recipe['directions'] else []
            }
        
        return recipe


# Quick test
if __name__ == '__main__':
    db = RecipeDatabase()
    
    # Test searches
    test_queries = ['dosa', 'sandwich', 'pasta', 'pizza', 'salad']
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Searching for: {query}")
        print('='*70)
        
        results = db.search_recipes(query, max_results=5)
        
        if results:
            print(f"Found {len(results)} matches:\n")
            for i, recipe in enumerate(results, 1):
                print(f"{i}. {recipe['name']} ({recipe['source']})")
                print(f"   Score: {recipe['score']}")
                if 'calories' in recipe:
                    print(f"   Nutrition: {recipe['calories']} cal, {recipe['protein']}g protein")
                print()
        else:
            print("No matches found.\n")
