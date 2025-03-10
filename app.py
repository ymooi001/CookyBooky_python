from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import pandas as pd
import numpy as np
from difflib import get_close_matches  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math  # Import math to check for NaN values
import re

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset
recipe_df = pd.read_csv("recipes.csv")

# Function to clean ingredients
def clean_ingredients(ingredients):
    if pd.isna(ingredients):
        return ""
    ingredients = ingredients.lower()
    ingredients = re.sub(r'\b(and|with|fresh|chopped|sliced|diced|ground|to taste|optional|tablespoon|teaspoon|cup|oz|ml|g|kg)\b', '', ingredients)
    return " ".join(set(ingredients.split(", ")))

# Apply cleaning function
recipe_df["cleaned_ingredients"] = recipe_df["ingredients"].apply(clean_ingredients)

# Recompute TF-IDF on Cleaned Ingredients
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(recipe_df["cleaned_ingredients"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def safe_value(val):
    """Convert Pandas NaN/NaT or float('nan') to None, otherwise return original val."""
    if pd.isna(val):
        return None
    return val

@app.route("/recommend", methods=["GET"])
def recommend():
    query = request.args.get("recipe_name", "").strip().lower()
    recipe_names = recipe_df["name"].str.lower().tolist()

    if query in recipe_names:
        matched_name = recipe_df.loc[recipe_df["name"].str.lower() == query, "name"].values[0]
    else:
        close_matches = get_close_matches(query, recipe_names, n=1, cutoff=0.4)
        if not close_matches:
            return jsonify({"error": "No similar recipe found!"}), 404
        matched_name = recipe_df.loc[recipe_df["name"].str.lower() == close_matches[0], "name"].values[0]

    idx = recipe_df.index[recipe_df["name"] == matched_name].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:5]

    recommendations = []
    for i, score in sim_scores:
        if i < len(recipe_df):
            recipe_data = recipe_df.iloc[i]
            recommendations.append({
               "name": safe_value(recipe_data["name"]),
	       "id": safe_value(recipe_data["recipe_id"]),
               "category": safe_value(recipe_data["category"]),
               "ingredients": safe_value(recipe_data["ingredients"]),
               "instructions": safe_value(recipe_data["instructions"]),
               "servings": (None if pd.isna(recipe_data["servings"]) 
                     else str(recipe_data["servings"])),
               "image_url": safe_value(recipe_data["image_url"]),
               "similarity_score": round(float(score), 4)
            })

    response_data = {"recipe_name": matched_name, "recommendations": recommendations}
    return jsonify(response_data)



# Run the Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
