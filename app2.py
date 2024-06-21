from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pickle
import streamlit as st
popular_df = pickle.load(open("artifacts/popular.pickle", "rb"))
app = Flask(__name__)
pt_cf = pickle.load(open("artifacts/pivot_CF.pickle", "rb"))
recipes_cf = pickle.load(open("artifacts/recipe_list_CF.pickle", "rb"))
model_cf = pickle.load(open("artifacts/model_CF.pickle", "rb"))

recipes_cb = pickle.load(open("artifacts/recipe_list_CB.pickle", "rb"))
matrix_cb = pickle.load(open("artifacts/matrix_CB.pickle", "rb"))
vectorizer_cb = pickle.load(open("artifacts/vectorizer_CB.pickle", "rb"))


@app.route("/")
def index():
    return render_template("index.html",
                           name = list(popular_df["Name"].values),
                           author = list(popular_df["AuthorName"].values),
                           image = list(popular_df["Images"].values),
                           votes = list(popular_df["Num_Ratings"].values),
                           rating = list(popular_df["Avg_Ratings"].values),
                           cooktime = list(popular_df["TotalTime"].values),
                           desc = list(popular_df["Description"].values),
                           category = list(popular_df["RecipeCategory"].values),

    )
@app.route("/recommend")
def recommend_ui():
    recipe_names = recipes_cf['Name'].tolist()
    return render_template("recommend.html", recipe_names=recipe_names)

@app.route("/recommend_recipes", methods=["POST"])
def recommend():
    user_input = request.form.get("user_input")
    if user_input not in pt_cf.index:
        return user_input

    id = np.where(pt_cf.index == user_input)[0][0]
    
    distances, indices = model_cf.kneighbors(pt_cf.iloc[id, :].values.reshape(1, -1), n_neighbors=6)
    data = []

    for i in indices[0]:  # indices is a 2D array, so we need to access the first element
        item = []
        temp_df = recipes_cf[recipes_cf["Name"] == pt_cf.index[i]]
        item.append(temp_df.drop_duplicates("Name")["Name"].values[0])
        item.append(temp_df.drop_duplicates("Name")["AuthorName"].values[0])
        item.append(temp_df.drop_duplicates("Name")["Images"].values[0])
        item.append(temp_df.drop_duplicates("Name")["TotalTime"].values[0])
        item.append(temp_df.drop_duplicates("Name")["Description"].values[0])
        item.append(temp_df.drop_duplicates("Name")["RecipeCategory"].values[0])
        data.append(item)
    print(data)
    recipe_names = recipes_cf['Name'].tolist()
    return render_template("recommend.html", data=data, recipe_names=recipe_names)
@app.route("/search")
def search_ui():
    return render_template("search.html")

@app.route("/search_recipes", methods=["POST"])
def search():
    user_input = request.form.get("user_input_search")
    num = int(request.form.get("user_input_num"))

    def preprocess_input(user_input):
        # Lowercase
        user_input = user_input.lower()
        # Remove non-alphabetic characters and keep spaces
        user_input = re.sub(r'[^a-zA-Z\s]', '', user_input)
        return user_input
    processed_input = preprocess_input(user_input)
        
        # Transform the user input to match the TF-IDF matrix
    user_input_tfidf = vectorizer_cb.transform([processed_input])
    print(preprocess_input)
        
        # Calculate cosine similarity between user input and all recipes
    cosine_similarities = cosine_similarity(user_input_tfidf, matrix_cb).flatten()
        
        # Get indices of the most similar recipes (sorted by similarity score)
    similar_indices = cosine_similarities.argsort()[::-1][:num]
        
        # Get the most similar recipes
    recommendations = recipes_cb.iloc[similar_indices]
    data = []

    for i in range(num):  # indices is a 2D array, so we need to access the first element
        item = []
        item.append(recommendations.iloc[i]["Name_y"])
        item.append(recommendations.iloc[i]["AuthorName"])
        item.append(recommendations.iloc[i]["Images"])
        item.append(recommendations.iloc[i]["TotalTime"])
        item.append(recommendations.iloc[i]["Description_y"])
        item.append(recommendations.iloc[i]["RecipeCategory_y"])
        item.append(recommendations.iloc[i]["RecipeId"])
        print(item[6])
        data.append(item)
    print(item)
    data.append(item)
    return render_template("search.html", data=data, input=user_input)
    


if __name__ == "__main__":
    app.run(debug=True)

