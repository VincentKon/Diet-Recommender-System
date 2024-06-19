import pickle
import streamlit as st
import numpy as np
from flask import Flask

app = Flask(__name__)
@app.route("/")
def index():
    return "Hello World"

# Set up the Streamlit app
st.set_page_config(page_title="Diet Recommender System", layout="wide")
st.header('Diet Recommender System Using Machine Learning')

# Load models and data
model = pickle.load(open('model/model1.pickle', 'rb'))
recipes_names = pickle.load(open('model/recipe_names.pickle', 'rb'))
final_df = pickle.load(open('model/final_df.pickle', 'rb'))
pivot = pickle.load(open('model/pivot.pickle', 'rb'))

# Function to fetch poster URLs
def fetch_poster(suggestion):
    poster_url = []
    for recipe_id in suggestion[0]: 
        idx = np.where(final_df['Name'] == pivot.index[recipe_id])[0][0]
        url = final_df.iloc[idx]['Images']
        urls = url.strip('c()').replace('"', '').split(', ')
        poster_url.append(urls[0])
    return poster_url

# Function to recommend recipes
def recommend_recipes(name):
    recipes_list = []
    id = np.where(pivot.index == name)[0][0]
    distance, suggestion = model.kneighbors(pivot.iloc[id, :].values.reshape(1, -1), n_neighbors=6)
    
    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion[0])):
        recipes_list.append(pivot.index[suggestion[0][i]])
    
    return recipes_list, poster_url

# UI elements
selected_recipes = st.selectbox(
    "Type or select a recipe from the dropdown",
    recipes_names
)

if st.button('Show Recommendation'):
    recommended_recipes, poster_url = recommend_recipes(selected_recipes)
    
    # Display recommendations in columns
    cols = st.columns(5)
    
    for i in range(1, 6):
        with cols[i-1]:
            st.text(recommended_recipes[i])
            st.image(poster_url[i])

