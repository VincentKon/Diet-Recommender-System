from flask import Flask, render_template, request
import numpy as np
import pickle
popular_df = pickle.load(open("model/popular.pickle", "rb"))
app = Flask(__name__)
pt = pickle.load(open("model/pt.pickle", "rb"))
recipes = pickle.load(open("model/recipe.pkl", "rb"))
model = pickle.load(open("model/model2.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html",
                           name = list(popular_df["Name"].values),
                           author = list(popular_df["AuthorName"].values),
                           image = list(popular_df["Images"].values),
                           votes = list(popular_df["Num_Ratings"].values),
                           rating = list(popular_df["Avg_Ratings"].values),

    )
@app.route("/recommend")
def recommend_ui():
    return render_template("recommend.html")

@app.route("/recommend_books", methods=["POST"])
def recommend():
    user_input = request.form.get("user_input")
    id = np.where(pt.index == user_input)[0][0]
    distances, indices = model.kneighbors(pt.iloc[id, :].values.reshape(1, -1), n_neighbors=6)
    data = []
    for i in indices[0]:  # indices is a 2D array, so we need to access the first element
        item = []
        temp_df = recipes[recipes["Name"] == pt.index[i]]
        item.append(temp_df.drop_duplicates("Name")["Name"].values[0])
        item.append(temp_df.drop_duplicates("Name")["AuthorName"].values[0])
        item.append(temp_df.drop_duplicates("Name")["Images"].values[0])
        data.append(item)
    print(data)
    return render_template("recommend.html", data=data)
    


if __name__ == "__main__":
    app.run(debug=True)

