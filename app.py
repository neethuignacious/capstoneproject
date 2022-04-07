# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 14:31:09 2022

@author: Neethu
"""

from flask import Flask,render_template,request,json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
import re 
app = Flask(__name__,template_folder='template')
model3 = pickle.load(open('recipebuilder.pickle', 'rb'))
similarity = pickle.load(open('Model1.pickle', 'rb'))
model1_df = pickle.load(open('Model1df.pickle', 'rb'))
column = json.load(open('columns.json', 'rb'))


def get_recommendations(N, scores):
    
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    title = []
    images = []
    for i in top:
        title.append(model1_df['recipe_name'][i])
        images.append(model1_df['image_url'][i])
    return title,images

def RecSys(ingredients, N=11):
    
    tfidf_encodings = pickle.load(open('TFIDF_ENCODING_PATH_newest', 'rb'))
    tfidf = pickle.load(open('TFIDF_MODEL_PATH_newest', 'rb'))
    
    ingredients_parsed = ingredients
    
    ingredients_tfidf = tfidf.transform([ingredients_parsed])
    
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)

    title,images = get_recommendations(N, scores)
    return title,images

def reco(dessert):
    dessert = str(dessert)
    names = []
    images = []
    index = model1_df[model1_df['recipe_name'] == dessert].index[0]
    image = model1_df['image_url'][index]
    category = model1_df['category'][index]
    detail = model1_df['details'][index]
    ingred = model1_df['ingredients'][index]
    instruc = model1_df['instructions'][index]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        names.append(model1_df.iloc[i[0]].recipe_name)
        images.append(model1_df.iloc[i[0]].image_url)
    return names,category,detail,ingred,instruc,image,images

@app.route('/')

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("aboutnew.html")


@app.route('/recommendor', methods=["GET"])
def recommendor():
    file_data = json.load(open('columns.json', 'rb'))
    dessert_name_array = file_data['data_columns']
    return render_template("dessertrec_new.html",dessert_name_array=dessert_name_array)

@app.route("/predict",methods =['POST'])
def predict():
    dessert = request.form['dessertname']
    names,category,detail,ingred,instruc,image,images = reco(dessert)
    return render_template('dessertdisplay.html', data = names, dn=dessert,category=category,detail=detail,ingred=ingred,instruc=instruc,image=image,images=images)

@app.route('/baking', methods=["GET"])
def baking():
    return render_template("baking.html")

@app.route('/printing/<detail>')
def printingdetails(detail):
    dessert = str(re.sub(r"%20",' ',detail))
    names,category,detail,ingred,instruc,image,images = reco(dessert)
    return render_template('dessertdisplay.html', data = names, dn=dessert,category=category,detail=detail,ingred=ingred,instruc=instruc,image=image,images=images)

@app.route('/recipe', methods=["POST"])
def recommend_recipe():
    global recommendation
    ingredients = request.form['ingredients']
    title,images = RecSys(ingredients)
    return(render_template("output.html", data = title, images=images))

@app.route('/<detail>')
def dessertdetails(detail):
    dessert = str(re.sub(r"%20",' ',detail))
    index = model1_df[model1_df['recipe_name'] == dessert].index[0]
    images = model1_df['image_url'][index]
    category = model1_df['category'][index]
    detail = model1_df['details'][index]
    ingred = model1_df['ingredients'][index]
    instruc = model1_df['instructions'][index]
    return(render_template("output1.html", dn = dessert,category=category,detail=detail,ingred=ingred,instruc=instruc,images=images))

@app.route("/builder")
def builder():
    return render_template("builder.html")

@app.route('/result',methods = ["POST"])
def result():
    x = np.zeros(16)
    x[0] = request.form['flour']
    x[1] = request.form['sugar']
    x[2] = request.form['butter']
    x[3] = request.form['milk']
    x[4] = request.form['egg']
    x[5] = request.form['oil']
    x[6] = request.form['water']
    x[7] = request.form['powder']
    x[8] = request.form['fruits']
    x[9] = request.form['chocolate']
    x[10] = request.form['cream']
    x[11] = request.form['juice']
    x[12] = request.form['yeast']
    x[13] = request.form['extract']
    x[14] = request.form['bakingpowder']
    x[15] = request.form['salt']
    
    result = model3.predict([x])[0]
    
    ing = ['Flour','Sugar','Butter','Milk','Eggs','Oil','Water','Powder','Fruits','Chocolate','Cream','Juice','Yeast','Extract','Baking powder','Salt']
    count = 0
    new_ing = []
    for i in x:
        if i != 0:
            
            if ing[count] == 'Flour':
                f = int(i)
                new_ing.append(str(f) + str(" cup") + str(" all-purpose flour,spoon and levelled"))
                count = count+1
            elif ing[count] == 'Sugar':
                s = int(i)
                new_ing.append(str(s) + str(" cup") + str(" granulated sugar"))
                count = count+1
            elif ing[count] == 'Butter':
                b = int(i)
                new_ing.append(str(b) + str(" cup") + str(" unsalted butter,softened to room temperature"))
                count = count+1
            elif ing[count] == 'Milk':
                m = int(i)
                new_ing.append(str(m) + str(" cup") + str(" whole milk, room temperature"))
                count = count+1
            elif ing[count] == 'Eggs':
                egg = int(i/0.25)
                new_ing.append(str(egg) + str(" large eggs, at room temperature"))
                count = count+1
            elif ing[count] == 'Oil':
                o = int(i)
                new_ing.append(str(o) + str(" cup") + str(" extra-virgin oil"))
                count = count+1
            elif ing[count] == 'Water':
                w = int(i)
                new_ing.append(str(w) + str(" cup") + str(" hot water"))
                count = count+1
            elif ing[count] == 'Powder':
                p = int(i)
                new_ing.append(str(p) + str(" cup") + str(" unsweetened dutch-process powder, sifted"))
                count = count+1
            elif ing[count] == 'Fruits':
                f = int(i)
                new_ing.append(str(f) + str(" cup") + str(" chopped fresh or frozen fruits"))
                count = count+1
            elif ing[count] == 'Chocolate':
                c = int(i)
                new_ing.append(str(c) + str(" cup") + str(" roughly chopped chocolates or mini chocolate chips"))
                count = count+1
            elif ing[count] == 'Cream':
                cr = int(i)
                new_ing.append(str(cr) + str(" cup") + str(" full-fat sour cream, at room temperature"))
                count = count+1
            elif ing[count] == 'Juice':
                j = int(i)
                new_ing.append(str(j) + str(" cup") + str(" chopped fresh or frozen fruits"))
                count = count+1
            elif ing[count] == 'Yeast':
                new_ing.append(str(i) + str(" cup") + str(" instant or active dry yeast"))
                count = count+1
            elif ing[count] == 'Extract':
                new_ing.append(str(i) + str(" cup") + str(" pure vanilla extract or vanilla paste or other extracts"))
                count = count+1
            elif ing[count] == 'Baking powder':
                new_ing.append(str(i) + str(" cup") + str(" baking powder"))
                count = count+1
            elif ing[count] == 'Salt':
                new_ing.append(str(i) + str(" cup") + str(" salt or fine sea salt"))
                count = count+1
        else:
            count = count+1 
    if result == 0:
        return(render_template("result.html", prediction="Cake",data = new_ing))
    elif result == 1:
        return(render_template("result1.html", prediction="Cookie",data = new_ing))
    else:
        return(render_template("result2.html", prediction="Bread",data = new_ing))
    

if __name__ == "__main__":
    app.run()