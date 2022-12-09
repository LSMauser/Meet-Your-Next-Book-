from flask import *
import pandas as pd
import json
import numpy as np
import nltk
import re

dataset = pd.read_csv("booksummaries.txt", header=None,sep="\t", names=["Wikipedia ID", "Freebase ID", "Book Title", "Book Author", "Pub date","Genres","Summary"])
dataset_v1 = dataset.dropna(subset=['Genres', 'Summary'])
genres_v1 = []
for item in dataset_v1['Genres']:
    genres_v1.append(list(json.loads(item).values()))
dataset_v1['Cleaned Genres']=genres_v1
dataset_v1 = dataset_v1[['Book Title','Book Author','Pub date','Summary','Cleaned Genres']]
#define stopwords
stopw = nltk.corpus.stopwords.words('english') 

#create normalizing function
def normalize_document(text):
    #remove special characters & whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)
    #lowercase all letters
    text = text.lower()
    text = text.strip()
    #create text tokens
    text_tokens = nltk.word_tokenize(text)
    #Remove stopwords
    remove_sw_text = [token for token in text_tokens if token not in stopw]
    # re-create document from filtered tokens
    text = ' '.join(remove_sw_text)
    return text

#vectorize function
normalized_doc = np.vectorize(normalize_document)
#normalize summary column values of dataframe 
norm_doc = normalized_doc(list(dataset_v1['Summary'])) 
#create new column "cleaned summary"
dataset_v1['Cleaned Summary']=norm_doc
#normalize Book Title column values of dataframe 
norm_doc_titles = normalized_doc(list(dataset_v1['Book Title'])) 
dataset_v1['Cleaned Titles']=norm_doc_titles

#Next - extract TF-IDF Features
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vect = TfidfVectorizer(ngram_range=(1, 2), min_df=2) #initialize vectors
tf_matrix = tf_vect.fit_transform(norm_doc) #create matrix of features
tf_matrix.shape #print shape
#Compute pairwise document similarity
from sklearn.metrics.pairwise import cosine_similarity
book_similarity_scores = cosine_similarity(tf_matrix)
book_sim_df = pd.DataFrame(book_similarity_scores) #turn in dataframe
#create list for title and cleaned title values
cleaned_book_title_list=dataset_v1['Cleaned Titles'].values #numpy array
book_title_list=dataset_v1['Book Title'].values

#create function to recommend 5 most similar books based on summary
def book_recommender(title, books=book_title_list, cleaned_books=cleaned_book_title_list, document_similarity=book_sim_df):
    #Find where the book title is located in book title list
    title_index = np.where(cleaned_books == title)[0][0]
    #Locate row in similarity matrix 
    book_cossim = document_similarity.iloc[title_index].values
    #Sort the title's similarity scores from highest to smallest
    #Save the top 5 highest scores and the corresponding index
    book_indx5 = np.argsort(-book_cossim)[1:6]
    # Find the corresponding book titles
    top5_sim_books = books[book_indx5]
    # return the top 5 movies
    return top5_sim_books


from Levenshtein import distance as lev

#use levenshtein distance to calculate how different two titles are 
def title_similarity2(title, title_list):
    #initialize list
    lev_dist=[]
    for ti in title_list:
        #calculate levenshtein diff b/w inputted title and title list
        lev_dist.append(lev(title,ti))
    #find index for minimum value (most similar title)
    min_index=lev_dist.index(min(lev_dist))
    #return most similar title
    similar_title=title_list[min_index]
    return similar_title

top5_titles=[]

app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def homepage():
    if request.method=='POST':
        result=request.form.get("Book Title")
        cleaned_input_title=normalize_document(result)
        if cleaned_input_title in cleaned_book_title_list:
            global top5_titles
            top5_titles=book_recommender(cleaned_input_title)
            #if request.form.get("Accept Book") == 'no':
            #    print('Testing5678')
            #    return render_template("index.html",test_title=top5_titles[0],test_title2=top5_titles[1])
            return render_template("index.html", test_title=top5_titles[0])
        else:
            global suggested_title
            suggested_title=title_similarity2(title=cleaned_input_title,title_list=cleaned_book_title_list)
            return render_template("index.html",suggestedtitle=suggested_title)
        
    return render_template("index.html")

@app.route('/test2', methods=['POST','GET'])
def suggestedtitle():
    if request.method=='POST':
        if request.form.get("Book Title2") =="yes":
            global top5_titles2
            top5_titles2=book_recommender(suggested_title)
            return render_template("index.html",suggestedtitle=suggested_title, test_titleS=top5_titles2[0])
        elif request.form.get("Book Title2") =="no":
            message="I am sorry, we do not recognize that book title. Try again with a different book!"
            return render_template("index.html",suggestedtitle=suggested_title, message2=message)

    return render_template("index.html")

@app.route('/test', methods=['POST','GET'])
def secondchoice():
    if request.method=='POST':
        if request.form.get("Accept Book") =="no":
            print("testing123")
            return render_template("index.html",test_title=top5_titles[0],test_title2=top5_titles[1])
        elif request.form.get("Accept Book") =="yes":
            final_message="Great! Happy Reading!"
            return render_template("index.html",test_title=top5_titles[0],fmessage=final_message)
    return render_template("index.html")

@app.route('/test3', methods=['POST','GET'])
def thirdchoice():
    if request.method=='POST':
        if request.form.get("Book Title3") =="no":
            return render_template("index.html",test_title=top5_titles[0],test_title2=top5_titles[1],test_title3=top5_titles[2])
        elif request.form.get("Book Title3") =="yes":
            final_message="Great! Happy Reading!"
            return render_template("index.html",test_title=top5_titles[0],test_title2=top5_titles[1],fmessage=final_message)
        elif request.form.get("SBook Title")=='no':
            return render_template("index.html",suggestedtitle=suggested_title, test_titleS=top5_titles2[0],test_titleS2=top5_titles2[1])
        elif request.form.get("SBook Title")=='yes':
            final_message="Great! Happy Reading!"
            return render_template("index.html",suggestedtitle=suggested_title, test_titleS=top5_titles2[0],fmessage=final_message)
    return render_template("index.html")

@app.route('/test4', methods=['POST','GET'])
def fourthchoice():
    if request.method=='POST':
        if request.form.get("Book Title4") =="no":
            return render_template("index.html",test_title=top5_titles[0],test_title2=top5_titles[1],test_title3=top5_titles[2], test_title4=top5_titles[3])
        elif request.form.get("Book Title4") =="yes":
            final_message="Great! Happy Reading!"
            return render_template("index.html",test_title=top5_titles[0],test_title2=top5_titles[1],test_title3=top5_titles[2], fmessage=final_message)
        elif request.form.get("SBook Title2")=='no':
            return render_template("index.html",suggestedtitle=suggested_title, test_titleS=top5_titles2[0],test_titleS2=top5_titles2[1],test_titleS3=top5_titles2[2])
        elif request.form.get("SBook Title2")=='yes':
            final_message="Great! Happy Reading!"
            return render_template("index.html",suggestedtitle=suggested_title, test_titleS=top5_titles2[0],test_titleS2=top5_titles2[1], fmessage=final_message)    
    return render_template("index.html")

@app.route('/test5', methods=['POST','GET'])
def fifthchoice():
    if request.method=='POST':
        if request.form.get("Book Title5") =="no":
            restart_message="I am sorry, you do not like any of our suggestions! Try again with a different book!"
            return render_template("index.html",test_title=top5_titles[0],test_title2=top5_titles[1],test_title3=top5_titles[2], test_title4=top5_titles[3],r_message=restart_message)
        elif request.form.get("Book Title5") =="yes":
            final_message="Great! Happy Reading!"
            return render_template("index.html",test_title=top5_titles[0],test_title2=top5_titles[1],test_title3=top5_titles[2],test_title4=top5_titles[3], fmessage=final_message)
        elif request.form.get("SBook Title3")=='no':
                return render_template("index.html",suggestedtitle=suggested_title, test_titleS=top5_titles2[0],test_titleS2=top5_titles2[1],test_titleS3=top5_titles2[2], test_titleS4=top5_titles2[3])
        elif request.form.get("SBook Title3")=='yes':
            final_message="Great! Happy Reading!"
            return render_template("index.html",suggestedtitle=suggested_title, test_titleS=top5_titles2[0],test_titleS2=top5_titles2[1], test_titleS3=top5_titles2[2], fmessage=final_message)    
    return render_template("index.html")

@app.route('/test6', methods=['POST','GET'])
def sixthchoice():
    if request.method=='POST':
        if request.form.get("SBook Title4")=='no':
            restart_message="I am sorry, you do not like any of our suggestions! Try again with a different book!"
            return render_template("index.html",suggestedtitle=suggested_title, test_titleS=top5_titles2[0],test_titleS2=top5_titles2[1],test_titleS3=top5_titles2[2], test_titleS4=top5_titles2[3],r_message=restart_message)
        elif request.form.get("SBook Title4")=='yes':
            final_message="Great! Happy Reading!"
            return render_template("index.html",suggestedtitle=suggested_title, test_titleS=top5_titles2[0],test_titleS2=top5_titles2[1], test_titleS3=top5_titles2[2],test_titleS4=top5_titles2[3], fmessage=final_message)    
    return render_template("index.html")
            


#@app.route('/test', methods=['POST','GET'])
#def secondchoice():
#    if request.method=='POST':
#        print(top5_titles)
#        if request.form.get("Accept Book") =="no":
#            print("testing123")
#            return render_template("index.html",test_title=top5_titles[0],test_title2=top5_titles[1])
#    return render_template("index.html")

if __name__ =="__main__":
    app.run()