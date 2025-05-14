
import os
import openai
import string
import re
import langdetect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import pandas as pd
import spacy


chatStr = ""
apikey = ''  # Replace with your OpenAI API key


def chat(predefined_prompt):
    global chatStr
    chatStr += f"You: {predefined_prompt}\n Natter: "

    openai.api_key = apikey
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=chatStr,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=1
    )

    response_text = response.choices[0].text.strip()

    chatStr += f"{response_text}\n"

    return response_text


def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.replace('\n', ' ')
    text = text.strip()
    text = re.sub(' +', ' ', text)
    return text


def is_english(text):
    try:
        return langdetect.detect(text) == 'en'
    except:
        return False


nlp = spacy.load("en_core_web_sm")



def text_tokenizer(doc):
    return ' '.join([token.lemma_ for token in nlp(doc) if not token.is_stop])


def preprocess_data(file_path):
    recipes = pd.read_excel(file_path)
    recipes = recipes.dropna(subset=['RecipeName', 'Ingredients', 'Instructions'])

    recipes['is_english'] = recipes['RecipeName'].apply(is_english)

    recipes = recipes[recipes['is_english']]

    recipes['RecipeName'] = recipes['RecipeName'].apply(clean_text)
    recipes['Ingredients'] = recipes['Ingredients'].apply(clean_text)
    recipes['Instructions'] = recipes['Instructions'].apply(clean_text)

    recipes['recipe_text'] = recipes['RecipeName'] + ' ' + recipes['Ingredients'] + ' ' + recipes['Instructions']

    return recipes


def recommend_recipes(query, vectorizer, svd, processed_recipes, num_recommendations=5):
    query = clean_text(query)
    query_tfidf = vectorizer.transform([query])

    query_tfidf_svd = svd.transform(query_tfidf)

    similarities = cosine_similarity(query_tfidf_svd,
                                     svd.transform(vectorizer.transform(processed_recipes['recipe_text'])))
    similar_indices = similarities.argsort(axis=1)[:, -num_recommendations:][:, ::-1]
    recommendations = processed_recipes.iloc[similar_indices[0]]

    return recommendations


app = Flask(__name__)

file_path = r'recipe_app\myrecipeapp\recipesdata2.xlsx'
processed_recipes = preprocess_data(file_path)

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))
text_tfidf = vectorizer.fit_transform(processed_recipes['recipe_text'])

svd = TruncatedSVD(n_components=50)
text_tfidf_svd = svd.fit_transform(text_tfidf)


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    query = request.form['query']
    num_recommendations = int(request.form.get('num_recommendations', 5))  # Default to 5 recommendations if not specified

    # Use user's query as a prompt for the chatbot
    chatbot_prompt = f"Recommend recipes based on '{query}'"
    openai.api_key = apikey
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=chatbot_prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=1
    )

    chat_response = response.choices[0].text.strip()

    # Extract recipe recommendations from chatbot response
    # Extract recipe recommendations from chatbot response
    recommendations = recommend_recipes(chat_response, vectorizer, svd, processed_recipes, num_recommendations)

    return render_template('recommendation.html', recommendations=recommendations, chat_response=chat_response)


if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred: {e}")