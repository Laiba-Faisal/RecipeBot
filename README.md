# RecipeBot

RecipeBot is a smart recipe recommendation system that combines traditional NLP techniques (TF-IDF, SVD) with OpenAI's GPT-based chatbot to provide personalized recipe suggestions based on user queries.

## Features

- Natural language query support
- Cleaned and preprocessed recipe data
- GPT-powered prompt-based suggestion
- Cosine similarity-based ranking

## How It Works

1. User submits a recipe idea or ingredient.
2. GPT interprets the query and enhances it as a prompt.
3. TF-IDF + SVD transforms both recipes and query.
4. Cosine similarity finds the most relevant recipes.
