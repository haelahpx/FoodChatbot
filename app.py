from flask import Flask, render_template, request, session
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import spacy
from fuzzywuzzy import process

app = Flask(__name__)
app.secret_key = "secretkey"  # Required for session storage

# Load NLP models
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv('data/recipes.csv')

def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def find_best_food_match(user_input):
    choices = df["food_name"].tolist()
    best_match, score = process.extractOne(user_input, choices)
    return best_match if score > 70 else None

def get_recipe_details(food_name):
    recipe = df[df['food_name'].str.lower() == food_name.lower()]
    if not recipe.empty:
        row = recipe.iloc[0]
        return {
            "ingredients": row['ingredients'],
            "instructions": row['instructions'],
            "calories": row['calories'],
            "category": row['category']
        }
    return None

def generate_response(question, context):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def recommend_recipes(condition):
    query_embedding = sentence_model.encode(condition, convert_to_tensor=True)
    recipe_embeddings = sentence_model.encode(df['category'].tolist(), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, recipe_embeddings)[0]
    top_matches = torch.argsort(similarities, descending=True)[:3]
    recommended_recipes = df.iloc[top_matches.cpu().numpy()]
    return recommended_recipes['food_name'].tolist()

# 🆕 Find highest or lowest calorie dish
def get_extreme_calories(highest=True):
    if highest:
        row = df[df['calories'] == df['calories'].max()].iloc[0]
    else:
        row = df[df['calories'] == df['calories'].min()].iloc[0]
    return f"{row['food_name']} with {row['calories']} kcal"

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    recommended = None

    if "last_food" not in session:
        session["last_food"] = None  # Track last mentioned dish

    if request.method == "POST":
        user_input = request.form.get("question")
        processed_input = preprocess_text(user_input)

        response_list = []  # 🆕 Store multiple responses

        # 🟢 **Check if user is asking for highest/lowest calorie dish**
        if "highest calorie" in processed_input or "most calorie" in processed_input:
            response_list.append("The highest calorie dish is: " + get_extreme_calories(highest=True))
        elif "lowest calorie" in processed_input or "least calorie" in processed_input:
            response_list.append("The lowest calorie dish is: " + get_extreme_calories(highest=False))
        
        # 🔵 **Check if user refers to a previously mentioned food**
        if session["last_food"]:
            food_match = session["last_food"]
        else:
            food_match = find_best_food_match(processed_input)

        # 🟣 **If a food is found, update session and handle multiple requests**
        if food_match:
            session["last_food"] = food_match  # Save context
            details = get_recipe_details(food_match)

            if details:
                if "ingredient" in processed_input:
                    response_list.append(f"Ingredients for {food_match}: {details['ingredients']}")
                if "instruction" in processed_input or "step" in processed_input:
                    response_list.append(f"Instructions for {food_match}: {details['instructions']}")
                if "calorie" in processed_input:
                    response_list.append(f"{food_match} contains {details['calories']} kcal.")

        # 🟡 **If no food is mentioned, try recommending based on condition**
        if not response_list:
            recommended = recommend_recipes(processed_input)
            if not recommended:
                response_list.append("I couldn't find a suitable recommendation. Please be more specific!")

        # Combine all responses into one message
        answer = " | ".join(response_list)

    return render_template("index.html", answer=answer, recommended=recommended)

if __name__ == "__main__":
    app.run(debug=True)
