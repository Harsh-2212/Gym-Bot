from flask import Flask, request, jsonify, render_template
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

model = joblib.load('tfidf_vectorizer.pkl')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

@app.route('/')
def home():
    return render_template('bot.html')

def preprocess_user_input(user_input):
    tokens = word_tokenize(user_input.lower())
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

workout_responses = {
    "chest": "Here's a chest workout: Bench press, Push-ups, Chest fly.",
    "muscle_gain": "For muscle gain, follow this 3-month routine: Monday - Chest and triceps, Wednesday - Back and biceps, Friday - Legs and shoulders.",
    "weight_loss": "For weight loss, try this: 30 minutes of cardio 5 times a week and strength training 3 times a week.",
    "muscle_building": "Here's a muscle building routine: Compound lifts 4 times a week focusing on progressive overload.",
    "cardio": "Here’s a cardio workout plan: 20-30 minutes of jogging or cycling at least 4 times a week.",
    "legs": "A good leg workout includes Squats, Lunges, and Leg Press.",
    "back": "Try these back exercises: Pull-ups, Rows, and Deadlifts.",
    "endurance": "For endurance, combine long runs with interval training."
}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        processed_input = preprocess_user_input(user_message)
        print("Processed input:", processed_input)

        prediction = model.predict([processed_input])[0]

        response_text = workout_responses.get(prediction, "Sorry, I don't have a workout plan for that.")

        response = {
            'response': response_text 
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
