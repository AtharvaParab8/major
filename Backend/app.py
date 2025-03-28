import cohere
from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

# Initialize Cohere client
API_KEY = "IKI636LmJxZLpIJJWOXQMlS5dSBpshN0odoSyTBM"
co = cohere.Client(API_KEY)

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the BART model and tokenizer for summarization
bart_model_name = 'sshleifer/distilbart-cnn-12-6'
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)

# Load the T5 model and tokenizer for question generation
t5_model_name = 'valhalla/t5-base-qg-hl'
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

def summarize_text_with_cohere(text):
    try:
        # Use Cohere to summarize the text
        response = co.summarize(
            text=text,
            length='auto',
            format='auto',
            model='summarize-xlarge',
            additional_command='',
            temperature=0.3,
        )
        # Get the summary from the response
        summary = response.summary
        return summary
    except Exception as e:
        print("Error:", str(e))
        return "Oops! Something went wrong."

@app.route('/generate-summary', methods=['POST'])
def generate_summary():
    data = request.get_json()
    video_id = data.get('video_id')

    try:
        # Get video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([item['text'] for item in transcript])

        # Summarize the transcription text using Cohere
        summary = summarize_text_with_cohere(text)

        return jsonify({'transcription': text, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    video_id = data.get('video_id')

    try:
        # Get video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([item['text'] for item in transcript])

        # Summarize the transcription text using BART
        inputs = bart_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = bart_model.generate(inputs, max_length=800, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return jsonify({'transcription': text, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    data = request.get_json()
    video_id = data.get('video_id')
    num_questions = data.get('num_questions', 5)  # Default to 5 questions if not specified

    try:
        # Get video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([item['text'] for item in transcript])

        # Generate questions using the T5 model
        max_length = 512
        input_text = "generate questions: " + text
        input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=max_length, truncation=True)

        # Generate the specified number of questions
        outputs = t5_model.generate(input_ids, max_length=1024, num_beams=5, num_return_sequences=num_questions)
        questions = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Generate options (dummy options for now, modify this as needed)
        questions_with_options = []
        for question in questions:
            options = [
                f"Option 1 : {question}",
                f"Option 2 : {question}",
                f"Option 3 : {question}",
            ]
            questions_with_options.append({
                'question': question,
                'options': options
            })

        return jsonify({'questions_with_options': questions_with_options})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
