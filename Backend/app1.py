from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Flask setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained T5 model for question generation
model_name = 'valhalla/t5-base-qg-hl'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate question stems
def generate_question_stems(paragraph, num_questions=5):
    input_text = "generate questions: " + paragraph
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=1024, num_beams=5, num_return_sequences=num_questions)
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

@app.route('/transcribe_and_generate_questions', methods=['POST'])
def transcribe_and_generate_questions():
    data = request.get_json()
    video_id = data.get('video_id')
    num_questions = data.get('num_questions', 5)  # Default to 5 questions if not provided

    try:
        # Get the transcription from YouTube
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        paragraph = ' '.join([item['text'] for item in transcript])

        # Generate questions based on the transcribed paragraph
        questions = generate_question_stems(paragraph, num_questions)

        return jsonify({'questions': questions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 
