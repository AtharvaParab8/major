from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
import torch
import json
import re
import random
from pytube import YouTube

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/transcript/<video_id>')
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Get transcript with timestamps
        formatted_transcript = []
        current_chapter = None
        
        for item in transcript:
            entry = {
                'text': item['text'],
                'start': item['start'],
                'duration': item['duration']
            }
            
            # If the entry has a chapter marker
            if 'chapter' in item:
                current_chapter = item['chapter']
            
            entry['chapter'] = current_chapter
            formatted_transcript.append(entry)
        
        # Combine text while preserving chapter information
        chapters = {}
        full_text = []
        
        for item in formatted_transcript:
            full_text.append(item['text'])
            if item['chapter']:
                if item['chapter'] not in chapters:
                    chapters[item['chapter']] = {
                        'start_time': item['start'],
                        'texts': []
                    }
                chapters[item['chapter']]['texts'].append(item['text'])
        
        return jsonify({
            'transcript': ' '.join(full_text),
            'chapters': chapters,
            'formatted_transcript': formatted_transcript
        })
    except Exception as e:
        error_message = str(e)
        if "Subtitles are disabled" in error_message:
            return jsonify({'error': 'This video has disabled subtitles/transcription'}), 400
        elif "Video unavailable" in error_message:
            return jsonify({'error': 'This video is unavailable or private'}), 400
        else:
            return jsonify({'error': f'Failed to get transcript: {error_message}'}), 400


@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    data = request.get_json()
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({'error': 'Missing video ID'}), 400

    try:
        # Get video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([item['text'] for item in transcript])

        # Split transcript into words to more precisely identify main content
        words = text.split()
        total_words = len(words)
        
        # Skip first 15% and last 15% more precisely
        start_idx = int(total_words * 0.15)
        end_idx = int(total_words * 0.85)
        
        # Extract main content
        main_content = ' '.join(words[start_idx:end_idx])
        
        # Calculate target summary length (25-30% of main content)
        min_summary_length = max(300, int(len(main_content) * 0.20))
        max_summary_length = max(600, int(len(main_content) * 0.30))

        # Generate longer summary using DistilBART focusing on main content
        inputs = bart_tokenizer.encode(
            "summarize the following content in detail, focusing on key technical concepts, relationships between ideas, and practical applications: " + main_content[:4000], 
            return_tensors="pt", 
            max_length=4000, 
            truncation=True
        )
        summary_ids = bart_model.generate(
            inputs, 
            max_length=max_summary_length,  
            min_length=min_summary_length,   
            length_penalty=2.0, 
            num_beams=5, 
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Extract topics from main content using a more focused prompt
        topic_inputs = bart_tokenizer.encode(
            "extract the main technical topics, key concepts, and important learning points from: " + main_content[:4000], 
            return_tensors="pt", 
            max_length=4000, 
            truncation=True
        )
        topic_ids = bart_model.generate(
            topic_inputs,
            max_length=300,
            min_length=100,
            length_penalty=2.0,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        topics = bart_tokenizer.decode(topic_ids[0], skip_special_tokens=True)
        
        # Format topics as bullet points if not already
        if not topics.startswith('- '):
            topics = '\n'.join(['- ' + t.strip() for t in topics.split('.')])
            
        # Calculate ROUGE scores
        scores = rouge_scorer.score(main_content[:1000], summary)
        rouge_scores = {
            'rouge1': round(scores['rouge1'].fmeasure * 100, 2),
            'rouge2': round(scores['rouge2'].fmeasure * 100, 2),
            'rougeL': round(scores['rougeL'].fmeasure * 100, 2)
        }
        
        # Add statistics
        stats = {
            'original_length': len(text),
            'main_content_length': len(main_content),
            'summary_length': len(summary),
            'compression_ratio': round(len(summary) / len(main_content) * 100, 2),
            'skipped_intro_words': start_idx,
            'skipped_outro_words': total_words - end_idx
        }

        return jsonify({
            'summary': summary,
            'topics': topics,
            'rouge_scores': rouge_scores,
            'stats': stats
        })
        
    except Exception as e:
        error_message = str(e)
        if "Subtitles are disabled" in error_message:
            return jsonify({'error': 'This video has disabled subtitles/transcription'}), 400
        elif "Video unavailable" in error_message:
            return jsonify({'error': 'This video is unavailable or private'}), 400
        else:
            return jsonify({'error': f'Failed to generate summary: {error_message}'}), 400
      
if __name__ == '__main__':
    app.run(debug=True, port=5000)
        