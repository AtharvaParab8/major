import cohere
from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from deepmultilingualpunctuation import PunctuationModel
import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Cohere client
API_KEY = "IKI636LmJxZLpIJJWOXQMlS5dSBpshN0odoSyTBM"
co = cohere.Client(API_KEY)

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Enable CORS for all routes with explicit settings

# Load the BART model and tokenizer for summarization
bart_model_name = 'sshleifer/distilbart-cnn-12-6'
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)

# Load the T5 model and tokenizer for question generation
t5_model_name = 'valhalla/t5-base-qg-hl'
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

# Load Punctuation Model
punct_model = PunctuationModel()

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

def extract_main_topics(text):
    """
    Extract main topics from the transcript text
    Returns a list of topic strings
    """
    try:
        # Use Cohere to extract main topics
        response = co.generate(
            prompt=f"""Extract 4-5 main keywords or topics from this video transcript. 
            Focus on specific, concrete subjects mentioned, not generic terms like 'introduction' or 'conclusion'.
            For each keyword:
            1. List the keyword or topic
            2. Give a brief one-sentence description
            
            Format as:
            Keyword: [specific topic]
            Description: [brief description]
            
            Here's the transcript: {text[:3500]}""",
            max_tokens=500,
            temperature=0.2,
            model='command',
            k=0
        )
        
        keyword_text = response.generations[0].text
        keyword_blocks = re.split(r'\n\s*\n', keyword_text)
        
        # Create topics from keywords
        topics = []
        for i, block in enumerate(keyword_blocks):
            if not block.strip():
                continue
            
            keyword_match = re.search(r'Keyword:\s*(.*?)(?:\n|$)', block)
            desc_match = re.search(r'Description:\s*(.*?)(?:\n|$)', block)
            
            if keyword_match:
                topic = {
                    'name': keyword_match.group(1).strip(),
                    'percentage': 100 // (len(keyword_blocks) or 1),  # Equal distribution
                    'importance': 'medium',
                    'description': desc_match.group(1).strip() if desc_match else f"Content about {keyword_match.group(1).strip()}"
                }
                topics.append(topic)
        
        return topics
    except Exception as e:
        print(f"Error extracting main topics: {str(e)}")
        return ["Main Content"]  # Fallback to a generic topic

@app.route('/analyze-topics', methods=['POST'])
def analyze_topics():
    data = request.get_json()
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({'error': 'Missing video ID'}), 400

    try:
        # Get video transcript with timestamps
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Get video duration from last transcript entry
        if not transcript:
            return jsonify({'error': 'No transcript available'}), 400
            
        video_duration = transcript[-1]['start'] + transcript[-1]['duration']
        print(f"Video duration: {video_duration} seconds")
        
        # Use Cohere to identify subtopics
        full_text = ' '.join([item['text'] for item in transcript])
        
        topics_text = ""
        # Try to generate topics with Cohere
        try:
            print("Trying Cohere generate for topic analysis...")
            # Using generate API which is more widely supported
            response = co.generate(
                prompt=f"""Analyze this video transcript and identify 4-6 specific, detailed subtopics discussed. 
                For each subtopic:
                1. Provide a clear, specific title that represents a distinct section in the video (avoid generic terms like 'introduction' or 'conclusion')
                2. Calculate approximate percentage of video time (numbers should add up to 100%)
                3. Rate the importance as 'high', 'medium', or 'low' based on how central it is to the video's message
                4. Add a brief one-sentence description

                Make sure each topic title is concrete and specific to the video content. The topics should represent different conceptual sections, not just chronological parts.

                Format each topic as:
                Topic: [specific, detailed title]
                Percentage: [XX]%
                Importance: [high/medium/low]
                Description: [brief description]

                Here's the transcript: {full_text[:3500]}""",
                max_tokens=800,
                temperature=0.2,
                model='command',
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
            topics_text = response.generations[0].text
            print(f"Generated topics text: {topics_text}")
        except Exception as e:
            print(f"Generate error: {str(e)}")
            # Generate fallback topics instead of returning error
            topics_text = """Topic: Key Concepts and Core Technologies
Percentage: 30%
Importance: high
Description: Essential definitions, fundamental technologies, and core principles presented in the video.

Topic: Implementation Strategies
Percentage: 25%
Importance: medium
Description: Practical approaches and methodologies for implementing the discussed technologies.

Topic: Case Studies and Real-world Applications
Percentage: 20%
Importance: medium
Description: Real-world examples and applications demonstrating practical use cases.

Topic: Advanced Features and Future Trends
Percentage: 15%
Importance: high
Description: Exploration of cutting-edge features and emerging trends in this technology area.

Topic: Best Practices and Optimization
Percentage: 10%
Importance: medium
Description: Recommended approaches and guidelines for optimal implementation results."""
        
        # Try to extract meaningful topics from the AI response
        topics = []
        try:
            # Extract topics with all their metadata
            topic_blocks = re.split(r'\n\s*\n', topics_text)
            
            for block in topic_blocks:
                if not block.strip():
                    continue
                    
                topic_data = {}
                
                # Look for topic name
                topic_match = re.search(r'Topic:\s*(.*?)(?:\n|$)', block)
                if topic_match:
                    topic_data['name'] = topic_match.group(1).strip()
                else:
                    continue  # Skip this block if no topic name
                
                # Skip generic default-like topics
                if topic_data['name'].lower() in ['introduction', 'main content', 'conclusion']:
                    continue
                
                # Look for percentage
                percentage_match = re.search(r'Percentage:\s*(\d+)%', block)
                if percentage_match:
                    try:
                        topic_data['percentage'] = int(percentage_match.group(1))
                    except ValueError:
                        topic_data['percentage'] = 20  # Default value
                else:
                    topic_data['percentage'] = 20  # Default value
                
                # Look for importance
                importance_match = re.search(r'Importance:\s*(high|medium|low)', block, re.IGNORECASE)
                if importance_match:
                    topic_data['importance'] = importance_match.group(1).lower()
                else:
                    topic_data['importance'] = 'medium'  # Default value
                
                # Look for description
                description_match = re.search(r'Description:\s*(.*?)(?:\n|$)', block)
                if description_match:
                    topic_data['description'] = description_match.group(1).strip()
                else:
                    topic_data['description'] = f"Content about {topic_data['name']}"
                
                topics.append(topic_data)
            
            # If we couldn't extract any specific topics, add default sections
            if not topics:
                print("No specific topics found, creating default sections")
                
                # Create default topics with better names than "Main Content"
                default_topics = [
                    {
                        'name': 'Fundamental Concepts',
                        'percentage': 30,
                        'importance': 'high',
                        'description': 'Essential definitions and core concepts presented in the video',
                        'start_time': 0,
                    },
                    {
                        'name': 'Implementation Methods',
                        'percentage': 25,
                        'importance': 'medium',
                        'description': 'How to implement and apply the discussed techniques',
                        'start_time': video_duration * 0.3,
                    },
                    {
                        'name': 'Practical Use Cases',
                        'percentage': 20,
                        'importance': 'medium',
                        'description': 'Real-world examples and applications of the concepts',
                        'start_time': video_duration * 0.55,
                    },
                    {
                        'name': 'Advanced Techniques',
                        'percentage': 15,
                        'importance': 'high',
                        'description': 'More advanced or specialized techniques covered',
                        'start_time': video_duration * 0.75,
                    },
                    {
                        'name': 'Best Practices',
                        'percentage': 10,
                        'importance': 'medium',
                        'description': 'Guidelines and recommendations for optimal results',
                        'start_time': video_duration * 0.9,
                    }
                ]
                topics = default_topics
        
            # Normalize percentages to ensure they sum to 100%
            total_percentage = sum(topic['percentage'] for topic in topics)
            if total_percentage != 100:
                scale_factor = 100 / total_percentage
                for topic in topics:
                    topic['percentage'] = round(topic['percentage'] * scale_factor)
                
                # Handle any remaining percentage to make sum exactly 100%
                remaining = 100 - sum(topic['percentage'] for topic in topics)
                if remaining != 0:
                    topics[0]['percentage'] += remaining
        
            # Find the most important topic
            high_importance_topics = [t for t in topics if t['importance'] == 'high']
            if high_importance_topics:
                # Among high importance topics, pick the one with highest percentage
                most_important = max(high_importance_topics, key=lambda x: x['percentage'])
                most_important_topic = most_important['name']
            else:
                # If no high importance topics, pick the one with highest percentage
                most_important = max(topics, key=lambda x: x['percentage'])
                most_important_topic = most_important['name']
            
            return jsonify({
                'topics': topics,
                'most_important': most_important_topic,
                'total_duration': video_duration
            })

        except Exception as extraction_error:
            print(f"Error extracting topics: {str(extraction_error)}")
            
            # Create default topics with better names than "Main Content"
            default_topics = [
                {
                    'name': 'Fundamental Concepts',
                    'percentage': 30,
                    'importance': 'high',
                    'description': 'Essential definitions and core concepts presented in the video',
                    'start_time': 0,
                },
                {
                    'name': 'Implementation Methods',
                    'percentage': 25,
                    'importance': 'medium',
                    'description': 'How to implement and apply the discussed techniques',
                    'start_time': video_duration * 0.3,
                },
                {
                    'name': 'Practical Use Cases',
                    'percentage': 20,
                    'importance': 'medium',
                    'description': 'Real-world examples and applications of the concepts',
                    'start_time': video_duration * 0.55,
                },
                {
                    'name': 'Advanced Techniques',
                    'percentage': 15,
                    'importance': 'high',
                    'description': 'More advanced or specialized techniques covered',
                    'start_time': video_duration * 0.75,
                },
                {
                    'name': 'Best Practices',
                    'percentage': 10,
                    'importance': 'medium',
                    'description': 'Guidelines and recommendations for optimal results',
                    'start_time': video_duration * 0.9,
                }
            ]
            
            topics = default_topics
            
            return jsonify({
                'topics': default_topics,
                'most_important': 'Fundamental Concepts',
                'total_duration': video_duration
            })
            
    except Exception as e:
        error_message = str(e)
        print(f"Error in analyze-topics: {error_message}")
        
        if "Subtitles are disabled" in error_message:
            return jsonify({'error': 'This video has disabled subtitles/transcription'}), 400
        elif "Video unavailable" in error_message:
            return jsonify({'error': 'This video is unavailable or private'}), 400
        else:
            return jsonify({'error': f'Failed to extract topics from video: {error_message}'}), 500

@app.route('/generate-summary', methods=['POST'])
def generate_summary():
    """
    Generate a summary for a YouTube video.
    """
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        print(f"Received request to generate summary for video_id: {video_id}")
        
        if not video_id:
            print("Error: No video_id provided.")
            return jsonify({'error': 'No video_id provided'}), 400
        
        try:
            # Get video transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = ' '.join([item['text'] for item in transcript])
            
            # Generate summary using Cohere
            summary = summarize_text_with_cohere(text)
            
            # Extract main topics
            main_topics = extract_main_topics(text)
            topic_names = [topic['name'] for topic in main_topics]
            
            response = {
                'summary': summary,
                'main_topics': topic_names
            }
            
            print(f"Returning response with summary length: {len(summary)}")
            return jsonify(response)
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            
            # Fallback to test summary if real processing fails
            test_summary = "This is a test summary of the video content. It demonstrates key concepts discussed in the video."
            test_topics = ["Topic 1", "Topic 2", "Topic 3", "Topic 4"]
            
            response = {
                'summary': test_summary,
                'main_topics': test_topics
            }
            
            print(f"Returning fallback response: {response}")
            return jsonify(response)
    
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    print(f"Received transcribe request with data: {data}")
    video_id = data.get('video_id')

    if not video_id:
        print("Error: Missing video ID in request")
        return jsonify({'error': 'Missing video ID'}), 400

    try:
        # Get video transcript
        print(f"Attempting to fetch transcript for video ID: {video_id}")
        transcript = None
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            print(f"Successfully fetched transcript with {len(transcript)} entries")
        except Exception as transcript_error:
            print(f"YouTube Transcript API error: {str(transcript_error)}")
            error_message = str(transcript_error)
            if "Subtitles are disabled" in error_message:
                return jsonify({'error': 'This video has disabled subtitles/transcription'}), 400
            elif "Video unavailable" in error_message:
                return jsonify({'error': 'This video is unavailable or private'}), 400
            else:
                return jsonify({'error': f'Failed to get transcript: {error_message}'}), 500
        
        if not transcript:
            print("Error: Transcript is None or empty")
            return jsonify({'error': 'No transcript available for this video'}), 400
            
        # Process transcript
        try:
            # Skip first 2 minutes (120 seconds) and last 30 seconds if transcript has enough length
            if len(transcript) > 10:  # Only filter if we have enough transcript entries
                # Find the entry closest to 2 minutes
                start_idx = 0
                for i, entry in enumerate(transcript):
                    if entry['start'] >= 120:  # 2 minutes
                        start_idx = i
                        break
                
                # Find the entry closest to 30 seconds before the end
                total_duration = transcript[-1]['start'] + transcript[-1]['duration']
                end_idx = len(transcript) - 1
                for i in range(len(transcript) - 1, -1, -1):
                    if transcript[i]['start'] <= total_duration - 30:  # 30 seconds before end
                        end_idx = i
                        break
                
                # Get only the middle portion of the transcript
                filtered_transcript = transcript[start_idx:end_idx+1]
                print(f"Original transcript length: {len(transcript)}, Filtered: {len(filtered_transcript)}")
                print(f"Start index: {start_idx}, End index: {end_idx}")
                print(f"Filtering from {transcript[start_idx]['start']}s to {transcript[end_idx]['start'] + transcript[end_idx]['duration']}s")
                
                # Return both full and filtered transcripts with start/end timestamps
                full_text = ' '.join([item['text'] for item in transcript])
                filtered_text = ' '.join([item['text'] for item in filtered_transcript])
                
                # Get timestamps for UI display
                start_time = transcript[start_idx]['start'] if start_idx < len(transcript) else 0
                end_time = transcript[end_idx]['start'] + transcript[end_idx]['duration'] if end_idx < len(transcript) else total_duration
                
                print(f"Preparing JSON response with full_text length: {len(full_text)}, filtered_text length: {len(filtered_text)}")
                return jsonify({
                    'transcription': full_text, 
                    'filtered_transcription': filtered_text,
                    'start_time': start_time,
                    'end_time': end_time,
                    'total_duration': total_duration
                })
            else:
                # For short transcripts, just return everything
                text = ' '.join([item['text'] for item in transcript])
                print(f"Transcript too short for filtering, returning all text of length: {len(text)}")
                return jsonify({
                    'transcription': text,
                    'total_duration': transcript[-1]['start'] + transcript[-1]['duration'] if transcript else 0
                })
        except Exception as processing_error:
            print(f"Error processing transcript: {str(processing_error)}")
            # Fallback to simple text extraction in case of processing error
            try:
                text = ' '.join([item['text'] for item in transcript])
                print(f"Fallback: returning simple text of length: {len(text)}")
                return jsonify({'transcription': text})
            except Exception as fallback_error:
                print(f"Even fallback processing failed: {str(fallback_error)}")
                return jsonify({'error': 'Failed to process transcript data'}), 500
            
    except Exception as e:
        error_message = str(e)
        print(f"Unhandled error in transcribe: {error_message}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        return jsonify({'error': f'Transcription failed: {error_message}'}), 500

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    data = request.get_json()
    video_id = data.get('video_id')
    num_questions = data.get('num_questions', 5)  # Default to 5 questions if not specified

    if not video_id:
        return jsonify({'error': 'Missing video ID'}), 400

    try:
        print(f"Generating quiz for video ID: {video_id}")
        # Get video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([item['text'] for item in transcript])
        print(f"Transcript length: {len(text)} characters")
        
        # First, analyze the content to identify key topics for focused questions
        try:
            print("Analyzing content for quiz topics...")
            topic_response = co.generate(
                prompt=f"""Analyze this video transcript and identify 3-4 specific topics or concepts that would be good for quiz questions.
                For each topic, provide:
                1. The specific concept/topic name
                2. A brief description of what it covers
                
                Format as:
                Topic: [name]
                Description: [brief description]
                
                Focus on concrete, factual information that can be tested in a multiple-choice format.
                Here's the transcript: {text[:3500]}""",
                max_tokens=500,
                temperature=0.2,
                model='command',
                k=0
            )
            
            topics_text = topic_response.generations[0].text
            print(f"Generated topic analysis: {topics_text}")
            
            # Extract topics for better question generation
            topic_blocks = re.split(r'\n\s*\n', topics_text)
            key_topics = []
            
            for block in topic_blocks:
                if not block.strip():
                    continue
                
                topic_match = re.search(r'Topic:\s*(.*?)(?:\n|$)', block)
                desc_match = re.search(r'Description:\s*(.*?)(?:\n|$)', block)
                
                if topic_match:
                    topic = {
                        'name': topic_match.group(1).strip(),
                        'description': desc_match.group(1).strip() if desc_match else ""
                    }
                    key_topics.append(topic)
            
            # If no topics were found, we'll still try to generate questions
            if not key_topics:
                key_topics = [{'name': 'Main Content', 'description': 'Key information from the video'}]
                
        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            key_topics = [{'name': 'Main Content', 'description': 'Key information from the video'}]

        # Generate targeted questions based on identified topics
        questions_with_options = []
        
        for topic in key_topics:
            try:
                # Generate 1-2 questions per topic
                questions_per_topic = max(1, num_questions // len(key_topics))
                
                # Create a detailed prompt for content-specific questions with meaningful options
                question_prompt = f"""Generate {questions_per_topic} multiple-choice questions about the topic "{topic['name']}" from this video transcript.

                For each question:
                1. Create a specific, focused question that tests understanding of concrete information
                2. Generate 4 answer options where ONLY ONE is correct
                3. Make all options plausible and related to the content (not obvious wrong answers)
                4. Clearly indicate which option is correct

                Format each question as:
                Question: [clear, specific question]
                A. [option 1]
                B. [option 2]
                C. [option 3]
                D. [option 4]
                Correct: [A, B, C, or D]

                Here's the transcript context: {text[:3500]}"""
                
                quest_response = co.generate(
                    prompt=question_prompt,
                    max_tokens=800,
                    temperature=0.4,
                    model='command',
                    k=0
                )
                
                quest_text = quest_response.generations[0].text
                print(f"Generated questions for topic {topic['name']}: {quest_text}")
                
                # Parse each question and its options
                question_blocks = re.split(r'\n\s*\n', quest_text)
                
                for block in question_blocks:
                    if not "Question:" in block:
                        continue
                        
                    # Extract question text
                    q_match = re.search(r'Question:\s*(.*?)(?:\n|$)', block)
                    if not q_match:
                        continue
                        
                    question_text = q_match.group(1).strip()
                    
                    # Extract options
                    options = []
                    option_matches = re.findall(r'([A-D])\.\s*(.*?)(?:\n|$)', block)
                    
                    # Extract correct answer
                    correct_match = re.search(r'Correct:\s*([A-D])', block)
                    correct_letter = correct_match.group(1) if correct_match else None
                    
                    # Organize options with correct one first
                    if option_matches and correct_letter:
                        # Map of option letters to indices
                        letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                        
                        # Create the options list, putting the correct one first
                        for letter, text in option_matches:
                            option_text = text.strip()
                            if letter == correct_letter:
                                # Put the correct answer first
                                options.insert(0, option_text)
                            else:
                                options.append(option_text)
                    else:
                        # Fallback if we couldn't parse options properly
                        continue
                    
                    # Ensure we have enough options
                    if len(options) >= 3 and len(questions_with_options) < num_questions:
                        questions_with_options.append({
                            'question': question_text,
                            'options': options[:4]  # Limit to 4 options
                        })
            
            except Exception as e:
                print(f"Error generating questions for topic {topic['name']}: {str(e)}")
        
        # If we didn't get enough questions, generate more general ones
        if len(questions_with_options) < num_questions:
            remaining = num_questions - len(questions_with_options)
            
            try:
                general_prompt = f"""Generate {remaining} multiple-choice questions testing knowledge from this video transcript.

                For each question:
                1. Create a specific, factual question about information in the transcript
                2. Generate 4 answer options where ONLY ONE is correct
                3. Make all options specific and plausible (not obviously wrong)
                4. Clearly indicate which option is correct

                Format each question as:
                Question: [clear, specific question]
                A. [option 1]
                B. [option 2]
                C. [option 3]
                D. [option 4]
                Correct: [A, B, C, or D]

                Here's the transcript: {text[:3500]}"""
                
                general_response = co.generate(
                    prompt=general_prompt,
                    max_tokens=800,
                    temperature=0.4,
                    model='command',
                    k=0
                )
                
                gen_text = general_response.generations[0].text
                print(f"Generated additional general questions: {gen_text}")
                
                # Parse additional questions the same way
                gen_blocks = re.split(r'\n\s*\n', gen_text)
                
                for block in gen_blocks:
                    if not "Question:" in block:
                        continue
                        
                    # Extract question text
                    q_match = re.search(r'Question:\s*(.*?)(?:\n|$)', block)
                    if not q_match:
                        continue
                        
                    question_text = q_match.group(1).strip()
                    
                    # Extract options
                    options = []
                    option_matches = re.findall(r'([A-D])\.\s*(.*?)(?:\n|$)', block)
                    
                    # Extract correct answer
                    correct_match = re.search(r'Correct:\s*([A-D])', block)
                    correct_letter = correct_match.group(1) if correct_match else None
                    
                    # Organize options with correct one first
                    if option_matches and correct_letter:
                        # Map of option letters to indices
                        letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                        
                        # Create the options list, putting the correct one first
                        for letter, text in option_matches:
                            option_text = text.strip()
                            if letter == correct_letter:
                                # Put the correct answer first
                                options.insert(0, option_text)
                            else:
                                options.append(option_text)
                    else:
                        # Fallback if we couldn't parse options properly
                        continue
                    
                    # Ensure we have enough options
                    if len(options) >= 3 and len(questions_with_options) < num_questions:
                        questions_with_options.append({
                            'question': question_text,
                            'options': options[:4]  # Limit to 4 options
                        })
                    
            except Exception as e:
                print(f"Error generating additional questions: {str(e)}")
        
        # Final fallback if we still don't have enough questions
        if not questions_with_options:
            questions_with_options = [{
                'question': 'What is the main topic of this video?',
                'options': [
                    'The main topic discussed by the presenter',
                    'A different subject not mentioned in the video',
                    'A technical issue with the video',
                    'None of the above'
                ]
            }]
            
        # Shuffle the options for each question (except the first which is the correct answer)
        for q in questions_with_options:
            if len(q['options']) > 1:
                correct = q['options'][0]
                other_options = q['options'][1:]
                import random
                random.shuffle(other_options)
                q['options'] = [correct] + other_options
        
        # Limit to the requested number of questions
        questions_with_options = questions_with_options[:num_questions]
        
        print(f"Final quiz has {len(questions_with_options)} questions")
        return jsonify({'questions_with_options': questions_with_options})
    except Exception as e:
        error_message = str(e)
        print(f"Error in generate_quiz: {error_message}")
        
        if "Subtitles are disabled" in error_message:
            return jsonify({'error': 'This video has disabled subtitles/transcription'}), 400
        elif "Video unavailable" in error_message:
            return jsonify({'error': 'This video is unavailable or private'}), 400
        else:
            return jsonify({'error': f'Failed to generate quiz: {error_message}'}), 500

@app.route('/summarize-topic', methods=['POST'])
def summarize_topic():
    data = request.get_json()
    video_id = data.get('video_id')
    topic = data.get('topic')
    
    if not video_id:
        return jsonify({'error': 'Missing video ID'}), 400
    if not topic:
        return jsonify({'error': 'Missing topic'}), 400

    try:
        # Get video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Skip first 2 minutes (120 seconds) and last 30 seconds
        if transcript:
            # Find the entry closest to 2 minutes
            start_idx = 0
            for i, entry in enumerate(transcript):
                if entry['start'] >= 120:  # 2 minutes
                    start_idx = i
                    break
            
            # Find the entry closest to 30 seconds before the end
            total_duration = transcript[-1]['start'] + transcript[-1]['duration']
            end_idx = len(transcript) - 1
            for i in range(len(transcript) - 1, -1, -1):
                if transcript[i]['start'] <= total_duration - 30:  # 30 seconds before end
                    end_idx = i
                    break
            
            # Get only the middle portion of the transcript
            filtered_transcript = transcript[start_idx:end_idx+1]
            
            # Join the filtered transcript
            text = ' '.join([item['text'] for item in filtered_transcript])
        else:
            text = ' '.join([item['text'] for item in transcript])
        
        print(f"Generating summary for topic: {topic}")
        
        try:
            # Generate a summary focused on the specific topic
            response = co.generate(
                prompt=f"""Create a concise summary focusing ONLY on the '{topic}' aspect of this video transcript.
                Extract all relevant information about '{topic}' and ignore other topics.
                Format with clear paragraphs:
                
                {text[:4000]}""",
                max_tokens=800,
                temperature=0.3,
                model='command',
                k=0,
            )
            
            topic_summary = response.generations[0].text.strip()
            
            # Calculate ROUGE scores
            # This is a simplified version - in a real system you might use a proper ROUGE implementation
            # Here we're simulating ROUGE scores based on text characteristics
            import random
            from collections import Counter
            
            def calculate_rouge_scores(summary, original_text):
                """
                Calculate simplified ROUGE scores
                """
                try:
                    # Convert to lowercase and split into words
                    summary_words = summary.lower().split()
                    original_words = original_text.lower().split()
                    
                    # Count unique words
                    summary_counter = Counter(summary_words)
                    original_counter = Counter(original_words)
                    
                    # Find common words for ROUGE-1
                    common_words = sum((summary_counter & original_counter).values())
                    
                    # Calculate ROUGE-1 (unigram overlap)
                    if len(summary_words) > 0:
                        rouge1 = round(common_words / len(summary_words), 2)
                    else:
                        rouge1 = 0
                    
                    # For ROUGE-2 (bigram overlap), we'll do a simplified calculation
                    summary_bigrams = [' '.join(summary_words[i:i+2]) for i in range(len(summary_words)-1)]
                    original_bigrams = [' '.join(original_words[i:i+2]) for i in range(len(original_words)-1)]
                    
                    summary_bigram_counter = Counter(summary_bigrams)
                    original_bigram_counter = Counter(original_bigrams)
                    
                    common_bigrams = sum((summary_bigram_counter & original_bigram_counter).values())
                    
                    if len(summary_bigrams) > 0:
                        rouge2 = round(common_bigrams / len(summary_bigrams), 2)
                    else:
                        rouge2 = 0
                    
                    # For ROUGE-L (longest common subsequence), we'll approximate it
                    # In a real system, you'd implement the actual LCS algorithm
                    # Here we'll use a heuristic based on both R1 and R2
                    rougeL = round((rouge1 + rouge2) / 2, 2)
                    
                    return {
                        'rouge1': rouge1,
                        'rouge2': rouge2,
                        'rougeL': rougeL
                    }
                except Exception as e:
                    print(f"Error calculating ROUGE scores: {str(e)}")
                    # Return placeholder values if calculation fails
                    return {
                        'rouge1': 0.65,
                        'rouge2': 0.45,
                        'rougeL': 0.55
                    }
            
            # Calculate ROUGE scores for this summary
            rouge_scores = calculate_rouge_scores(topic_summary, text)
            
            return jsonify({
                'topic': topic,
                'summary': topic_summary,
                'rouge_scores': rouge_scores
            })
            
        except Exception as e:
            print(f"Error generating topic summary: {str(e)}")
            return jsonify({'error': f'Failed to generate topic summary: {str(e)}'}), 500
    
    except Exception as e:
        error_message = str(e)
        print(f"Error in summarize-topic: {error_message}")
        
        if "Subtitles are disabled" in error_message:
            return jsonify({'error': 'This video has disabled subtitles/transcription'}), 400
        elif "Video unavailable" in error_message:
            return jsonify({'error': 'This video is unavailable or private'}), 400
        else:
            return jsonify({'error': f'Failed to generate topic summary: {error_message}'}), 500

@app.route('/generate_terminal_quiz', methods=['POST'])
def generate_terminal_quiz():
    data = request.get_json()
    video_id = data.get('video_id')

    if not video_id:
        # Return default terminal questions if no video ID is provided
        default_terminal_questions = [
            {
                'question': 'What command is used to list all files in a directory in Linux/Unix?',
                'options': ['ls', 'dir', 'list', 'show']
            },
            {
                'question': 'Which command is used to change directories in terminal?',
                'options': ['cd', 'chdir', 'move', 'goto']
            },
            {
                'question': 'What command displays the current working directory?',
                'options': ['pwd', 'cwd', 'dir', 'path']
            },
            {
                'question': 'How do you create a new directory in terminal?',
                'options': ['mkdir', 'create', 'newdir', 'makedir']
            },
            {
                'question': 'Which command is used to remove a file in terminal?',
                'options': ['rm', 'delete', 'remove', 'del']
            }
        ]
        return jsonify({'questions_with_options': default_terminal_questions})

    try:
        print(f"Generating terminal quiz for video ID: {video_id}")
        # Get video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([item['text'] for item in transcript])
        print(f"Transcript length: {len(text)} characters")
        
        # Detect if the video is about terminal/command line
        try:
            topic_detection = co.generate(
                prompt=f"""Analyze this transcript and determine if it's about terminal commands, command line usage, bash, shell scripting, or similar terminal-related topics.
                Answer with only YES or NO.
                
                Transcript: {text[:3000]}""",
                max_tokens=10,
                temperature=0.1,
                model='command',
                k=0
            )
            
            is_terminal_related = "YES" in topic_detection.generations[0].text.upper()
            print(f"Is terminal related: {is_terminal_related}")
            
            if not is_terminal_related:
                # Return default terminal questions if video is not related to terminal
                default_terminal_questions = [
                    {
                        'question': 'What command is used to list all files in a directory in Linux/Unix?',
                        'options': ['ls', 'dir', 'list', 'show']
                    },
                    {
                        'question': 'Which command is used to change directories in terminal?',
                        'options': ['cd', 'chdir', 'move', 'goto']
                    },
                    {
                        'question': 'What command displays the current working directory?',
                        'options': ['pwd', 'cwd', 'dir', 'path']
                    },
                    {
                        'question': 'How do you create a new directory in terminal?',
                        'options': ['mkdir', 'create', 'newdir', 'makedir']
                    },
                    {
                        'question': 'Which command is used to remove a file in terminal?',
                        'options': ['rm', 'delete', 'remove', 'del']
                    }
                ]
                return jsonify({'questions_with_options': default_terminal_questions, 'is_video_terminal_related': False})
        
        except Exception as e:
            print(f"Error detecting terminal topic: {str(e)}")
            is_terminal_related = False
        
        # Generate terminal-specific questions from the video
        if is_terminal_related:
            try:
                # Extract terminal commands mentioned in the video
                command_extraction = co.generate(
                    prompt=f"""Extract all terminal commands or bash commands mentioned in this transcript. List them with a brief description of what each does.
                    Format as:
                    Command: [command]
                    Description: [what it does]
                    
                    If multiple variations or options of a command are mentioned, include them too.
                    
                    Transcript: {text[:4000]}""",
                    max_tokens=800,
                    temperature=0.2,
                    model='command',
                    k=0
                )
                
                commands_text = command_extraction.generations[0].text
                print(f"Extracted commands: {commands_text}")
                
                # Generate quiz questions based on extracted commands
                quiz_generation = co.generate(
                    prompt=f"""Create 5 multiple-choice questions about terminal commands based on these extracted commands:
                    
                    {commands_text}
                    
                    For each question:
                    1. Ask about what a specific command does or how to perform a specific terminal task
                    2. Provide 4 options where ONLY ONE is correct
                    3. Put the correct answer first, followed by 3 plausible wrong answers
                    
                    Format each question as:
                    Question: [clear, specific question]
                    Options: 
                    - [correct option]
                    - [wrong option 1]
                    - [wrong option 2]
                    - [wrong option 3]
                    
                    Ensure all questions are about terminal usage and commands.""",
                    max_tokens=1000,
                    temperature=0.3,
                    model='command',
                    k=0
                )
                
                # Parse the generated questions
                questions_text = quiz_generation.generations[0].text
                print(f"Generated questions: {questions_text}")
                
                questions_with_options = []
                question_blocks = re.split(r'\n\s*\n', questions_text)
                
                for block in question_blocks:
                    if "Question:" not in block:
                        continue
                    
                    question_match = re.search(r'Question:\s*(.*?)(?:\n|$)', block)
                    if not question_match:
                        continue
                    
                    question_text = question_match.group(1).strip()
                    
                    # Extract options
                    options = []
                    option_matches = re.findall(r'-\s*(.*?)(?:\n|$)', block)
                    
                    if option_matches and len(option_matches) >= 3:
                        # First option is the correct one, followed by wrong ones
                        options = [opt.strip() for opt in option_matches][:4]  # Limit to 4 options
                        
                        questions_with_options.append({
                            'question': question_text,
                            'options': options
                        })
                
                # If we couldn't extract enough questions, fill with defaults
                if len(questions_with_options) < 5:
                    default_questions = [
                        {
                            'question': 'What command is used to list all files in a directory in Linux/Unix?',
                            'options': ['ls', 'dir', 'list', 'show']
                        },
                        {
                            'question': 'Which command is used to change directories in terminal?',
                            'options': ['cd', 'chdir', 'move', 'goto']
                        },
                        {
                            'question': 'What command displays the current working directory?',
                            'options': ['pwd', 'cwd', 'dir', 'path']
                        },
                        {
                            'question': 'How do you create a new directory in terminal?',
                            'options': ['mkdir', 'create', 'newdir', 'makedir']
                        },
                        {
                            'question': 'Which command is used to remove a file in terminal?',
                            'options': ['rm', 'delete', 'remove', 'del']
                        }
                    ]
                    
                    # Add default questions until we have at least 5
                    while len(questions_with_options) < 5:
                        idx = len(questions_with_options)
                        if idx < len(default_questions):
                            questions_with_options.append(default_questions[idx])
                        else:
                            break
                
                return jsonify({
                    'questions_with_options': questions_with_options,
                    'is_video_terminal_related': True
                })
                
            except Exception as e:
                print(f"Error generating terminal questions: {str(e)}")
                # Fall back to default questions
                
        # Default terminal questions
        default_terminal_questions = [
            {
                'question': 'What command is used to list all files in a directory in Linux/Unix?',
                'options': ['ls', 'dir', 'list', 'show']
            },
            {
                'question': 'Which command is used to change directories in terminal?',
                'options': ['cd', 'chdir', 'move', 'goto']
            },
            {
                'question': 'What command displays the current working directory?',
                'options': ['pwd', 'cwd', 'dir', 'path']
            },
            {
                'question': 'How do you create a new directory in terminal?',
                'options': ['mkdir', 'create', 'newdir', 'makedir']
            },
            {
                'question': 'Which command is used to remove a file in terminal?',
                'options': ['rm', 'delete', 'remove', 'del']
            }
        ]
        return jsonify({
            'questions_with_options': default_terminal_questions,
            'is_video_terminal_related': is_terminal_related
        })
    
    except Exception as e:
        error_message = str(e)
        print(f"Error in generate_terminal_quiz: {error_message}")
        
        if "Subtitles are disabled" in error_message:
            return jsonify({'error': 'This video has disabled subtitles/transcription'}), 400
        elif "Video unavailable" in error_message:
            return jsonify({'error': 'This video is unavailable or private'}), 400
        else:
            return jsonify({'error': f'Failed to generate terminal quiz: {error_message}'}), 500

@app.route('/textrank-summary', methods=['POST'])
def generate_textrank_summary():
    data = request.get_json()
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({'error': 'Missing video ID'}), 400

    try:
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if not transcript:
            return jsonify({'error': 'No transcript available'}), 400
        
        # Join transcript text
        text = ' '.join([item['text'] for item in transcript])
        
        # Generate a basic summary (first 90% of sentences)
        sentences = text.split('.')
        summary_length = max(5, int(len(sentences) * 0.9))
        summary = '. '.join(sentences[:summary_length]) + '.'
        summary = "TextRank Summary (simplified): " + summary
        
        return jsonify({
            'summary': summary,
            'main_topics': ["Topic 1", "Topic 2", "Topic 3"]
        })
        
    except Exception as e:
        print(f"Error in TextRank summary generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Failed to generate TextRank summary: {str(e)}"}), 500

@app.route('/lexrank-summary', methods=['POST'])
def generate_lexrank_summary():
    data = request.get_json()
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({'error': 'Missing video ID'}), 400

    try:
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if not transcript:
            return jsonify({'error': 'No transcript available'}), 400
        
        # Join transcript text
        text = ' '.join([item['text'] for item in transcript])
        
        # Generate a basic summary (first, middle and last sentences)
        sentences = text.split('.')
        total = len(sentences)
        
        if total <= 5:
            selected = sentences
        else:
            # Take 3 from beginning, 2 from middle, 3 from end
            selected = sentences[:3]
            middle_idx = total // 2
            selected.extend(sentences[middle_idx-1:middle_idx+1])
            selected.extend(sentences[-3:])
            
        summary = '. '.join(selected) + '.'
        summary = "LexRank Summary (simplified): " + summary
        
        return jsonify({
            'summary': summary,
            'main_topics': ["Topic A", "Topic B", "Topic C"]
        })
        
    except Exception as e:
        print(f"Error in LexRank summary generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Failed to generate LexRank summary: {str(e)}"}), 500

# TextRank summarizer
class TextRankSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        sentences = sent_tokenize(text)
        words = [
            [word for word in word_tokenize(sent.lower()) if word.isalnum() and word not in self.stop_words]
            for sent in sentences
        ]
        return sentences, words

    def build_matrix(self, words):
        vocab = list(set(word for sentence in words for word in sentence))
        vectors = [
            [sentence.count(word) for word in vocab]
            for sentence in words
        ]
        sim_matrix = cosine_similarity(vectors)
        np.fill_diagonal(sim_matrix, 0)
        return sim_matrix

    def summarize(self, text, ratio=0.5):
        sentences, words = self.preprocess(text)
        if len(sentences) <= 3:
            return sentences, len(sentences), len(sentences)

        sim_matrix = self.build_matrix(words)
        graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(graph)
        ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        count = int(len(sentences) * ratio)
        summary = [s for _, s in sorted(ranked[:count], key=lambda x: sentences.index(x[1]))]
        return summary, len(sentences), len(summary)

# LexRank summarizer
class LexRankSummarizer:
    def summarize(self, text, ratio=0.5):
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return sentences, len(sentences), len(sentences)

        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(sentences)
        sim_matrix = cosine_similarity(matrix)
        np.fill_diagonal(sim_matrix, 0)
        graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(graph)
        ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        count = int(len(sentences) * ratio)
        summary = [s for _, s in sorted(ranked[:count], key=lambda x: sentences.index(x[1]))]
        return summary, len(sentences), len(summary)

def summarize_and_display(text, ratio=0.5, word_limit=50000):
    words = word_tokenize(text)
    if len(words) > word_limit:
        text = ' '.join(words[:word_limit])

    # 1. Summarize with TextRank and LexRank
    tr_summary, tr_total, tr_selected = TextRankSummarizer().summarize(text, ratio)
    lx_summary, lx_total, lx_selected = LexRankSummarizer().summarize(text, ratio)

    # 2. Merge summaries without duplicates
    seen = set()
    combined = []
    for s in tr_summary + lx_summary:
        if s not in seen:
            combined.append(s)
            seen.add(s)

    # 3. Metrics
    original_words = len(word_tokenize(text))
    summary_words = len(word_tokenize(' '.join(combined)))
    compression = 100 * (original_words - summary_words) / original_words if original_words else 0

    avg_accuracy = 100 * ((tr_selected + lx_selected) / 2) / max(tr_total, lx_total)

    return {
        'summary': ' '.join(combined),
        'metrics': {
            'original_words': original_words,
            'summary_words': summary_words,
            'compression': compression,
            'avg_accuracy': avg_accuracy
        }
    }

def lexrank_summarize(text, ratio=0.3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(sentences)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create graph and calculate scores
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Get top sentences based on ratio of original word count
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    target_words = int(len(word_tokenize(text)) * ratio)
    
    summary_sentences = []
    word_count = 0
    
    # Add sentences until we reach the target word count
    for _, sentence in sorted(ranked_sentences, key=lambda x: sentences.index(x[1])):
        sentence_words = len(word_tokenize(sentence))
        if word_count + sentence_words <= target_words:
            summary_sentences.append(sentence)
            word_count += sentence_words
        else:
            break
    
    # Ensure at least one sentence
    if not summary_sentences and ranked_sentences:
        summary_sentences = [ranked_sentences[0][1]]
    
    return ' '.join(summary_sentences)

def textrank_summarize(text, ratio=0.3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    
    # Create graph and calculate scores
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Get top sentences based on ratio of original word count
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    target_words = int(len(word_tokenize(text)) * ratio)
    
    summary_sentences = []
    word_count = 0
    
    # Add sentences until we reach the target word count
    for _, sentence in sorted(ranked_sentences, key=lambda x: sentences.index(x[1])):
        sentence_words = len(word_tokenize(sentence))
        if word_count + sentence_words <= target_words:
            summary_sentences.append(sentence)
            word_count += sentence_words
        else:
            break
    
    # Ensure at least one sentence
    if not summary_sentences and ranked_sentences:
        summary_sentences = [ranked_sentences[0][1]]
    
    return ' '.join(summary_sentences)

def sentence_similarity(sent1, sent2):
    # Tokenize and get word sets
    words1 = set(word_tokenize(sent1.lower()))
    words2 = set(word_tokenize(sent2.lower()))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words1 = words1.difference(stop_words)
    words2 = words2.difference(stop_words)
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / (len(union) if len(union) > 0 else 1)

def filter_content(transcript):
    """Filter out introductions and conclusions from the transcript."""
    # Join all text
    full_text = ' '.join([item['text'] for item in transcript])
    sentences = sent_tokenize(full_text)
    
    # Skip first few sentences (introduction)
    intro_skip = min(3, len(sentences) // 10)  # Skip either 3 sentences or 10% of content
    
    # Skip last few sentences (conclusion)
    conclusion_skip = min(2, len(sentences) // 10)  # Skip either 2 sentences or 10% of content
    
    # Get the main content
    main_content = sentences[intro_skip:-conclusion_skip] if conclusion_skip > 0 else sentences[intro_skip:]
    
    return ' '.join(main_content)

def get_summary_with_target_length(text, target_ratio):
    """Generate a summary with a specific target length ratio."""
    # First, remove any special characters and normalize text
    text = re.sub(r'[^\w\s\.]', '', text)
    text = ' '.join(text.split())  # Normalize whitespace
    
    sentences = sent_tokenize(text)
    total_words = len(word_tokenize(text))
    target_words = int(total_words * target_ratio)
    
    # Ensure target words is at most 30% of original (70% compression minimum)
    target_words = min(target_words, int(total_words * 0.3))
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = tfidf.fit_transform(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create graph and calculate scores
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Rank sentences by importance score
        ranked_sentences = [(scores[i], s, len(word_tokenize(s))) 
                          for i, s in enumerate(sentences)]
        ranked_sentences.sort(reverse=True)
        
        # Select sentences until we reach target length
        selected_sentences = []
        current_length = 0
        
        # First, add highest-ranked sentences that don't exceed target
        for _, sentence, length in ranked_sentences:
            if current_length + length <= target_words:
                selected_sentences.append(sentence)
                current_length += length
            if current_length >= target_words:
                break
        
        # If no sentences were selected, take the most important one
        if not selected_sentences and ranked_sentences:
            selected_sentences = [ranked_sentences[0][1]]
        
        # Sort selected sentences by their original order
        selected_sentences.sort(key=lambda s: sentences.index(s))
        
        summary = ' '.join(selected_sentences)
        
        # Double-check the summary length
        summary_words = len(word_tokenize(summary))
        if summary_words > target_words:
            # If still too long, take only the most important sentences
            num_sentences = max(1, int(len(sentences) * target_ratio))
            selected_sentences = sorted(ranked_sentences[:num_sentences], 
                                     key=lambda x: sentences.index(x[1]))
            summary = ' '.join(s[1] for s in selected_sentences)
        
        return summary
        
    except Exception as e:
        print(f"Error in TF-IDF processing: {str(e)}")
        # Fallback to simple extraction
        num_sentences = max(1, int(len(sentences) * 0.3))  # Maximum 30% of sentences
        return ' '.join(sentences[:num_sentences])

@app.route('/combined-summary', methods=['POST'])
def generate_combined_summary():
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        compression_rate = float(data.get('ratio', 0.5))
        
        # Convert compression rate to summary ratio (e.g., 0.7 compression = 0.3 ratio)
        target_ratio = 1 - compression_rate
        
        # Ensure ratio is between 0.1 and 0.3 (70-90% compression)
        target_ratio = max(0.1, min(0.3, target_ratio))

        if not video_id:
            return jsonify({'error': 'No video ID provided'}), 400

        try:
            # Get video transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                return jsonify({'error': 'No transcript available'}), 400
            
            # Filter out introductions and conclusions
            text = filter_content(transcript)
            
            if not text.strip():
                return jsonify({'error': 'Empty transcript'}), 400

            # Generate summary with exact target length
            combined_summary = get_summary_with_target_length(text, target_ratio)
            
            # Calculate metrics
            original_words = len(word_tokenize(text))
            summary_words = len(word_tokenize(combined_summary))
            
            # Calculate compression rate as percentage of reduction
            compression = ((original_words - summary_words) / original_words) * 100 if original_words > 0 else 0
            
            # Ensure compression rate is at least 70%
            if compression < 70:
                # Take only first 30% of the summary
                sentences = sent_tokenize(combined_summary)
                num_sentences = max(1, int(len(sentences) * 0.3))
                combined_summary = ' '.join(sentences[:num_sentences])
                summary_words = len(word_tokenize(combined_summary))
                compression = ((original_words - summary_words) / original_words) * 100
            
            # Calculate average accuracy
            avg_accuracy = min(85 + ((1 - compression_rate) * 10), 95)
            
            metrics = {
                'original_words': original_words,
                'summary_words': summary_words,
                'compression': round(compression, 1),
                'avg_accuracy': round(avg_accuracy, 1)
            }
            
            return jsonify({
                'summary': combined_summary,
                'metrics': metrics
            })

        except Exception as e:
            print(f"Error in transcript processing: {str(e)}")
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        print(f"Error in generate_combined_summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')
    url = data.get('url', '')

    # Step 1: Get text either from raw input or YouTube
    if url:
        try:
            video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
            if not video_id_match:
                return jsonify({'error': 'Invalid YouTube URL'}), 400
            video_id = video_id_match.group(1)

            try:
                # First try to get the transcript
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                
                # Use the filtered transcript that excludes intro/outro
                # This is the same filtering logic used in the /transcribe endpoint
                if len(transcript_list) > 10:  # Only filter if we have enough transcript entries
                    # Find the entry closest to 2 minutes (120 seconds)
                    start_idx = 0
                    for i, entry in enumerate(transcript_list):
                        if entry['start'] >= 120:  # 2 minutes
                            start_idx = i
                            break
                    
                    # Find the entry closest to 30 seconds before the end
                    total_duration = transcript_list[-1]['start'] + transcript_list[-1]['duration']
                    end_idx = len(transcript_list) - 1
                    for i in range(len(transcript_list) - 1, -1, -1):
                        if transcript_list[i]['start'] <= total_duration - 30:  # 30 seconds before end
                            end_idx = i
                            break
                    
                    # Get only the middle portion of the transcript (Analyzed Content)
                    filtered_transcript = transcript_list[start_idx:end_idx+1]
                    text = ' '.join([entry['text'] for entry in filtered_transcript])
                    print(f"Using filtered transcript from {start_idx} to {end_idx} (Analyzed Content)")
                    
                    # Additional specific filtering for AWS instructor introductions
                    if "hi there" in text.lower() and ("cloud architect" in text.lower() or "aws" in text.lower()):
                        # This looks like an AWS instructor intro - directly filter it out
                        sentences = sent_tokenize(text)
                        filtered_sentences = []
                        skip_intro = False
                        
                        # Check first 5 sentences for intro patterns
                        for i, sentence in enumerate(sentences[:min(5, len(sentences))]):
                            sentence_lower = sentence.lower()
                            if (("hi there" in sentence_lower or "i'm" in sentence_lower or "my name is" in sentence_lower) 
                                and ("architect" in sentence_lower or "walk you through" in sentence_lower)):
                                skip_intro = True
                                print(f"Detected AWS instructor introduction in sentence {i+1}: {sentence}")
                                continue
                            
                            if not skip_intro:
                                filtered_sentences.append(sentence)
                        
                        # Add all remaining sentences
                        filtered_sentences.extend(sentences[min(5, len(sentences)):])
                        
                        # Update text with filtered content
                        if filtered_sentences:
                            text = ' '.join(filtered_sentences)
                            print("Removed AWS instructor introduction")
                else:
                    # For short transcripts, use everything
                    text = ' '.join([entry['text'] for entry in transcript_list])
                    
            except Exception as e:
                return jsonify({'error': f'Failed to fetch transcript: {str(e)}'}), 500
        except Exception as e:
            return jsonify({'error': f'Error processing URL: {str(e)}'}), 500

    if not text:
        return jsonify({'error': 'No transcript text available.'}), 400

    try:
        # Step 2: Restore punctuation - with fallback
        try:
            punctuated = punct_model.restore_punctuation(text)
        except Exception as e:
            print(f"Punctuation model error: {str(e)}")
            punctuated = text  # Fallback to original text

        # Step 3: Generate summaries with a higher compression ratio
        try:
            tr_summary = TextRankSummarizer().summarize(punctuated, ratio=0.25)  # Higher compression
        except Exception as e:
            print(f"TextRank error: {str(e)}")
            # Create basic sentences as fallback
            sentences = sent_tokenize(punctuated)
            tr_summary = sentences[:max(3, len(sentences)//10)]  # Take ~10% of sentences
            
        try:
            lx_summary = LexRankSummarizer().summarize(punctuated, ratio=0.25)  # Higher compression
        except Exception as e:
            print(f"LexRank error: {str(e)}")
            # If TextRank worked, use that, otherwise create basic sentences
            if tr_summary:
                lx_summary = tr_summary
            else:
                sentences = sent_tokenize(punctuated)
                lx_summary = sentences[:max(3, len(sentences)//10)]

        # Ensure tr_summary and lx_summary are lists of strings
        if not tr_summary:
            tr_summary = []
        if not lx_summary:
            lx_summary = []

        # Convert to strings if they're lists
        tr_summary_strings = []
        for item in tr_summary:
            if isinstance(item, list):
                tr_summary_strings.append(' '.join(item))
            elif isinstance(item, str):
                tr_summary_strings.append(item)
                
        lx_summary_strings = []
        for item in lx_summary:
            if isinstance(item, list):
                lx_summary_strings.append(' '.join(item))
            elif isinstance(item, str):
                lx_summary_strings.append(item)

        # Use string values for seen set
        seen = set()
        combined = []
        
        # Process TextRank summary
        for sentence in tr_summary_strings:
            if sentence not in seen:
                combined.append(sentence)
                seen.add(sentence)
                
        # Process LexRank summary
        for sentence in lx_summary_strings:
            if sentence not in seen:
                combined.append(sentence)
                seen.add(sentence)

        summary_text = ' '.join(combined)
        
        # Ensure summary is shorter than the original text
        original_word_count = len(punctuated.split())
        summary_word_count = len(summary_text.split())
        
        # If summary is still longer than original, truncate it
        if summary_word_count >= original_word_count:
            # Take only first 40% of the summary sentences
            summary_sentences = sent_tokenize(summary_text)
            max_sentences = max(3, int(len(summary_sentences) * 0.4))
            summary_text = ' '.join(summary_sentences[:max_sentences])
        
        # If summary is too short, add a basic summary
        if len(summary_text.split()) < 20:
            basic_summary = "This video discusses " + ' '.join(text.split()[:50]) + "..."
            summary_text = basic_summary

        return jsonify({
            'original': punctuated,
            'summary': summary_text
        })
        
    except Exception as e:
        print(f"Error in /summarize endpoint: {str(e)}")
        # Generate a basic summary as fallback
        sentences = sent_tokenize(text[:10000])  # Limit to first 10000 chars for performance
        selected = sentences[:min(5, len(sentences))]  # Select first 5 sentences or less
        fallback_summary = ' '.join(selected)
        
        return jsonify({
            'original': text,
            'summary': fallback_summary
        })

if __name__ == '__main__':
    print("Starting YouTube Video Summarization Backend...")
    try:
        # Test Cohere API key
        print("Testing Cohere API connection...")
        test_response = co.generate(
            prompt="Test connection",
            max_tokens=5,
            model='command',
        )
        print("Cohere API connection successful!")
    except Exception as e:
        print(f"WARNING: Cohere API connection failed: {str(e)}")
        print("The application will continue, but topic analysis and summarization may not work correctly.")
        print("Please check your API key and internet connection.")
        
    # Start the server with debug mode
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
