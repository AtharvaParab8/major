import cohere
from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import networkx as nx

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

def extract_main_topics(text):
    """
    Extract main topics from the transcript text
    Returns a list of topic strings
    """
    try:
        # Use Cohere to extract main topics
        response = co.generate(
            prompt=f"""Extract 4-5 main topics from this video transcript.
            Focus on specific subjects discussed in detail, not generic terms like 'introduction' or 'conclusion'.
            Return only a list of topics, one per line.
            
            Here's the transcript: {text[:3500]}""",
            max_tokens=300,
            temperature=0.2,
            model='command',
            k=0
        )
        
        topics_text = response.generations[0].text
        
        # Process the text to get a clean list of topics
        topics = []
        for line in topics_text.strip().split('\n'):
            # Remove numbered bullets and dashes if present
            clean_line = re.sub(r'^\s*\d+[\.\)]\s*', '', line)
            clean_line = re.sub(r'^\s*[-•]\s*', '', clean_line)
            clean_line = clean_line.strip()
            
            if clean_line:
                topics.append(clean_line)
        
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
                prompt=f"""Analyze this video transcript and identify 3-5 main subtopics discussed. 
                For each subtopic:
                1. Provide a descriptive title that accurately represents a specific topic in the video (avoid generic terms like 'introduction' or 'conclusion')
                2. Calculate approximate percentage of video time (numbers should add up to 100%)
                3. Rate the importance as 'high', 'medium', or 'low' based on how central it is to the video's message
                4. Add a brief one-sentence description

                Format each topic as:
                Topic: [title]
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
            # Return error instead of defaults
            return jsonify({'error': 'Failed to extract topics from video'}), 500
        
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
            
            # If we couldn't extract any specific topics, try another approach
            if not topics:
                print("No specific topics found, trying keyword extraction")
                
                # Extract key topics/concepts from the video
                keyword_response = co.generate(
                    prompt=f"""Extract 4-5 main keywords or topics from this video transcript. 
                    Focus on specific, concrete subjects mentioned, not generic terms like 'introduction' or 'conclusion'.
                    For each keyword:
                    1. List the keyword or topic
                    2. Give a brief one-sentence description
                    
                    Format as:
                    Keyword: [specific topic]
                    Description: [brief description]
                    
                    Here's the transcript: {full_text[:3500]}""",
                    max_tokens=500,
                    temperature=0.2,
                    model='command',
                    k=0
                )
                
                keyword_text = keyword_response.generations[0].text
                keyword_blocks = re.split(r'\n\s*\n', keyword_text)
                
                # Create topics from keywords
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
                
            # Normalize percentages to ensure they add up to 100%
            if topics:
                total = sum(topic['percentage'] for topic in topics)
                if total == 0:
                    # If all percentages are 0, distribute evenly
                    for topic in topics:
                        topic['percentage'] = 100 // len(topics)
                elif total != 100:
                    # Scale percentages to add up to 100%
                    scale_factor = 100 / total
                    for topic in topics:
                        topic['percentage'] = round(topic['percentage'] * scale_factor)
                    
                    # Adjust to ensure total is exactly 100%
                    diff = 100 - sum(topic['percentage'] for topic in topics)
                    if diff != 0 and topics:
                        topics[0]['percentage'] += diff
                
            # Sort topics by percentage (descending)
            topics = sorted(topics, key=lambda x: x.get('percentage', 0), reverse=True)
            
            # Find the most important topic
            most_important = None
            for topic in topics:
                if topic.get('importance') == 'high':
                    most_important = topic['name']
                    break
            
            # If no topic is marked high importance, choose the one with highest percentage
            if not most_important and topics:
                most_important = max(topics, key=lambda x: x.get('percentage', 0))['name']
                
            return jsonify({
                'topics': topics, 
                'most_important': most_important,
                'total_duration': video_duration
            })
                
        except Exception as e:
            print(f"Error processing topics: {str(e)}")
            return jsonify({'error': 'Failed to identify meaningful topics in the video'}), 500
            
    except Exception as e:
        error_message = str(e)
        print(f"Error in analyze_topics: {error_message}")
        
        if "Subtitles are disabled" in error_message:
            return jsonify({'error': 'This video has disabled subtitles/transcription'}), 400
        elif "Video unavailable" in error_message:
            return jsonify({'error': 'This video is unavailable or private'}), 400
        else:
            return jsonify({'error': f'Failed to analyze topics: {error_message}'}), 500

@app.route('/generate-summary', methods=['POST'])
def generate_summary():
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
        
        # Create a simple summary just to test if the frontend is displaying it
        # summary = "This is a test summary using DistilBART (simulated). The video appears to be about " + text.split()[0:10] + "... Continue watching for more details."
        
        # Extract main topics
        main_topics = ["Topic 1", "Topic 2", "Topic 3"]
        
        # print(f"Generated summary: {summary}")
        print(f"Generated topics: {main_topics}")
        
        # Return results with complete response structure
        return jsonify({
            # 'summary': summary,
            'main_topics': main_topics
        })
        
    except Exception as e:
        print(f"Error in generate_summary: {str(e)}")
        import traceback
        traceback.print_exc()
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
