from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
import cohere
import re
import random

# Initialize Cohere client
API_KEY = "IKI636LmJxZLpIJJWOXQMlS5dSBpshN0odoSyTBM"  # Replace with your API key if needed
co = cohere.Client(API_KEY)

# Flask setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/transcript/<video_id>')
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([item['text'] for item in transcript])
        return jsonify({'transcript': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

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
        
        # Generate summary using Cohere
        summary_prompt = f"""Please provide a comprehensive summary of this video transcript. 
        Include:
        1. Main points and key takeaways
        2. Important details and examples
        3. Any conclusions or recommendations
        
        Transcript: {text[:4000]}"""
        
        summary_response = co.generate(
            prompt=summary_prompt,
            max_tokens=500,
            temperature=0.3,
            model='command',
            k=0
        )
        
        summary = summary_response.generations[0].text.strip()
        
        # Generate topics
        topics_prompt = f"""List the main topics covered in this video transcript. 
        Format as a bullet-point list.
        
        Transcript: {text[:4000]}"""
        
        topics_response = co.generate(
            prompt=topics_prompt,
            max_tokens=200,
            temperature=0.2,
            model='command',
            k=0
        )
        
        topics = topics_response.generations[0].text.strip()
        
        return jsonify({
            'summary': summary,
            'topics': topics
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/enhanced_quiz', methods=['POST'])
def enhanced_quiz():
    data = request.get_json()
    video_id = data.get('video_id')
    num_questions = data.get('num_questions', 5)  # Default to 5 questions

    if not video_id:
        return jsonify({'error': 'Missing video ID'}), 400

    try:
        # Get video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([item['text'] for item in transcript])
        print(f"Transcript length: {len(text)} characters")
        
        # First, identify key concepts from the transcript
        concept_prompt = f"""Extract 5 specific factual concepts or key points from this video transcript. 
        For each concept, identify:
        1. The key concept/fact
        2. A brief explanation (1-2 sentences)
        
        Format as:
        Concept: [specific concept/fact]
        Explanation: [brief explanation]
        
        Choose concepts that are specific to this content and can be tested in a quiz.
        Transcript: {text[:4000]}"""
        
        try:
            concept_response = co.generate(
                prompt=concept_prompt,
                max_tokens=600,
                temperature=0.2,
                model='command',
                k=0
            )
            concepts_text = concept_response.generations[0].text
            print(f"Generated concepts: {concepts_text}")
            
            # Extract concept blocks
            concept_blocks = re.split(r'\n\s*\n', concepts_text)
            key_concepts = []
            
            for block in concept_blocks:
                if not block.strip():
                    continue
                
                concept_match = re.search(r'Concept:\s*(.*?)(?:\n|$)', block)
                explanation_match = re.search(r'Explanation:\s*(.*?)(?:\n|$)', block)
                
                if concept_match:
                    concept = {
                        'concept': concept_match.group(1).strip(),
                        'explanation': explanation_match.group(1).strip() if explanation_match else ""
                    }
                    key_concepts.append(concept)
            
            # Generate questions based on identified concepts
            questions_with_options = []
            
            for concept in key_concepts:
                try:
                    # Create a question-specific prompt for better quality
                    question_prompt = f"""Create a multiple-choice question about this concept from the video:
                    
                    Concept: {concept['concept']}
                    Explanation: {concept['explanation']}
                    
                    Create:
                    1. A clear, specific question testing understanding of this concept
                    2. 4 answer options, with exactly ONE correct answer
                    3. Make sure all options are plausible and related to the content
                    
                    Format as:
                    Question: [question text]
                    A. [correct option]
                    B. [incorrect option 1]
                    C. [incorrect option 2]
                    D. [incorrect option 3]
                    
                    Here's additional context from the transcript: {text[:1500]}"""
                    
                    question_response = co.generate(
                        prompt=question_prompt,
                        max_tokens=400,
                        temperature=0.4,
                        model='command',
                        k=0
                    )
                    
                    generated_question = question_response.generations[0].text.strip()
                    print(f"Generated question: {generated_question}")
                    
                    # Parse the question and options
                    question_match = re.search(r'Question:\s*(.*?)(?:\n|$)', generated_question)
                    
                    if not question_match:
                        continue
                        
                    question_text = question_match.group(1).strip()
                    options = []
                    
                    # Extract options (A is always correct in our format)
                    option_matches = re.findall(r'([A-D])\.\s*(.*?)(?:\n|$)', generated_question)
                    
                    if len(option_matches) >= 4:
                        # Extract all options and their letter labels
                        letter_options = {letter: text.strip() for letter, text in option_matches}
                        
                        # A is the correct answer in our format
                        correct_answer = letter_options.get('A', '')
                        
                        # Put correct answer first, followed by wrong answers
                        if correct_answer:
                            options = [correct_answer]  # Correct answer first
                            # Add other options
                            for letter in ['B', 'C', 'D']:
                                if letter in letter_options:
                                    options.append(letter_options[letter])
                        
                        # Ensure we have at least 3 options total
                        if len(options) >= 3:
                            questions_with_options.append({
                                'question': question_text,
                                'options': options[:4]  # Limit to 4 options
                            })
                
                except Exception as question_error:
                    print(f"Error generating question for concept {concept['concept']}: {str(question_error)}")
                    continue
                    
            # If we have fewer questions than requested, generate additional questions
            if len(questions_with_options) < num_questions:
                remaining = num_questions - len(questions_with_options)
                
                # Generate general questions from the transcript
                general_prompt = f"""Create {remaining} multiple-choice questions from this video transcript.
                
                For each question:
                1. Ask about specific content/facts from the transcript
                2. Create 4 answer options with ONLY ONE correct answer
                3. Put the correct answer as option A, followed by 3 incorrect options B, C, and D
                
                Format each question as:
                Question: [clear, specific question]
                A. [correct option]
                B. [incorrect option 1]
                C. [incorrect option 2] 
                D. [incorrect option 3]
                
                Here's the transcript: {text[:3000]}"""
                
                try:
                    general_response = co.generate(
                        prompt=general_prompt,
                        max_tokens=800,
                        temperature=0.4,
                        model='command',
                        k=0
                    )
                    
                    general_questions = general_response.generations[0].text
                    
                    # Split into question blocks
                    question_blocks = re.split(r'\n\s*\n', general_questions)
                    
                    for block in question_blocks:
                        if not "Question:" in block:
                            continue
                            
                        question_match = re.search(r'Question:\s*(.*?)(?:\n|$)', block)
                        if not question_match:
                            continue
                            
                        question_text = question_match.group(1).strip()
                        options = []
                        
                        # Extract options
                        option_matches = re.findall(r'([A-D])\.\s*(.*?)(?:\n|$)', block)
                        
                        if len(option_matches) >= 3:
                            # In this format, option A is always correct
                            correct_option = next((text for letter, text in option_matches if letter == 'A'), None)
                            
                            if correct_option:
                                options = [correct_option.strip()]  # Correct answer first
                                
                                # Add incorrect options
                                for letter, text in option_matches:
                                    if letter != 'A':
                                        options.append(text.strip())
                                        
                                # Add to question list if we have enough options
                                if len(options) >= 3 and len(questions_with_options) < num_questions:
                                    questions_with_options.append({
                                        'question': question_text,
                                        'options': options[:4]  # Limit to 4 options
                                    })
                
                except Exception as general_error:
                    print(f"Error generating additional questions: {str(general_error)}")
            
            # Fill with default questions if we don't have enough
            default_questions = [
                {
                    'question': 'What is the main topic of this video?',
                    'options': [
                        'The content presented in the video',
                        'A topic not related to the video',
                        'The background music in the video',
                        'The video editing techniques'
                    ]
                },
                {
                    'question': 'What would be the best title for this video?',
                    'options': [
                        'A title related to the main topic',
                        'A clickbait title',
                        'A misleading title',
                        'A generic video title'
                    ]
                }
            ]
            
            # Add default questions if needed
            while len(questions_with_options) < num_questions and len(default_questions) > 0:
                questions_with_options.append(default_questions.pop(0))
            
            # Shuffle the options (except the first which is correct)
            for question in questions_with_options:
                if len(question['options']) > 1:
                    correct = question['options'][0]
                    other_options = question['options'][1:]
                    random.shuffle(other_options)
                    question['options'] = [correct] + other_options
            
            # Return the final set of questions
            print(f"Final enhanced quiz has {len(questions_with_options)} questions")
            return jsonify({'questions_with_options': questions_with_options})
            
        except Exception as e:
            print(f"Error in concept extraction: {str(e)}")
            return jsonify({'error': f'Failed to generate quiz: {str(e)}'}), 500
            
    except Exception as e:
        error_message = str(e)
        print(f"Error in enhanced_quiz: {error_message}")
        
        if "Subtitles are disabled" in error_message:
            return jsonify({'error': 'This video has disabled subtitles/transcription'}), 400
        elif "Video unavailable" in error_message:
            return jsonify({'error': 'This video is unavailable or private'}), 400
        else:
            return jsonify({'error': f'Failed to generate quiz: {error_message}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 
