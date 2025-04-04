# YouTube Video Summarization with Enhanced Quiz

This application provides tools to analyze, summarize, and quiz users on YouTube video content.

## Features

- Transcribe YouTube videos
- Generate AI-powered summaries of video content
- Identify key topics in videos
- Generate topic-specific summaries
- Create content-specific quizzes with visual feedback
- Terminal command quizzes

## Setup Instructions

### Backend Setup

1. Navigate to the Backend directory:

   ```
   cd Backend
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Start the enhanced quiz backend server:
   ```
   python app1.py
   ```
   This will run the server on port 5001.

### Frontend Setup

1. Navigate to the Frontend directory:

   ```
   cd Frontend
   ```

2. Install the required dependencies:

   ```
   npm install
   ```

3. Start the frontend development server:
   ```
   npm start
   ```
   This will start the React app on http://localhost:3000.

## Using the Application

1. Enter a YouTube URL in the input field.
2. Click "GET TRANSCRIPTION & SUMMARY" to fetch the video transcript.
3. Use "GENERATE AI SUMMARY" to create a comprehensive summary of the video.
4. Click "ANALYZE TOPICS" to identify key topics in the video.
5. Use "GENERATE QUIZ" to create a content-specific quiz with multiple-choice questions.
6. Try "TERMINAL QUIZ" for command-line related questions.

## Enhanced Quiz Features

The enhanced quiz feature:

- Generates questions specifically related to the video content
- Provides color-coded feedback for correct and incorrect answers
- Shows a score with visual feedback
- Allows multiple attempts

## Troubleshooting

- If the enhanced quiz server (port 5001) fails, the app will automatically fall back to the original quiz generation method.
- Ensure both frontend and backend servers are running simultaneously.
- Check that the Cohere API key is valid if you experience issues with summary or quiz generation.
