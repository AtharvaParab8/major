import React, { useState } from 'react';
import './App.css';
import { FaYoutube, FaSpinner } from 'react-icons/fa';

function App() {
  const [url, setUrl] = useState('');
  const [videoId, setVideoId] = useState('');
  const [transcript, setTranscript] = useState('');
  const [summary, setSummary] = useState('');
  const [topics, setTopics] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showQuiz, setShowQuiz] = useState(false);
  const [questions, setQuestions] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [score, setScore] = useState(0);
  const [showScore, setShowScore] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [isCorrect, setIsCorrect] = useState(false);

  const extractVideoId = (url) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setTranscript('');
    setSummary('');
    setTopics('');
    setShowQuiz(false);
    setQuestions([]);
    setCurrentQuestion(0);
    setScore(0);
    setShowScore(false);
    setSelectedAnswer(null);
    setShowFeedback(false);
    setIsCorrect(false);

    const id = extractVideoId(url);
    if (!id) {
      setError('Invalid YouTube URL');
      setLoading(false);
      return;
    }

    setVideoId(id);

    try {
      // Get transcript
      const transcriptResponse = await fetch(`http://localhost:5001/transcript/${id}`);
      const transcriptData = await transcriptResponse.json();
      
      if (transcriptData.error) {
        setError(transcriptData.error);
        setLoading(false);
        return;
      }

      setTranscript(transcriptData.transcript);

      // Generate summary
      const summaryResponse = await fetch('http://localhost:5001/generate_summary', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ video_id: id }),
      });
      
      const summaryData = await summaryResponse.json();
      
      if (summaryData.error) {
        setError(summaryData.error);
        setLoading(false);
        return;
      }

      setSummary(summaryData.summary);
      setTopics(summaryData.topics);

      // Generate quiz based on the summary
      const quizResponse = await fetch('http://localhost:5001/enhanced_quiz', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          video_id: id,
          num_questions: 5
        }),
      });
      
      const quizData = await quizResponse.json();
      
      if (quizData.error) {
        setError(quizData.error);
        setLoading(false);
        return;
      }

      setQuestions(quizData.questions_with_options);
      setShowQuiz(true);
      setLoading(false);
    } catch (err) {
      setError('Failed to process video');
      setLoading(false);
    }
  };

  const handleAnswerSelect = (answer) => {
    if (selectedAnswer !== null) return; // Prevent multiple selections
    
    setSelectedAnswer(answer);
    const correct = answer === questions[currentQuestion].options[0];
    setIsCorrect(correct);
    
    if (correct) {
      setScore(score + 1);
    }
    
    setShowFeedback(true);
  };

  const handleNextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedAnswer(null);
      setShowFeedback(false);
    } else {
      setShowScore(true);
    }
  };

  const handleRestartQuiz = () => {
    setCurrentQuestion(0);
    setScore(0);
    setShowScore(false);
    setSelectedAnswer(null);
    setShowFeedback(false);
    setIsCorrect(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <FaYoutube className="youtube-icon" />
        <h1>YouTube Video Summarizer</h1>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="url-form">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Enter YouTube URL"
            className="url-input"
            disabled={loading}
          />
          <button type="submit" className="submit-button" disabled={loading}>
            {loading ? <FaSpinner className="spinner" /> : 'Analyze'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {transcript && (
          <div className="content-section">
            <h2>Transcript</h2>
            <div className="transcript-content">{transcript}</div>
          </div>
        )}

        {summary && (
          <div className="content-section">
            <h2>AI Summary</h2>
            <div className="summary-content">{summary}</div>
          </div>
        )}

        {topics && (
          <div className="content-section">
            <h2>Topics Covered</h2>
            <div className="topics-content">{topics}</div>
          </div>
        )}

        {showQuiz && !showScore && (
          <div className="quiz-section">
            <h2>Quiz</h2>
            <div className="quiz-container">
              <div className="question-counter">
                Question {currentQuestion + 1} of {questions.length}
              </div>
              <div className="question">
                {questions[currentQuestion].question}
              </div>
              <div className="options">
                {questions[currentQuestion].options.map((option, index) => (
                  <button
                    key={index}
                    className={`option-button ${
                      selectedAnswer === option
                        ? isCorrect
                          ? 'correct'
                          : 'incorrect'
                        : ''
                    }`}
                    onClick={() => handleAnswerSelect(option)}
                    disabled={selectedAnswer !== null}
                  >
                    {option}
                  </button>
                ))}
              </div>
              {showFeedback && (
                <div className="feedback">
                  {isCorrect ? (
                    <div className="correct-feedback">Correct!</div>
                  ) : (
                    <div className="incorrect-feedback">
                      Incorrect. The correct answer is: {questions[currentQuestion].options[0]}
                    </div>
                  )}
                  <button
                    className="next-button"
                    onClick={handleNextQuestion}
                  >
                    {currentQuestion === questions.length - 1 ? 'Finish Quiz' : 'Next Question'}
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {showScore && (
          <div className="score-section">
            <h2>Quiz Complete!</h2>
            <div className="score">
              Your score: {score} out of {questions.length}
            </div>
            <button className="restart-button" onClick={handleRestartQuiz}>
              Try Again
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 