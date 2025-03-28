import React, { useState, useEffect } from "react";
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Tabs,
  Tab,
  Box,
  ThemeProvider,
  createTheme,
  Alert,
  CircularProgress,
} from "@mui/material";
import {
  YouTube as YouTubeIcon,
  Description as DescriptionIcon,
  List as ListIcon,
  QuestionAnswer as QuestionAnswerIcon,
  BarChart as BarChartIcon,
} from "@mui/icons-material";
import NavbarWithRouter from "./Components/Navbar";
import Footer from "./Components/Footer";

const theme = createTheme({
  palette: {
    primary: { main: "#2196f3" },
    secondary: { main: "#f50057" },
  },
});

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          <Typography>{children}</Typography>
        </Box>
      )}
    </div>
  );
}

function App() {
  const [url, setUrl] = useState("");
  const [activeTab, setActiveTab] = useState(0);
  const [videoId, setVideoId] = useState("");
  const [error, setError] = useState("");
  const [transcription, setTranscription] = useState("");
  const [summary, setSummary] = useState("");
  const [aiSummary, setAiSummary] = useState(""); // State for AI summary
  const [quizQuestionsList, setQuizQuestionsList] = useState([]);
  const [answersState, setAnswersState] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");

  useEffect(() => {
    const extractVideoId = (url) => {
      const regExp =
        /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
      const match = url.match(regExp);
      return match && match[2].length === 11 ? match[2] : null;
    };

    const id = extractVideoId(url);
    if (id) {
      setVideoId(id);
      setError("");
    } else if (url) {
      setVideoId("");
      setError("Invalid YouTube URL");
    } else {
      setVideoId("");
      setError("");
    }
  }, [url]);

  const handleTranscript = async () => {
    if (videoId) {
      setLoading(true);
      setLoadingMessage("Fetching transcription and summary...");
      try {
        const response = await fetch("http://localhost:5000/transcribe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: videoId }),
        });
        if (!response.ok) throw new Error("Network response was not ok");

        const result = await response.json();
        setTranscription(result.transcription);
        setSummary(result.summary);
      } catch (error) {
        console.error("Error:", error);
        setError(
          "Failed to fetch transcription and summary. Please try again."
        );
      } finally {
        setLoading(false);
        setLoadingMessage("");
      }
    } else {
      setError("Please enter a valid YouTube URL");
    }
  };

  const handleGenerateSummary = async () => {
    if (videoId) {
      setLoading(true);
      setLoadingMessage("Generating AI summary...");
      try {
        const response = await fetch("http://localhost:5000/generate-summary", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: videoId }), // Ensure video_id key matches backend requirements
        });
        if (!response.ok) throw new Error("Network response was not ok");

        const result = await response.json();
        setAiSummary(result.summary); // Store AI summary in state
      } catch (error) {
        console.error("Error:", error);
        setError("Failed to generate AI summary. Please try again.");
      } finally {
        setLoading(false);
        setLoadingMessage("");
      }
    } else {
      setError("Please enter a valid YouTube URL");
    }
  };

  const handleQuizGeneration = async () => {
    if (videoId) {
      setLoading(true);
      setLoadingMessage("Generating quiz...");
      try {
        const response = await fetch("http://localhost:5000/generate_quiz", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: videoId, num_questions: 5 }),
        });
        if (!response.ok) throw new Error("Network response was not ok");

        const result = await response.json();
        setQuizQuestionsList(result.questions_with_options);
        setAnswersState(
          new Array(result.questions_with_options.length).fill("")
        );
      } catch (error) {
        console.error("Error:", error);
        setError("Failed to generate quiz. Please try again.");
      } finally {
        setLoading(false);
        setLoadingMessage("");
      }
    } else {
      setError("Please enter a valid YouTube URL");
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleAnswerChange = (index, answer) => {
    const newAnswersState = [...answersState];
    newAnswersState[index] = answer;
    setAnswersState(newAnswersState);
  };

  const handleSubmit = () => {
    const answers = quizQuestionsList.map((question, index) => ({
      question: question.question,
      answer: answersState[index],
    }));
    console.log("Submitted Answers:", answers);
  };

  return (
    <>
      <ThemeProvider theme={theme}>
        <NavbarWithRouter />
        <Container maxWidth="md" sx={{ mt: 4 }}>
          <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
            <Typography variant="h4" component="h1" gutterBottom>
              YouTube Video Analyzer
            </Typography>
            <form>
              <TextField
                fullWidth
                label="YouTube URL"
                variant="outlined"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                margin="normal"
                error={!!error}
                helperText={error}
              />
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "16px",
                  justifyContent: "space-around",
                }}
              >
                <Button
                  type="button"
                  variant="contained"
                  color="primary"
                  onClick={handleTranscript}
                  startIcon={<DescriptionIcon />}
                  sx={{ mt: 2, mb: 2 }}
                  disabled={loading}
                >
                  Get Transcription & Summary
                </Button>
                <Button
                  type="button"
                  variant="contained"
                  color="secondary"
                  onClick={handleQuizGeneration}
                  startIcon={<QuestionAnswerIcon />}
                  sx={{ mt: 2, mb: 2 }}
                  disabled={loading}
                >
                  Generate Quiz
                </Button>
                <Button
                  type="button"
                  variant="contained"
                  color="primary"
                  onClick={handleGenerateSummary}
                  startIcon={<ListIcon />}
                  sx={{ mt: 2, mb: 2 }}
                  disabled={loading}
                >
                  Generate Summary with AI
                </Button>
              </div>
            </form>
            {videoId && (
              <Box sx={{ position: "relative", paddingTop: "56.25%", mb: 2 }}>
                <iframe
                  src={`https://www.youtube.com/embed/${videoId}`}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                  }}
                />
              </Box>
            )}
            {loading ? (
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  mt: 2,
                }}
              >
                <CircularProgress />
                <Typography variant="body1" sx={{ mt: 2 }}>
                  {loadingMessage}
                </Typography>
              </Box>
            ) : (
              <>
                {error && (
                  <Alert severity="error" sx={{ mt: 2, mb: 2 }}>
                    {error}
                  </Alert>
                )}
                <Box sx={{ borderBottom: 1, borderColor: "divider", mt: 4 }}>
                  <Tabs
                    value={activeTab}
                    onChange={handleTabChange}
                    aria-label="analysis tabs"
                  >
                    <Tab icon={<DescriptionIcon />} label="Transcription" />
                    <Tab icon={<ListIcon />} label="Summary" />
                    <Tab icon={<ListIcon />} label="AI Summary" />
                    <Tab icon={<QuestionAnswerIcon />} label="Quiz" />
                    <Tab icon={<BarChartIcon />} label="Analysis" />
                  </Tabs>
                </Box>
                <TabPanel value={activeTab} index={0}>
                  {transcription ? (
                    <div dangerouslySetInnerHTML={{ __html: transcription }} />
                  ) : (
                    "Transcription content will appear here"
                  )}
                </TabPanel>
                <TabPanel value={activeTab} index={1}>
                  {summary ? (
                    <div dangerouslySetInnerHTML={{ __html: summary }} />
                  ) : (
                    "Summary content will appear here"
                  )}
                </TabPanel>
                <TabPanel value={activeTab} index={2}>
                  {aiSummary ? (
                    <div>
                      {aiSummary.includes("-") ? (
                        <ul>
                          {aiSummary
                          
                          .split(/(?<!\d)-|-(?!\d)/) // Split on dashes that are not between numbers
                            .filter((item) => item.trim() !== "")
                            .map((item, idx) => (
                              <li key={idx}>{item.trim()}</li>
                            ))}
                        </ul>
                      ) : (
                        <div dangerouslySetInnerHTML={{ __html: aiSummary }} />
                      )}
                    </div>
                  ) : (
                    "Summary content will appear here"
                  )}
                </TabPanel>

                <TabPanel value={activeTab} index={3}>
                  {quizQuestionsList.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="h6">Generated Questions:</Typography>
                      {quizQuestionsList.map((questionData, index) => (
                        <Box key={index} sx={{ mb: 2 }}>
                          <Typography variant="body1">
                            {index + 1}. {questionData.question}
                          </Typography>
                          {questionData.options.map((option, optionIndex) => (
                            <Box
                              key={optionIndex}
                              sx={{ display: "flex", alignItems: "center" }}
                            >
                              <input
                                type="radio"
                                name={`question-${index}`}
                                value={option}
                                checked={answersState[index] === option}
                                onChange={() =>
                                  handleAnswerChange(index, option)
                                }
                              />
                              <Typography variant="body2">{option}</Typography>
                            </Box>
                          ))}
                        </Box>
                      ))}
                      <Button
                        variant="contained"
                        color="primary"
                        onClick={handleSubmit}
                      >
                        Submit
                      </Button>
                    </Box>
                  )}
                </TabPanel>
                <TabPanel value={activeTab} index={4}>
                  Analysis content will appear here
                </TabPanel>
              </>
            )}
          </Paper>
        </Container>
        <Footer />
      </ThemeProvider>
    </>
  );
}

export default App;
