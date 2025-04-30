# Team Name- **AI Automators**
# Team Members-
**Sri Harsha Juttu**
**Trikambhai, Chauhan Maheshbhai**
**K S, Sandhya**


## Overview
This repository contains the code and documentation for our innovative solution: **Call Transcription Agent**. Our project aims to provide an audio sentiment analysis pipeline that enables users to upload a WAV audio file and receive an interactive, segment-wise transcription and sentiment analysis via a web interface.

## Explanation
Our project leverages **FastAPI**, **Streamlit**, **LangGraph**, **LangChain**, **Azure OpenAI**, **pydub**, **SpeechRecognition**, and **TextBlob** to create a modular, agentic workflow for audio file analysis. The core functionality includes:
- **Audio Splitting:** Automatically splits uploaded audio files into segments based on silence detection using `pydub`.
- **Transcription:** Transcribes each audio segment using Google Speech Recognition.
- **Sentiment Analysis:** Analyzes the sentiment (positive, negative, or neutral) of each transcribed segment using TextBlob and presents the results clearly in an interactive web UI.

## Intent
The primary intent of our project is to simplify and automate the process of extracting actionable insights from audio calls, such as customer service recordings or interviews. We aim to empower businesses and analysts to quickly understand both the content and sentiment of conversations, enabling faster response, feedback, and improvement cycles.

## Use Case
Assigned Industry Use Case:
**Contact Center Analytics / Customer Experience Management**
 
Our solution is designed to be used by customer support teams, business analysts, and operation managers in industries such as telecommunications, banking, and retail. It can be applied in scenarios such as:
- **Scenario 1:** Analyzing customer sentiment in support calls to identify at-risk customers or agents who need coaching.
- **Scenario 2:** Extracting and summarizing feedback from product review hotlines or interviews.
- **Scenario 3:** Monitoring compliance and emotion in sales or collections calls.

## Contributors
This project was developed by a dedicated team of contributors:
- **Sri Harsha Juttu**: Led research, backend development, frontend implementation, API design, system integration, agentic AI workflow creation, and comprehensive testing.
- **Trikambhai, Chauhan Maheshbhai**: Research & Architecture design
- **K S, Sandhya**: Research, Documentation, Audio convertions & Testing

## Images
![Screenshot 1](experimentation\UI_Design_1.png)
![Screenshot 2](experimentation\UI_design_2.png)
![Screenshot 3](experimentation\UI_design_3.png)

## Implementation
Our solution is architected as follows:

- **Agentic Workflow (LangGraph & LangChain, `agentic_workflow_main.py`)**  
  The workflow is built on LangGraph and orchestrated by LangChain with Azure OpenAI. Four modular tools are defined:  
    - `tool_classify_sentiment(text: str)`: Classifies sentiment using TextBlob.
    - `tool_transcribe_audio(path: str)`: Transcribes using Google Speech Recognition.
    - `tool_split_audio_on_silence(path: str, output_dir: str)`: Splits audio on silence via pydub.
    - `tool_transcribe_and_analyze(audio_path: str, output_dir: str)`: Full pipeline for batch processing.

  Sensitive API keys and endpoints are managed via a `.env` file.

- **Backend (FastAPI, `app.py`)**  
  The backend exposes a `/analyze-audio/` endpoint that accepts a `.wav` file, processes it using the agentic workflow, and returns segment-wise transcriptions and sentiment as JSON. It features robust error handling for file type, processing issues, and missing dependencies.

- **Frontend (Streamlit, `app_streamlit.py`)**  
  The frontend lets users upload `.wav` files, calls the backend API, and interactively displays each segment’s transcription and sentiment. Errors from the backend are shown directly to the user.

- **Dependencies & Environment**  
  All dependencies are specified in the [Requirements](#requirements) section. `ffmpeg` is required for audio handling by pydub.

- **Running the Application**  
  1. Start the FastAPI backend:  
    `python api.py`
  2. Start the Streamlit frontend:  
    `streamlit run app_streamlit.py`
  3. Access the Streamlit UI in your browser and upload a `.wav` file for analysis.

## Additional Information
- **Customization:** Silence detection thresholds and sentiment analysis logic can be tuned for specific use cases.
- **Known Issues:** Only `.wav` files are supported; large files may require longer processing times.
- **Future Plans:** We plan to add speaker diarization, diarized sentiment, multi-language support, and richer analytics dashboards.
- **Acknowledgements:**  
  - [FastAPI](https://fastapi.tiangolo.com/)
  - [Streamlit](https://streamlit.io/)
  - [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
  - [TextBlob](https://textblob.readthedocs.io/en/dev/)
  - [LangChain](https://python.langchain.com/)
  - [LangGraph](https://github.com/langchain-ai/langgraph)
