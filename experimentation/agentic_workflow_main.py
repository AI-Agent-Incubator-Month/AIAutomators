import os
from textblob import TextBlob
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState

# Load environment variables (.env) for LLM config
load_dotenv()

# Configure the Azure OpenAI LLM (adjust env vars as needed)
llm = AzureChatOpenAI(
    model=os.getenv("DEPLOYMENT_NAME"),
    temperature=0,
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version="2023-12-01-preview",
    openai_api_type="azure",
    azure_endpoint=os.getenv("ENDPOINT_URL")
)

# ---- Tool 1: Sentiment Classification ----
def tool_classify_sentiment(text: str):
    """
    Classifies the sentiment of given text as positive, negative, or neutral.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {"sentiment": sentiment, "text": text}

# ---- Tool 2: Audio Transcription ----
def tool_transcribe_audio(path: str):
    """
    Transcribes speech from the given audio file path.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio_listened = recognizer.record(source)
        text = recognizer.recognize_google(audio_listened)
    return text

# ---- Tool 3: Audio Chunking ----
def tool_split_audio_on_silence(path: str, output_dir: str):
    """
    Splits a large audio file into chunks on silence and saves to output_dir.
    Returns: {'chunk_paths': list}
    """
    sound = AudioSegment.from_file(path)
    chunks = split_on_silence(
        sound,
        min_silence_len=500,
        silence_thresh=sound.dBFS - 14,
        keep_silence=500,
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    chunk_paths = []
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(output_dir, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        chunk_paths.append(chunk_filename)
    return {"chunk_paths": chunk_paths}

# ---- Tool 4: Full Pipeline (Split, Transcribe, Sentiment) ----
def tool_transcribe_and_analyze(audio_path: str, output_dir: str):
    """
    Splits audio on silence, transcribes each chunk, and classifies sentiment.
    Combines all transcriptions and finds overall sentiment.
    Returns: {'results': list of dicts with chunk, text, sentiment, overall_sentiment}
    """
    split_result = tool_split_audio_on_silence(audio_path, output_dir)
    chunk_paths = split_result['chunk_paths']
    results = []
    combined_text = ""
    recognizer = sr.Recognizer()
    for chunk in chunk_paths:
        try:
            text = tool_transcribe_audio(chunk)
        except sr.UnknownValueError:
            continue
        sentiment_result = tool_classify_sentiment(text)
        results.append({'text': text, 'sentiment': sentiment_result['sentiment']})
        combined_text += text + " "
    
    # Analyze overall sentiment of combined text
    overall_sentiment_result = tool_classify_sentiment(combined_text.strip())
    results.append({
        "text": combined_text.strip(),
        "sentiment": overall_sentiment_result['sentiment']
    })
    return {"results": results, "overall_sentiment": overall_sentiment_result}

# ---- Register Tools ----
tools = [
    tool_classify_sentiment,
    tool_transcribe_audio,
    tool_split_audio_on_silence,
    tool_transcribe_and_analyze,
]

# ---- System Message ----
sys_msg = SystemMessage(content=(
    "You are an AI assistant that can split audio on silence, transcribe audio, and classify sentiment for each chunk. "
    "You expose tools for these functions. "
    "To process a large audio file: first split it into chunks on silence, then transcribe each chunk, then classify its sentiment. "
    "You can also perform the full pipeline in one step using the transcribe_and_analyze tool."
))

def assistant(state: MessagesState):
    # Bind tools for single-call (not parallel)
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# ---- Build LangGraph Workflow ----
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
react_graph = builder.compile(checkpointer=memory)

def analyze_audio_workflow(audio_path, output_dir):
    input_text = (
        f"Please transcribe the audio at {audio_path}, split on silence, and classify the sentiment of each segment. "
        f"Save the chunks to {output_dir}."
    )
    messages = [HumanMessage(content=input_text)]
    config = {"configurable": {"thread_id": "audio_run_1", "recursion_limit": 50}}
    # config = {"recursion_limit": 50}
    react_graph.invoke({"messages": messages}, config=config)
    messages = react_graph.invoke({"messages": messages}, config=config)
    for m in messages['messages']:
        m.pretty_print()

if __name__ == "__main__":
    # Example usage (change paths as needed)
    audio_path = os.path.join(os.path.dirname(__file__), "test_1.wav")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    analyze_audio_workflow(audio_path, output_dir)