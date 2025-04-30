import os
import tempfile
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from agentic_workflow_main import react_graph
import uvicorn

app = FastAPI(
    title="Audio Sentiment Analysis API",
    description="Upload a WAV audio file. The API splits on silence, transcribes each chunk, and classifies sentiment.",
    version="1.4"
)

@app.post("/analyze-audio/")
async def analyze_audio(
    audio_file: UploadFile = File(...)
):
    if not audio_file.filename.lower().endswith(".wav"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only WAV audio files are supported."}
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, audio_file.filename)

        # Save uploaded file to temp directory
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())

        output_dir = os.path.join(tmpdir, "chunks")

        input_text = (
            f"Please transcribe the audio at {audio_path}, split on silence, and classify the sentiment of each segment. "
            f"Save the chunks to {output_dir}. "
            "Format each segment as: **Text:** \"<text>\" **Sentiment:** <sentiment>"
        )

        try:
            messages = [HumanMessage(content=input_text)]
            config = {"configurable": {"thread_id": "audio_run_1"}}
            response = react_graph.invoke({"messages": messages}, config=config)
            extracted_data = response['messages'][-1].content
            print(f"Extracted data: {extracted_data}")

            # Pattern to extract text and sentiment
            pattern = re.compile(
                r'\*\*Text:\*\*\s*"(.*?)"\s*\*\*Sentiment:\*\*\s*(\w+)',
                re.DOTALL
            )

            results = []
            for match in pattern.finditer(extracted_data):
                text = match.group(1).strip().replace('\n', ' ')
                sentiment = match.group(2).strip()
                results.append({"text": text, "sentiment": sentiment})
            
            if not results:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No segments could be processed from the audio"}
                )

            # Combine all extracted text for overall sentiment analysis
            combined_text = " ".join([item["text"] for item in results])

            # Prompt for structured sentiment response
            overall_sentiment_input = (
                f"Analyze the overall sentiment of this text: \"{combined_text}\". "
                "Respond with exactly: **Sentiment:** <sentiment>"
            )
            
            overall_messages = [HumanMessage(content=overall_sentiment_input)]
            overall_response = react_graph.invoke({"messages": overall_messages}, config=config)
            overall_content = overall_response['messages'][-1].content.strip()
            
            # Extract sentiment using same pattern
            overall_match = pattern.search(overall_content)
            if overall_match:
                overall_sentiment = overall_match.group(2).strip()
            else:
                # Fallback if structured format not found
                sentiment_words = re.findall(r'positive|negative|neutral', overall_content.lower())
                overall_sentiment = sentiment_words[0].capitalize() if sentiment_words else "Unknown"

            results.append({"text": combined_text, "sentiment": overall_sentiment})

            return JSONResponse(content={"results": results})

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"An error occurred: {str(e)}"}
            )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8007)
