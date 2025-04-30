import streamlit as st
import requests

st.set_page_config(
    page_title="Audio Sentiment Analysis",
    page_icon="🔊",
    layout="centered"
)

st.title("🔊 Call Transcription Agent")
st.write(
    "Upload a WAV audio file. The app will split it on silence, transcribe each chunk, and classify the sentiment."
)

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Analyze Audio"):
        with st.spinner("Analyzing audio..."):
            files = {"audio_file": (uploaded_file.name, uploaded_file, "audio/wav")}
            try:
                # Make request to FastAPI backend
                response = requests.post(
                    "http://127.0.0.1:8007/analyze-audio/",
                    files=files
                )
                if response.status_code == 200:
                    result = response.json()
                    results = result.get("results", [])
                    
                    st.subheader("Per-Segment Sentiment")
                    for i, r in enumerate(results[:-1]):  # last one is overall
                        st.markdown(
                            f"**Segment {i+1}:**\n\n"
                            f"- Text: `{r['text']}`\n"
                            f"- Sentiment: **{r['sentiment']}**"
                        )
                    
                    st.subheader("Overall Sentiment")
                    overall = results[-1]
                    st.markdown(
                        f"**Combined Text:** `{overall['text']}`\n\n"
                        f"**Overall Sentiment:** {overall['sentiment']}"
                    )
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# streamlit run app_streamlit.py (To run the script)