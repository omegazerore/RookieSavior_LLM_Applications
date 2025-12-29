import base64
import os
import requests
from io import BytesIO

import streamlit as st
from PIL import Image

# The parent directory where you save your content
# When you run a Streamlit app, Streamlit changes the working directory to the script’s directory
dir_ = "DEMO"


# --- CONFIG ---
st.set_page_config(page_title="AI Story Chatbot", layout="wide")

# --- INITIALIZE SESSION STATE ---
if "chapters" not in st.session_state:
    st.session_state["chapters"] = []  # stores text
# if "images" not in st.session_state:
#     st.session_state["images"] = []  # stores base64-encoded images
if "story" not in st.session_state:
    st.session_state["story"] = None
if "image" not in st.session_state:
    st.session_state["image"] = None
if "chapter_index" not in st.session_state:
    st.session_state["chapter_index"] = 1
if "nl_prompt" not in st.session_state:
    st.session_state["nl_prompt"] = None
if "nl_prompt_temp" not in st.session_state:
    st.session_state["nl_prompt_temp"] = None


# --- FUNCTIONS ---

def generate_story(prompt: str):
    """Simulate or request story generation."""
    # Replace this with your API call to generate story text
    response = requests.post(
        "http://127.0.0.1:5000/story",
        json={
            "scratch": prompt,                 # user input or latest part of story
            "context": "\n\n".join(st.session_state.get("chapters", [])),  # optional context
        },
        timeout=60
    )

    if response.status_code == 200:
        data = response.json()
        return data.get("output", "")
    else:
        st.error(f"Story generation failed: {response.text}")
        return ""


def generate_image():
    """Simulate or request image generation."""
    
    if st.session_state["nl_prompt"]:
        story = st.session_state["story"] + f"\nPrevious image description:\n\n{st.session_state['nl_prompt']}"
    
    try:
        response = requests.post(
            "http://127.0.0.1:5000/image",
            json={
                "story": st.session_state["story"],
                "image_io": [] #st.session_state["images"]
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("output", "")
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return ""


def generate_audio():
    """Simulate or request image generation."""
    try:
        response = requests.post(
            "http://127.0.0.1:5000/tts",
            json={
                "input": st.session_state["story"]
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("output", "")
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return ""


def display_story(col):
    # STORY PANE
    with col:
        st.header("📖 Story")
        if st.session_state["story"]:
            whole_stories = st.session_state["chapters"] + [st.session_state["story"]] 
            whole_story = "\n----------------------------\n".join(whole_stories)
            st.text_area("Generated Story", whole_story, height=400)
        elif st.session_state["chapters"]:
            whole_stories = st.session_state["chapters"]
            whole_story = "\n----------------------------\n".join(whole_stories)
            st.text_area("Generated Story", whole_story, height=400)
        else:
            st.info("No story generated yet.")


def display_image(col):
    # IMAGE PANE
    with col:
        st.header("🖼️ Image")
        if st.session_state["image"]:
            # img_base64 = st.session_state["images"][-1]
            img_data = base64.b64decode(st.session_state["image"])
            st.image(img_data, caption="Generated Image", use_container_width=True)
        else:
            st.info("No image generated yet.")


# --- LAYOUT ---

col_story, col_image, col_input = st.columns([3, 2, 2])


# PROMPT PANE
with col_input:
    st.header("💬 User Prompt")
    user_prompt = st.text_area("Enter your next chapter prompt", "")
    if st.button("Generate Next Chapter"):

        print(f"User prompt: {user_prompt}")
        
        if user_prompt.strip():
            story = generate_story(user_prompt)
            st.session_state["story"] = story

            image_output = generate_image()
            
            image_base64 = image_output['image_base64']
            nl_prompt = image_output['nl_prompt']
            st.session_state['image'] = image_base64
            st.session_state['nl_prompt_temp'] = nl_prompt

        else:
            st.warning("Please enter a prompt before generating.")

display_story(col_story)
display_image(col_image)

if st.button("Accept Result"):
    st.session_state["chapters"].append(st.session_state['story'])
    # st.session_state["images"].append(st.session_state['image'])
    st.session_state['nl_prompt'] = st.session_state['nl_prompt_temp']

    dir_chapter = os.path.join(dir_, f"Chapter_{st.session_state['chapter_index']}")

    st.session_state["chapter_index"] += 1
    
    if not os.path.isdir(dir_chapter):
        os.makedirs(dir_chapter)

    with open(os.path.join(dir_chapter, "story.txt"), "w", encoding="utf-8") as file:
        file.write(st.session_state['story'])

    with open(os.path.join(dir_chapter, "image.png"), "wb") as fh:
        image_bytes = base64.b64decode(st.session_state['image'])
        fh.write(image_bytes)

    audio_base64 = generate_audio()

    with open(os.path.join(dir_chapter, "audio.mp3"), "wb") as fh:
        audio_bytes = base64.b64decode(audio_base64)
        fh.write(audio_bytes)

    st.session_state['story'] = None
    st.session_state['image'] = None

    
    
    
    

