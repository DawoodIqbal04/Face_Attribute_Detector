import streamlit as st
import google.generativeai as genai
import os
import PIL.Image

# Streamlit Page Configuration
st.set_page_config(page_title='Face-Attributes-Dectector', page_icon='icons/face-scan.png', layout='centered', initial_sidebar_state='auto')


#Setting up API and loading model
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCAvWRmfIUTObnMB0418ouypQgx2rGheCw'

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel("models/gemini-1.5-flash-latest")


# Function to analyze image with custom prompt
def analyze_image(img):

    prompt = """
    Your are an AI trained to analze human attributes from images with high accuracy.
    Carefully analyze the given image and provide the following structured datails:

    No Notes and Discliamers in results.

    You have to return all results as you have image, don't want any apologize or empty results.

    - **Gender** (Male/Female/Non-binary)
    - **Age-Estimate** (e.g, 25 Yeras)
    - **Ethnicity** (e.g, Caucasian, African, Asian, etc.)
    - **Mood** (e.g, Happy, Sad, Angry, etc.)
    - **Hair-Color** (e.g, Black, Brown, Blonde, etc.)
    - **Hair-Length** (e.g, Short, Medium, Long)
    - **Facial-Hair** (e.g, Beard, Mustache, None)
    - **Eye-Color** (e.g, Brown, Blue, Green, etc.)
    - **Glasses** (e.g, Yes, No)
    - **Head-Wear** (e.g, Hat, Cap, None)
    - **Accessories** (e.g, Earrings, Necklace, etc.)
    - **Emotion-Detection** (e.g, Happy, Sad, Angry, Joyful, etc.)
    - **Confidence-Level** (Accuracy of prediction in percentage)
"""
    result = model.generate_content([prompt, img])

    return result.text.strip()


# Streamlit UI/ App Creation
col1, col2 = st.columns([1,5], vertical_alignment='center')

with col1:
    st.image('icons/face-scan.png', width=250)

with col2:
    st.markdown('# AI Face Attributes Finder')
    st.markdown('###### ✨ AI Powered Human Face Attributes Detector')

st.title('')

uploaded_image = st.file_uploader('Upload An Image To Detect Attributes', type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    img = PIL.Image.open(uploaded_image)
    person_info = analyze_image(img)

    col1, col2 =st.columns(2)

    with col1:
        st.image(img, use_container_width=True)
        
    with col2:
        st.write(person_info)