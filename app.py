
from transformers import pipeline
from PIL import Image
import streamlit as st



st.set_page_config(page_title='이미지 분류', page_icon="🖼️")

@st.cache_resource
def get_model_instance():
    return pipeline('image-classification', model = 'google/vit-base-patch16-224')


st.title('이미지 분류')


uploaded_file  = st.file_uploader('이미지를 업로드해주세요.', type = ['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image)
    submit = st.button('분류하기')

    if submit:
        model = get_model_instance()
        with st.spinner('분류 중'):
            result = model(image)[0]
        label = result['label']
        score = result['score']
        st.write(label)
        st.metric(label = '신뢰도', value = f'{score:.2%}')
        st.progress(score)

