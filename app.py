from wordcloud import WordCloud
import os
import re 
import json
import spacy
from spacy import displacy
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

@st.cache_resource
def load_dict():
    # spacy.cli.download("en_core_web_sm")
    return spacy.load('en_core_web_sm')

def generate_wordcloud(text):
    wordcloud = WordCloud(width = 1200,height = 800,background_color = 'black',min_font_size = 10).generate(text)
    wordcloud.to_file("wc.png")
    return "wc.png"

def ner(text):
    nlp = load_dict() #python -m spacy download en_core_web_sm
    doc = nlp(text)
    print("-------doc------")
    print(doc)
    html = displacy.render(doc, style = 'ent',jupyter = False)
    html = html.replace("\n\n","\n")
    st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)

def gen_summarization(text):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    pmpt = "Please find the summary from the below text in maximum of 50 words:\n {text}"
    prompt = ChatPromptTemplate.from_template(pmpt)
    pipeline = prompt|llm|StrOutputParser()
    response = pipeline.invoke({"text":text})
    return str(response)

def main():
    st.title("Text Analitics 🕵️‍♂️")
    input_text = st.text_area("Enter your paragraph here")

    if input_text:
        if st.button("Analyze Text 🧐"):
            col1 , col2 = st.columns([1,1])
            with col1:
                st.markdown("**Key Findings based on your Text**")
                st.success(gen_summarization(input_text))
            with col2:
                st.markdown("**Output Wordcloud**")
                st.image(generate_wordcloud(input_text))
            st.markdown("**Named Entity Recognition**")
            ner(input_text)

if __name__ =="__main__":
    main()