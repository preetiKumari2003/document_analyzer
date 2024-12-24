import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import openai
from datetime import datetime, timedelta


os.environ["OPENAI_API_KEY"] = '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'


st.set_page_config(page_title="Travel Itinerary Generator", page_icon="✈️", layout="wide")


st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            color: #2A9D8F;
            font-weight: bold;
        }
        .header {
            text-align: center;
            font-size: 30px;
            color: #264653;
            margin-bottom: 20px;
        }
        .input-container {
            margin-top: 30px;
            padding: 25px;
            border: 2px solid #2A9D8F;
            border-radius: 10px;
            background-color: #F1FAEE;
        }
        .stButton>button {
            background-color: #2A9D8F;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #1D6F56;
        }
        .itinerary-day {
            background-color: #E9F7F1;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 5px solid #2A9D8F;
            border-radius: 5px;
            font-size: 18px;
        }
        .stTextInput input {
            border-radius: 10px;
            padding: 10px;
            border: 2px solid #2A9D8F;
            font-size: 16px;
            background-color: #F1FAEE;
        }
        .stTextArea textarea {
            border-radius: 10px;
            padding: 10px;
            border: 2px solid #2A9D8F;
            font-size: 16px;
            background-color: #F1FAEE;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">Travel Itinerary Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Plan your trip in minutes!</div>', unsafe_allow_html=True)


with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    from_location = st.text_input("From which location are you traveling?")
    destination = st.text_input("Enter your destination location:")
    days = st.number_input("How many days will you stay?", min_value=1, max_value=30, step=1)
    preferences = st.text_area("Enter your preferences (e.g., adventure, budget-friendly, food):")
    start_date = st.date_input("Select your start date")
    total_budget = st.number_input("Enter your total budget for the trip ($)", min_value=1, step=1)
    st.markdown('</div>', unsafe_allow_html=True)


if total_budget > 0 and days > 0:
    daily_budget = total_budget / days
else:
    daily_budget = 0


if st.button("Generate Itinerary"):
    if not from_location or not destination or not days or not start_date or not total_budget:
        st.error("Please fill in all required fields.")
    elif daily_budget > total_budget:
        st.error("Your daily budget exceeds your total budget. Please adjust your budget.")
    else:

        raw_text = f"""
        act as a AI assistant trip plannner.
        Plan for {days}-day trip to {destination}:
        - Top some fmous Adventure spots of location {destination}.
        - Budget-friendly hotels around {destination}.
        - Local Food of {destination} to explore.
        - Travel tips according to weather of {destination} and {days}.
        short example for refrence:
        - Adventure spots: Solang Valley, Rohtang Pass, Jogini Waterfalls.
        - Budget-friendly hotels: XYZ Hotel, ABC Hostel.
        - Food to explore: Himachali Dham, local cafes, street food at Mall Road.
        - Travel tips: Carry warm clothes and book adventure activities in advance.
        """


        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)


        embeddings = OpenAIEmbeddings()


        document_search = FAISS.from_texts(texts, embeddings)


        chain = load_qa_chain(OpenAI(), chain_type="stuff")


        with st.spinner("Creating your itinerary..."):
            days_plan = []
            for day in range(1, days + 1):
                query = f"Create a plan for Day {day} in {destination} with preferences: {preferences}"
                docs = document_search.similarity_search(query)
                answer = chain.run(input_documents=docs, question=query)


                day_date = start_date + timedelta(days=day - 1)


                days_plan.append(f"**Day {day} ({day_date.strftime('%B %d, %Y')}):**\n{answer}\n\nEstimated Daily Budget: ${daily_budget:.2f}\n")


        st.subheader(f"{destination} Trip Itinerary ({days} Days)")
        for day_plan in days_plan:
            st.markdown(f'<div class="itinerary-day">{day_plan}</div>', unsafe_allow_html=True)


        st.markdown(f"### Total Budget for {days} Days: ${total_budget}")
