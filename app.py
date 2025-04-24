import random
import glob
import os
import cv2
import numpy as np
import streamlit as st
import warnings
from fer import FER
from collections import Counter
from openai import OpenAI
from io import BytesIO

# --- Hide warnings ---
warnings.filterwarnings("ignore")

# --- Init ---
detector = FER(mtcnn=True)
input_dir = "./tupian"
gif_dir = "./gifs"
client = OpenAI(api_key="sk-c3d932d36b5b4deaaf8c3c6136dc38ce", base_url="https://api.deepseek.com")

# --- Emotion info ---
emotion_emojis = {
    "happy": "😊", "sad": "😢", "angry": "😠", "surprise": "😲",
    "neutral": "😐", "fear": "😨", "disgust": "🤢", "unknown": "❓"
}
emotion_sentences = {
    "happy": "You're doing great! Keep smiling!",
    "sad": "It's okay to feel sad. Better days are ahead.",
    "angry": "Take a deep breath. Find your calm.",
    "surprise": "Embrace the unexpected moments!",
    "neutral": "Steady and balanced. Keep going.",
    "fear": "You're stronger than you think.",
    "disgust": "It's okay to step back and regroup.",
    "unknown": "Every day is a new opportunity."
}

# --- Emotion detection for each day ---
calendar = np.full((5, 7), "unknown", dtype=object)
days_in_month = 31

for day in range(1, days_in_month + 1):
    img_path = os.path.join(input_dir, f"{day}.png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.detect_emotions(img_rgb)
        emotion = "unknown"
        if result:
            emotion = max(result[0]["emotions"], key=result[0]["emotions"].get)
        calendar[(day - 1) // 7, (day - 1) % 7] = emotion

# --- Most frequent emotion ---
emotion_counts = Counter(calendar.flatten())
most_frequent_emotion = emotion_counts.most_common(1)[0][0]
if most_frequent_emotion == "unknown" and len(emotion_counts) > 1:
    most_frequent_emotion = emotion_counts.most_common(2)[1][0]

# --- Title + Message ---
st.title("📅 Your Emotion Calendar")

# --- Elegant Calendar Table ---
# Weekday labels
weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
calendar_html = """
<style>
    th, td { text-align: center; padding: 8px; font-size: 20px; }
    table { border-collapse: collapse; width: 100%; }
    th { background-color: #f2f2f2; }
    td { border: 1px solid #ddd; }
</style>
<table><thead><tr>""" + "".join([f"<th>{day}</th>" for day in weekdays]) + "</tr></thead><tbody>"

day_counter = 1
for week in range(5):
    calendar_html += "<tr>"
    for weekday in range(7):
        if day_counter <= days_in_month:
            emoji = emotion_emojis.get(calendar[week, weekday], "❓")
            calendar_html += f"<td>{day_counter}<br>{emoji}</td>"
            day_counter += 1
        else:
            calendar_html += "<td></td>"
    calendar_html += "</tr>"
calendar_html += "</tbody></table>"

st.markdown(calendar_html, unsafe_allow_html=True)

# --- Pet GIF + Initial Emotion Message ---
st.header(f"Most Frequent Emotion: {most_frequent_emotion.capitalize()}")
cols = st.columns([1, 1])
with cols[0]:
    gif_candidates = glob.glob(os.path.join(gif_dir, f"{most_frequent_emotion}*.gif"))
    gif_path = random.choice(gif_candidates) if gif_candidates else None
    if os.path.exists(gif_path):
        st.image(gif_path, width=250)

# --- Chatting with your emotion pet ---
st.markdown("---")
st.subheader("💬 Talk with your Emotion Pet")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": f"你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"},
        {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
    ]

# Load GIF bytes for avatar
gif_avatar = None
gif_candidates = glob.glob(os.path.join(gif_dir, f"{most_frequent_emotion}*.gif"))
gif_path = random.choice(gif_candidates) if gif_candidates else None
if os.path.exists(gif_path):
    with open(gif_path, "rb") as f:
        gif_avatar = BytesIO(f.read())

# Display chat history (excluding system message)
for idx, msg in enumerate(st.session_state.chat_history[1:]):
    avatar = gif_avatar if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
# User input
user_input = st.chat_input("对小宠物说些什么吧")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=st.session_state.chat_history,
            stream=False
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = "Oops, something went wrong connecting to the pet brain 🧠"

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    avatar = gif_avatar if msg["role"] == "assistant" else None
    with st.chat_message("assistant", avatar=avatar):
        st.markdown(reply)
