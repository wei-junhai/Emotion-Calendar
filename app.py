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
from zhipuai import ZhipuAI

# --- Hide warnings ---
warnings.filterwarnings("ignore")

# --- Init ---
detector = FER(mtcnn=True)
input_dir = "./tupian"
gif_dir = "./gifs"
# client = OpenAI(api_key="sk-c3d932d36b5b4deaaf8c3c6136dc38ce", base_url="https://api.deepseek.com")
client = ZhipuAI(api_key="1e029a2bd2624e3da4c0e72b572ea42a.Ke0QfQKOaf0aBmUx")
chat_model_id = "glm-4"
# chat_model_id = "deepseek-chat"

# --- Emotion info ---
emotion_emojis = {
    "happy": "😊", "sad": "😢", "angry": "😠", "surprise": "😲",
    "neutral": "😐", "fear": "😨", "disgust": "🤢", "unknown": "❓"
}
emotion_sentences = {
    "happy": "你看起来很开心，继续保持微笑哦！",
    "sad": "伤心是正常的情绪，未来会更好。",
    "angry": "深呼吸一下，冷静一下自己。",
    "surprise": "惊喜是生活的小确幸呢！",
    "neutral": "保持平衡很棒，继续加油！",
    "fear": "你比你想象的更坚强。",
    "disgust": "有时候躲开让人不适的事也无妨。",
    "unknown": "每一天都是新的开始。"
}

emotion_labels_zh = {
    "happy": "开心",
    "sad": "伤心",
    "angry": "生气",
    "surprise": "惊讶",
    "neutral": "平静",
    "fear": "恐惧",
    "disgust": "厌恶",
    "unknown": "未知"
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
st.title("📅 你的情绪日历")

# --- Elegant Calendar Table ---
# Weekday labels
weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
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

# --- Upload a photo and update emotion ---
st.markdown("---")
st.header("📤 上传情绪照片")

uploaded_file = st.file_uploader("选择一张照片（png/jpg）", type=["png", "jpg", "jpeg"])
selected_day = st.number_input("选择要情绪照片的日期（1-31）", min_value=1, max_value=31, step=1)

if uploaded_file and st.button("🔄 更新情绪日历"):
    # 保存图片到 input_dir
    new_img_path = os.path.join(input_dir, f"{selected_day}.png")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite(new_img_path, img)

    # 重新识别这张图片的情绪
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.detect_emotions(img_rgb)
    emotion = "unknown"
    if result:
        emotion = max(result[0]["emotions"], key=result[0]["emotions"].get)

    # 更新 calendar 中该日期对应情绪
    calendar[(selected_day - 1) // 7, (selected_day - 1) % 7] = emotion

    # 重新计算最常见情绪
    emotion_counts = Counter(calendar.flatten())
    most_frequent_emotion = emotion_counts.most_common(1)[0][0]
    if most_frequent_emotion == "unknown" and len(emotion_counts) > 1:
        most_frequent_emotion = emotion_counts.most_common(2)[1][0]

    # 更新 chat_history 的 system 提示
    st.session_state.chat_history = [
        {"role": "system", "content": f"你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"},
        {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
    ]

    st.success(f"第 {selected_day} 天的照片与情绪已更新为 {emotion_labels_zh.get(emotion, '未知')} {emotion_emojis.get(emotion, '❓')}")
    st.rerun()


# --- Pet GIF + Initial Emotion Message ---
st.markdown("---")
st.header(f"本月最常见情绪：{emotion_labels_zh.get(most_frequent_emotion, '未知')}")
cols = st.columns([1, 1])
with cols[0]:
    gif_candidates = glob.glob(os.path.join(gif_dir, f"{most_frequent_emotion}*.gif"))
    gif_path = random.choice(gif_candidates) if gif_candidates else None
    if os.path.exists(gif_path):
        st.image(gif_path, width=250)

# --- Chatting with your emotion pet ---
st.markdown("---")
st.header("💬 和你的情绪宠物聊聊天吧")

# --- Clear Chat Button ---
if st.button("🗑️ 清除聊天记录"):
    st.session_state.chat_history = [
        {"role": "system", "content": f"你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"},
        {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
    ]
    st.rerun()

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
            model=chat_model_id,
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
