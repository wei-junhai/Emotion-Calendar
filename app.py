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
import hashlib
import json

# --- Hide warnings ---
warnings.filterwarnings("ignore")

# --- Directories ---
input_dir = "./tupian"
gif_dir = "./gifs"
data_dir = "./userdata"  # Local storage for user data
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# --- Init emotion detector and AI client ---
detector = FER(mtcnn=True)
client = ZhipuAI(api_key="1e029a2bd2624e3da4c0e72b572ea42a.Ke0QfQKOaf0aBmUx")
chat_model_id = "glm-4"

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

# --- Helper functions for user data management ---

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def get_user_path(username: str) -> str:
    return os.path.join(data_dir, username)

def load_user_data(username: str):
    user_path = get_user_path(username)
    calendar_path = os.path.join(user_path, "calendar.json")
    chat_path = os.path.join(user_path, "chat_history.json")
    calendar = None
    chat_history = None
    if os.path.exists(calendar_path):
        with open(calendar_path, "r", encoding="utf-8") as f:
            calendar = np.array(json.load(f))
    if os.path.exists(chat_path):
        with open(chat_path, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
    return calendar, chat_history

def save_user_data(username: str, calendar, chat_history):
    user_path = get_user_path(username)
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    calendar_path = os.path.join(user_path, "calendar.json")
    chat_path = os.path.join(user_path, "chat_history.json")
    with open(calendar_path, "w", encoding="utf-8") as f:
        json.dump(calendar.tolist(), f, ensure_ascii=False)
    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False)

def save_user_image(username: str, day: int, img):
    user_path = get_user_path(username)
    img_dir = os.path.join(user_path, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, f"{day}.png")
    cv2.imwrite(img_path, img)
    return img_path

def load_user_image(username: str, day: int):
    img_path = os.path.join(get_user_path(username), "images", f"{day}.png")
    if os.path.exists(img_path):
        return cv2.imread(img_path)
    else:
        return None

# --- User management: simple registration and login ---
if "users" not in st.session_state:
    st.session_state.users = {}  # username -> password_hash

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "calendar" not in st.session_state:
    st.session_state.calendar = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

def register_user(username: str, password: str):
    users_file = os.path.join(data_dir, "users.json")
    if os.path.exists(users_file):
        with open(users_file, "r", encoding="utf-8") as f:
            users = json.load(f)
    else:
        users = {}
    if username in users:
        return False, "用户名已存在"
    users[username] = hash_password(password)
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False)
    return True, "注册成功"

def login_user(username: str, password: str):
    users_file = os.path.join(data_dir, "users.json")
    if not os.path.exists(users_file):
        return False, "用户不存在"
    with open(users_file, "r", encoding="utf-8") as f:
        users = json.load(f)
    if username not in users:
        return False, "用户不存在"
    if users[username] != hash_password(password):
        return False, "密码错误"
    return True, "登录成功"

def initialize_user_data(username: str):
    # Load calendar and chat history if exists, else init
    calendar, chat_history = load_user_data(username)
    if calendar is None:
        # Initialize calendar with "unknown"
        calendar = np.full((5, 7), "unknown", dtype=object)
        # Try load any existing images in user folder
        for day in range(1, 32):
            img = load_user_image(username, day)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = detector.detect_emotions(img_rgb)
                emotion = "unknown"
                if result:
                    emotion = max(result[0]["emotions"], key=result[0]["emotions"].get)
                calendar[(day - 1) // 7, (day - 1) % 7] = emotion
    if chat_history is None:
        # Set default chat history
        most_frequent_emotion = Counter(calendar.flatten()).most_common(1)[0][0]
        if most_frequent_emotion == "unknown" and len(Counter(calendar.flatten())) > 1:
            most_frequent_emotion = Counter(calendar.flatten()).most_common(2)[1][0]
        chat_history = [
            {"role": "system", "content": f"你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"},
            {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
        ]
    return calendar, chat_history

# --- UI: login/register ---
st.title("🐶 情绪宠物日历")

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["登录", "注册"])
    with tab1:
        login_username = st.text_input("用户名", key="login_username")
        login_password = st.text_input("密码", type="password", key="login_password")
        if st.button("登录"):
            success, msg = login_user(login_username, login_password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success(msg)
                # Load or init user data
                calendar, chat_history = initialize_user_data(login_username)
                st.session_state.calendar = calendar
                st.session_state.chat_history = chat_history
                st.rerun()
            else:
                st.error(msg)
    with tab2:
        reg_username = st.text_input("用户名", key="reg_username")
        reg_password = st.text_input("密码", type="password", key="reg_password")
        reg_password2 = st.text_input("确认密码", type="password", key="reg_password2")
        if st.button("注册"):
            if reg_password != reg_password2:
                st.error("两次输入密码不一致")
            elif not reg_username or not reg_password:
                st.error("用户名和密码不能为空")
            else:
                success, msg = register_user(reg_username, reg_password)
                if success:
                    st.success(msg + "，请登录")
                else:
                    st.error(msg)
    st.stop()

# --- Main app after login ---

username = st.session_state.username
calendar = st.session_state.calendar
chat_history = st.session_state.chat_history

days_in_month = 31

# Calculate most frequent emotion
emotion_counts = Counter(calendar.flatten())
most_frequent_emotion = emotion_counts.most_common(1)[0][0]
if most_frequent_emotion == "unknown" and len(emotion_counts) > 1:
    most_frequent_emotion = emotion_counts.most_common(2)[1][0]

st.title(f"📅 {username} 的情绪日历")

# --- Elegant Calendar Table ---
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
st.header("📤 上传或拍摄情绪照片")
upload_tab, camera_tab = st.tabs(["📁 上传图片", "📸 拍照"])
uploaded_file = None
camera_image = None

with upload_tab:
    uploaded_file = st.file_uploader("选择一张照片（png/jpg）", type=["png", "jpg", "jpeg"])
with camera_tab:
    camera_image = st.camera_input("使用摄像头拍摄一张照片")
selected_day = st.number_input("选择要情绪照片的日期（1-31）", min_value=1, max_value=31, step=1)

if (uploaded_file or camera_image) and st.button("🔄 更新情绪日历"):
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        st.error("没有提供有效的图片。")
        st.stop()

    # Save user image locally
    save_user_image(username, selected_day, img)

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
    chat_history = [
        {"role": "system", "content": f"你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"},
        {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
    ]

    st.session_state.calendar = calendar
    st.session_state.chat_history = chat_history

    # Save user data
    save_user_data(username, calendar, chat_history)

    st.success(f"第 {selected_day} 天的照片与情绪已更新为 {emotion_labels_zh.get(emotion, '未知')} {emotion_emojis.get(emotion, '❓')}")
    st.rerun()

# --- Pet GIF + Initial Emotion Message ---
st.markdown("---")
st.header(f"本月最常见情绪：{emotion_labels_zh.get(most_frequent_emotion, '未知')}")

cols = st.columns([1, 1])
with cols[0]:
    gif_candidates = glob.glob(os.path.join(gif_dir, f"{most_frequent_emotion}*.gif"))
    gif_path = random.choice(gif_candidates) if gif_candidates else None
    if gif_path and os.path.exists(gif_path):
        st.image(gif_path, width=250)

# --- Chatting with your emotion pet ---
st.markdown("---")
st.header("💬 和你的情绪宠物聊聊天吧")

# --- Clear Chat Button ---
if st.button("🗑️ 清除聊天记录"):
    chat_history = [
        {"role": "system", "content": f"你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"},
        {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
    ]
    st.session_state.chat_history = chat_history
    save_user_data(username, calendar, chat_history)
    st.rerun()

if chat_history is None or len(chat_history) == 0:
    chat_history = [
        {"role": "system", "content": f"你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"},
        {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
    ]
    st.session_state.chat_history = chat_history

# Load GIF bytes for avatar
gif_avatar = None
with open("./gifs/angry1.gif", "rb") as f:
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
    save_user_data(username, calendar, st.session_state.chat_history)
    with st.chat_message("assistant", avatar=gif_avatar):
        st.markdown(reply)

# --- Logout button ---
st.markdown("---")
if st.button("退出登录"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.calendar = None
    st.session_state.chat_history = None
    st.rerun()
