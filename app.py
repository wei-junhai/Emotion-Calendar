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
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import bcrypt
import json

# --- Google Sheets Setup ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["google"], scope)
gc = gspread.authorize(credentials)
sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1ihAXZ3bl82Rf3mlohzPwWmIlhkfwW-oCps3_8iOgui8/edit?usp=sharing").sheet1

def find_user(username):
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row['username'] == username:
            return idx, row
    return None, None

def register_user(username, password):
    if not username or not password:
        return False, "用户名或密码不能为空"
    _, existing = find_user(username)
    if existing:
        return False, "用户名已存在"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    sheet.append_row([username, hashed, "{}"])
    return True, "注册成功"

def login_user(username, password):
    idx, user = find_user(username)
    if not user:
        return False, "用户不存在"
    if bcrypt.checkpw(password.encode(), user['password'].encode()):
        return True, user
    return False, "密码错误"

# --- Login/Register UI ---
st.sidebar.title("🔐 用户登录")
if "user" not in st.session_state:
    login_tab, register_tab = st.sidebar.tabs(["登录", "注册"])
    with login_tab:
        login_user_input = st.text_input("用户名", key="login_user")
        login_pass_input = st.text_input("密码", type="password", key="login_pass")
        if st.button("登录"):
            success, user_data = login_user(login_user_input, login_pass_input)
            if success:
                st.session_state.user = login_user_input
                st.success("登录成功")
                st.rerun()
            else:
                st.error(user_data)
    with register_tab:
        register_user_input = st.text_input("新用户名", key="register_user")
        register_pass_input = st.text_input("新密码", type="password", key="register_pass")
        if st.button("注册"):
            success, msg = register_user(register_user_input, register_pass_input)
            if success:
                st.success(msg)
            else:
                st.error(msg)
    st.stop()
else:
    st.sidebar.success(f"已登录：{st.session_state.user}")
    if st.sidebar.button("退出登录"):
        del st.session_state.user
        st.rerun()

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
    new_img_path = os.path.join(input_dir, f"{selected_day}.png")
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
