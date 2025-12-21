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
data_dir = "./userdata"
os.makedirs(data_dir, exist_ok=True)

# ✅ NEW: Prompt file path
prompt_path = "./prompt.md"

# --- Init emotion detector and AI client ---
detector = FER(mtcnn=True)
client = ZhipuAI(api_key="1221554b5a3c4965b546469e2658325b.XHvRHc8OPg4ZGyNf")
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
    "happy": "开心", "sad": "伤心", "angry": "生气", "surprise": "惊讶",
    "neutral": "平静", "fear": "恐惧", "disgust": "厌恶", "unknown": "未知"
}

# ✅ NEW: Read prompt.md as system prompt template (with cache)
@st.cache_data(show_spinner=False)
def load_system_prompt_template(path: str) -> str:
    """
    Load prompt template from local markdown file.
    Supports python str.format placeholders, e.g. {most_frequent_emotion}.
    """
    if not os.path.exists(path):
        # Fallback: keep your original template to avoid app crash
        return (
            "你是一个活泼的机器人，叫 Moodi。你会关注主人情绪，并帮主人化解坏情绪。"
            "记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，"
            "你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

system_prompt_template = load_system_prompt_template(prompt_path)

def build_system_prompt(most_frequent_emotion: str) -> str:
    """
    Render the system prompt from template using python format().
    Keep it robust even if template contains unknown placeholders.
    """
    try:
        return system_prompt_template.format(most_frequent_emotion=most_frequent_emotion)
    except KeyError as e:
        # If prompt.md contains placeholders you didn't provide, don't crash:
        # just append a note + still include the critical variable.
        return (
            system_prompt_template
            + "\n\n"
            + f"（提示：prompt.md 中存在未提供的占位符：{e}。当前 most_frequent_emotion={most_frequent_emotion}）"
        )

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
    os.makedirs(user_path, exist_ok=True)
    calendar_path = os.path.join(user_path, "calendar.json")
    chat_path = os.path.join(user_path, "chat_history.json")
    with open(calendar_path, "w", encoding="utf-8") as f:
        json.dump(calendar.tolist(), f, ensure_ascii=False)
    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False)

def save_user_image(username: str, day: int, img):
    user_path = get_user_path(username)
    img_dir = os.path.join(user_path, "images")
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"{day}.png")
    cv2.imwrite(img_path, img)
    return img_path

def load_user_image(username: str, day: int):
    img_path = os.path.join(get_user_path(username), "images", f"{day}.png")
    return cv2.imread(img_path) if os.path.exists(img_path) else None

# --- User Management ---
if "users" not in st.session_state:
    st.session_state.users = {}

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
    users = {}
    if os.path.exists(users_file):
        with open(users_file, "r", encoding="utf-8") as f:
            users = json.load(f)
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
    calendar, chat_history = load_user_data(username)
    if calendar is None:
        calendar = np.full((5, 7), "unknown", dtype=object)
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
        most_frequent_emotion = Counter(calendar.flatten()).most_common(1)[0][0]
        if most_frequent_emotion == "unknown" and len(Counter(calendar.flatten())) > 1:
            most_frequent_emotion = Counter(calendar.flatten()).most_common(2)[1][0]

        # ✅ CHANGED: use prompt.md template as system prompt
        chat_history = [
            {"role": "system", "content": build_system_prompt(most_frequent_emotion)},
            {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
        ]
    return calendar, chat_history

# --- Login/Register UI ---
st.set_page_config(page_title="情绪日历", layout="wide")
st.title("🤖 情绪日历")

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

# --- Main Tabs ---
username = st.session_state.username
calendar = st.session_state.calendar
chat_history = st.session_state.chat_history
days_in_month = 31

emotion_counts = Counter(calendar.flatten())
most_frequent_emotion = emotion_counts.most_common(1)[0][0]
if most_frequent_emotion == "unknown" and len(emotion_counts) > 1:
    most_frequent_emotion = emotion_counts.most_common(2)[1][0]

tab1, tab2, tab3 = st.tabs(["📅 情绪日历", "💬 情绪聊天", "📖 心情日记"])

# --- Tab 1: Calendar ---
with tab1:
    st.header(f"🧭 {username} 的情绪月历")
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

    st.subheader("📷 上传或拍照记录每日情绪")
    upload_tab, camera_tab = st.tabs(["📁 上传图片", "📸 拍照"])
    uploaded_file = None
    camera_image = None
    with upload_tab:
        uploaded_file = st.file_uploader("选择一张照片（png/jpg）", type=["png", "jpg", "jpeg"])
    with camera_tab:
        camera_image = st.camera_input("拍一张照片吧")

    selected_day = st.number_input("请选择日期（1-31）", min_value=1, max_value=31, step=1)

    if (uploaded_file or camera_image) and st.button("🔄 更新情绪"):
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif camera_image:
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            st.error("无效图片")
            st.stop()

        save_user_image(username, selected_day, img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.detect_emotions(img_rgb)
        emotion = "unknown"
        if result:
            emotion = max(result[0]["emotions"], key=result[0]["emotions"].get)
        calendar[(selected_day - 1) // 7, (selected_day - 1) % 7] = emotion

        emotion_counts = Counter(calendar.flatten())
        most_frequent_emotion = emotion_counts.most_common(1)[0][0]
        if most_frequent_emotion == "unknown" and len(emotion_counts) > 1:
            most_frequent_emotion = emotion_counts.most_common(2)[1][0]

        # ✅ CHANGED: reset chat history using prompt.md template
        chat_history = [
            {"role": "system", "content": build_system_prompt(most_frequent_emotion)},
            {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
        ]

        st.session_state.calendar = calendar
        st.session_state.chat_history = chat_history
        save_user_data(username, calendar, chat_history)

        st.success(f"第 {selected_day} 天更新为 {emotion_labels_zh.get(emotion, '未知')} {emotion_emojis.get(emotion, '❓')}")
        st.rerun()

    st.subheader(f"🌟 本月最常见情绪：{emotion_labels_zh.get(most_frequent_emotion, '未知')}")
    gif_candidates = glob.glob(os.path.join(gif_dir, f"{most_frequent_emotion}*.gif"))
    gif_path = random.choice(gif_candidates) if gif_candidates else None
    if gif_path and os.path.exists(gif_path):
        st.image(gif_path, width=250)

# --- Tab 2: Chat ---
with tab2:
    st.header("🗣️ 和 Moodi 聊聊天")

    if st.button("🗑️ 清除聊天记录"):
        # ✅ CHANGED: reset with prompt.md template
        chat_history = [
            {"role": "system", "content": build_system_prompt(most_frequent_emotion)},
            {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
        ]
        st.session_state.chat_history = chat_history
        save_user_data(username, calendar, chat_history)
        st.rerun()

    gif_avatar = None
    with open("./gifs/angry1.gif", "rb") as f:
        gif_avatar = BytesIO(f.read())

    for msg in st.session_state.chat_history[1:]:
        with st.chat_message(msg["role"], avatar=gif_avatar if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    user_input = st.chat_input("说点什么吧")
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
        except Exception:
            reply = "机器人处理器出错了，请稍后再试 🧠"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        save_user_data(username, calendar, st.session_state.chat_history)
        with st.chat_message("assistant", avatar=gif_avatar):
            st.markdown(reply)

# --- Tab 3: Diary ---
with tab3:
    st.header("📝 心情日记")

    from datetime import datetime
    this_month = datetime.today().month  # 当前月份
    today = datetime.today().day        # 今天几号

    # --- Diary state ---
    if "diary_day" not in st.session_state:
        st.session_state.diary_day = today
    if "diary" not in st.session_state:
        user_path = get_user_path(username)
        diary_path = os.path.join(user_path, "diary.json")
        if os.path.exists(diary_path):
            with open(diary_path, "r", encoding="utf-8") as f:
                st.session_state.diary = json.load(f)
        else:
            st.session_state.diary = {}

    diary = st.session_state.diary
    current_day = st.session_state.diary_day

    # 🔑 Use current_day for display
    st.subheader(f"📅 {this_month}月{current_day}日的日记")

    # Load existing content for this page
    current_text = diary.get(str(current_day), "")

    # Text input area (auto-save on change)
    text = st.text_area("写下今天的心情吧：", value=current_text, height=200, key=f"diary_{current_day}")

    # Auto-save if changed
    if text != current_text:
        diary[str(current_day)] = text
        user_path = get_user_path(username)
        os.makedirs(user_path, exist_ok=True)
        diary_path = os.path.join(user_path, "diary.json")
        with open(diary_path, "w", encoding="utf-8") as f:
            json.dump(diary, f, ensure_ascii=False, indent=2)
        st.toast("日记已自动保存 ✅", icon="💾")

    # Navigation buttons = 昨天 / 明天
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ 上一天") and current_day > 1:
            st.session_state.diary_day -= 1
            st.rerun()
    with col2:
        if st.button("➡️ 下一天") and current_day < days_in_month:
            st.session_state.diary_day += 1
            st.rerun()

# --- Logout ---
st.markdown("---")
if st.button("退出登录"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.calendar = None
    st.session_state.chat_history = None
    st.rerun()
