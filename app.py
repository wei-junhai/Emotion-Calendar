import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import hashlib
import os
import random
import glob
import cv2
import numpy as np
import warnings
from fer import FER
from collections import Counter
from io import BytesIO
from zhipuai import ZhipuAI

# --- Hide warnings ---
warnings.filterwarnings("ignore")

# --- Init ---
detector = FER(mtcnn=True)
input_dir = "./tupian"
gif_dir = "./gifs"
client = ZhipuAI(api_key="1e029a2bd2624e3da4c0e72b572ea42a.Ke0QfQKOaf0aBmUx")
chat_model_id = "glm-4"

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

# --- Connect to Google Sheets ---
conn = st.connection("gsheets", type=GSheetsConnection)
# The sheet contains columns: user_id | password_hash | chat_history_json

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_users_df():
    try:
        df = conn.read(worksheet="Sheet1")
        if df.empty:
            df = pd.DataFrame(columns=["user_id", "password_hash", "chat_history_json"])
        return df
    except Exception:
        # Sheet read error fallback
        return pd.DataFrame(columns=["user_id", "password_hash", "chat_history_json"])

def save_users_df(df: pd.DataFrame):
    # This method overwrites the whole sheet with the new dataframe.
    # streamlit_gsheets does not provide update row; you could reload and overwrite all.
    # If performance is critical, consider batch update or API client.
    conn.write(df, worksheet="Sheet1")

def register_user(user_id: str, password: str) -> bool:
    df = load_users_df()
    if user_id in df["user_id"].values:
        return False  # User exists
    password_hash = hash_password(password)
    new_row = {"user_id": user_id, "password_hash": password_hash, "chat_history_json": ""}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users_df(df)
    return True

def authenticate_user(user_id: str, password: str) -> bool:
    df = load_users_df()
    if user_id not in df["user_id"].values:
        return False
    password_hash = hash_password(password)
    stored_hash = df.loc[df["user_id"] == user_id, "password_hash"].values[0]
    return stored_hash == password_hash

def get_user_chat_history(user_id: str):
    df = load_users_df()
    row = df.loc[df["user_id"] == user_id]
    if row.empty:
        return None
    chat_json = row["chat_history_json"].values[0]
    if chat_json:
        try:
            return pd.io.json.loads(chat_json)
        except Exception:
            return None
    return None

def update_user_chat_history(user_id: str, chat_history):
    df = load_users_df()
    idx = df.index[df["user_id"] == user_id]
    if len(idx) == 0:
        return
    df.at[idx[0], "chat_history_json"] = pd.io.json.dumps(chat_history)
    save_users_df(df)


# --- Authentication UI ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

if not st.session_state.logged_in:
    st.title("用户登录 / 注册")

    auth_mode = st.radio("请选择操作", ("登录", "注册"))

    user_id_input = st.text_input("用户名")
    password_input = st.text_input("密码", type="password")

    if st.button("提交"):
        if auth_mode == "注册":
            if user_id_input and password_input:
                if register_user(user_id_input, password_input):
                    st.success("注册成功，请登录。")
                else:
                    st.error("用户名已存在。")
            else:
                st.error("用户名和密码不能为空。")
        else:  # 登录
            if user_id_input and password_input:
                if authenticate_user(user_id_input, password_input):
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id_input
                    st.experimental_rerun()
                else:
                    st.error("用户名或密码错误。")
            else:
                st.error("用户名和密码不能为空。")

else:
    st.sidebar.write(f"已登录用户: {st.session_state.user_id}")
    if st.sidebar.button("登出"):
        st.session_state.logged_in = False
        st.session_state.user_id = ""
        st.experimental_rerun()

    # --- Load or init user chat history ---
    user_chat_history = get_user_chat_history(st.session_state.user_id)
    if user_chat_history is None:
        user_chat_history = [
            {"role": "system", "content": "你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。"},
            {"role": "assistant", "content": emotion_sentences["neutral"]}
        ]
    st.session_state.chat_history = user_chat_history

    # --- Emotion calendar logic ---

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

    emotion_counts = Counter(calendar.flatten())
    most_frequent_emotion = emotion_counts.most_common(1)[0][0]
    if most_frequent_emotion == "unknown" and len(emotion_counts) > 1:
        most_frequent_emotion = emotion_counts.most_common(2)[1][0]

    # --- UI ---

    st.title("📅 你的情绪日历")

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

    # --- Upload photo and update emotion ---
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

        # Update system prompt and assistant reply in chat_history
        st.session_state.chat_history = [
            {"role": "system", "content": f"你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"},
            {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
        ]
        update_user_chat_history(st.session_state.user_id, st.session_state.chat_history)

        st.success(f"第 {selected_day} 天的照片与情绪已更新为 {emotion_labels_zh.get(emotion, '未知')} {emotion_emojis.get(emotion, '❓')}")
        st.experimental_rerun()

    # --- Display most frequent emotion with GIF ---
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

    if st.button("🗑️ 清除聊天记录"):
        st.session_state.chat_history = [
            {"role": "system", "content": f"你是一个活泼，有趣的比熊犬，叫Lucky。你会关注主人情绪，并帮主人化解坏情绪。记住，无情绪时请保持中立。你主人当前的情绪是{most_frequent_emotion}，你在对话中需要关注主人这个情绪，提供相应的情绪价值以及帮助。"},
            {"role": "assistant", "content": f'"{emotion_sentences[most_frequent_emotion]}"'}
        ]
        update_user_chat_history(st.session_state.user_id, st.session_state.chat_history)
        st.experimental_rerun()

    gif_avatar = None
    gif_candidates = glob.glob(os.path.join(gif_dir, f"{most_frequent_emotion}*.gif"))
    gif_path = random.choice(gif_candidates) if gif_candidates else None
    if gif_path and os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            gif_avatar = BytesIO(f.read())

    for idx, msg in enumerate(st.session_state.chat_history[1:]):
        avatar = gif_avatar if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

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
        except Exception:
            reply = "Oops, something went wrong connecting to the pet brain 🧠"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        update_user_chat_history(st.session_state.user_id, st.session_state.chat_history)

        with st.chat_message("assistant", avatar=gif_avatar):
            st.markdown(reply)
