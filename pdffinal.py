#cd C:\PycharmProjects\NEWB11108063\AIweek4
#streamlit run AI081_hand1.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from pdf2image import convert_from_path
import time
import os

# PDF 轉圖片
def load_pdf_as_images(pdf_path):
    poppler_path = r"C:\PycharmProjects\NEWB11108063\AIweek4\poppler-24.08.0\Library\bin"
    pages = convert_from_path(pdf_path, dpi=150, poppler_path=poppler_path)
    page_images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]
    return page_images

# Mediapipe 設定
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.5)

# 初始 PDF 路徑與圖片
pdf_path = st.file_uploader("Upload your PDF", type="pdf")
if pdf_path is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf_path.read())
    page_images = load_pdf_as_images("uploaded.pdf")

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'gesture_text' not in st.session_state:
        st.session_state.gesture_text = ""
    if 'serious_time' not in st.session_state:
        st.session_state.serious_time = 0.0
    if 'lazy_time' not in st.session_state:
        st.session_state.lazy_time = 0.0
    if 'prev_hand_x' not in st.session_state:
        st.session_state.prev_hand_x = None
    if 'last_switch_time' not in st.session_state:
        st.session_state.last_switch_time = time.time()
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'movement_buffer' not in st.session_state:
        st.session_state.movement_buffer = []

    st.title("📄 Gesture-Controlled PDF Viewer")
    st.markdown("Use right-hand gestures to flip pages. Focus is tracked.")

    start_btn = st.button("▶️ Start")
    stop_btn = st.button("🛑 Stop")

    pdf_placeholder = st.empty()
    info_placeholder = st.empty()
    webcam_placeholder = st.empty()

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
        st.markdown("## 📊 Session Summary")
        st.markdown(f"""
        - **Total Focus Time:** {st.session_state.serious_time:.1f} seconds  
        - **Total Away Time:** {st.session_state.lazy_time:.1f} seconds  
        - **Last Viewed Page:** {st.session_state.current_page + 1}  
        - **Last Gesture:** {st.session_state.gesture_text}  
        """)

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(image_rgb)
            face_results = face_mesh.process(image_rgb)

            # 手勢偵測
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for idx, hand_handedness in enumerate(hand_results.multi_handedness):
                    if hand_handedness.classification[0].label == "Right":
                        hand_landmarks = hand_results.multi_hand_landmarks[idx]
                        palm_x = np.mean([lm.x for lm in hand_landmarks.landmark])
                        current_time = time.time()

                        st.session_state.movement_buffer.append(palm_x)
                        if len(st.session_state.movement_buffer) > 4:
                            st.session_state.movement_buffer.pop(0)
                            movement_avg = st.session_state.movement_buffer[-1] - st.session_state.movement_buffer[0]

                            cooldown = 0.6
                            threshold = 0.05

                            if movement_avg > threshold and current_time - st.session_state.last_switch_time > cooldown:
                                st.session_state.current_page = min(len(page_images) - 1, st.session_state.current_page + 1)
                                st.session_state.gesture_text = "➡️ Next Page"
                                st.session_state.last_switch_time = current_time
                                st.session_state.movement_buffer.clear()
                            elif movement_avg < -threshold and current_time - st.session_state.last_switch_time > cooldown:
                                st.session_state.current_page = max(0, st.session_state.current_page - 1)
                                st.session_state.gesture_text = "⬅️ Previous Page"
                                st.session_state.last_switch_time = current_time
                                st.session_state.movement_buffer.clear()

            # 專心與偷懶時間統計
            if face_results.multi_face_landmarks:
                st.session_state.serious_time += 0.1
            else:
                st.session_state.lazy_time += 0.1

            # 顯示 PDF 頁面
            pdf_page = page_images[st.session_state.current_page]
            pdf_placeholder.image(pdf_page, caption=f"📄 Page {st.session_state.current_page + 1}", channels="BGR")

            # 顯示 Webcam 畫面（可視化除錯）
            webcam_placeholder.image(frame, caption="🖐 Webcam Preview", channels="BGR")

            # 顯示資訊
            info_placeholder.markdown(f"""
            - **Gesture:** {st.session_state.gesture_text}  
            - **Focus Time:** {st.session_state.serious_time:.1f} sec  
            - **Away Time:** {st.session_state.lazy_time:.1f} sec  
            - **Page:** {st.session_state.current_page + 1} / {len(page_images)}
            """)

            time.sleep(0.1)
        cap.release()


