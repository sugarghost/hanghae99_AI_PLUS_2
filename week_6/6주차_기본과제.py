import base64
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Streamlit 페이지 설정
st.set_page_config(page_title="Image Chat Bot", layout="wide")

# API 키 입력 섹션
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요:", 
                               value=st.session_state.api_key,
                               type="password")

if api_key:
    st.session_state.api_key = api_key
    
# API 키 검증
if not st.session_state.api_key:
    st.warning("👈 사이드바에 OpenAI API 키를 입력해주세요.")
    st.stop()

st.title("Image Chat Bot")

try:
    model = ChatOpenAI(
        api_key=st.session_state.api_key,
        model="gpt-4o-mini", 
        max_tokens=1024
    )

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "images" not in st.session_state:
        st.session_state.images = []

    # 다중 이미지 업로더
    uploaded_files = st.file_uploader(
        "이미지를 업로드해주세요!", 
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    # 업로드된 이미지 처리
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file not in st.session_state.images:
                st.session_state.images.append(uploaded_file)
        
        # 업로드된 모든 이미지 표시
        cols = st.columns(len(st.session_state.images))
        for idx, image in enumerate(st.session_state.images):
            cols[idx].image(image)

    # 채팅 인터페이스
    if st.session_state.images:
        # 이전 대화 내역 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력
        if prompt := st.chat_input("질문을 입력해주세요"):
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 이미지 데이터 준비
            image_contents = []
            for img in st.session_state.images:
                img.seek(0)
                base64_image = base64.b64encode(img.read()).decode("utf-8")
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            # Assistant 응답
            with st.chat_message("assistant"):
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        *image_contents
                    ]
                )
                result = model.invoke([message])
                response = result.content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    if "Invalid API key" in str(e):
        st.warning("올바르지 않은 API 키입니다. 다시 확인해주세요.")