import streamlit as st
from transformers import pipeline

# Hugging Face Transformers를 사용하여 요약 모델을 불러옵니다.
summarizer = pipeline("summarization". model="t5-small")

# Streamlit 애플리케이션 설정
# st.title("자연어 요약 앱")
# st.write("이 앱은 텍스트를 입력하고 자동으로 요약하는 데 사용됩니다.")

# 사용자로부터 텍스트 입력을 받습니다.
# user_input = st.text_area("텍스트를 입력하세요:")

# 사용자가 입력한 텍스트가 있는 경우 요약 결과를 생성합니다.
# if user_input:
#     # 요약 모델을 사용하여 텍스트 요약 수행
#     summary = summarizer(user_input, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
    
#     # 요약 결과를 출력
#     st.subheader("자동 생성된 요약:")
#     st.write(summary)
