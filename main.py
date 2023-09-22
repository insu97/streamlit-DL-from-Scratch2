import streamlit as st
from common.util import preprocess

st.title("밑바닥부터 시작하는 딥러닝2")

CH02 = st.checkbox('CH02')

if CH02 :
    text = st.text_input('영어문장을 입력해주세요.')
    if text :
        corpus, word_to_id, id_to_word = preprocess(text)

        st.write('corpus : %s' % corpus)
        st.write('word_to_id : %s' % word_to_id)
        st.write('id_to_word : %s' % id_to_word)