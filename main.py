import numpy as np
import streamlit as st
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi
import plotly.express as px

st.title("밑바닥부터 시작하는 딥러닝2")

CH02 = st.checkbox('CH02')

if CH02 :
    text = st.text_input('영어문장을 입력해주세요.')
    if text :
        corpus, word_to_id, id_to_word = preprocess(text)

        st.write('corpus : %s' % corpus)
        st.write('word_to_id : %s' % word_to_id)
        st.write('id_to_word : %s' % id_to_word)

        vocab_size = len(word_to_id)
        C = create_co_matrix(corpus, vocab_size)
        
        st.write('동시발생 행렬')
        st.write(C)

        c0 = st.text_input('벡터간 유사도를 구하고 싶은 단어1을 입력하세요!')
        c1 = st.text_input('벡터간 유사도를 구하고 싶은 단어2을 입력하세요!')

        if c0 and c1:
            c0 = C[word_to_id[c0]]
            c1 = C[word_to_id[c1]]

            st.write('벡터간 코사인 유사도 : %s' % cos_similarity(c0, c1))

        ms = st.text_input('유사 단어 랭킹을 알고싶은 단어를 입력하세요.')

        if ms:
            st.write(most_similar(ms, word_to_id, id_to_word, C))

        W = ppmi(C)

        np.set_printoptions(precision=3)
        st.write('양의 상호정보량(PPMI)')
        st.write(W)

        st.write('SVD 계산... 후 2차원으로 변경 중')
        try:
            # 빠른 code
            from sklearn.utils.extmath import randomized_svd

            wordvec_size = 2
            U, S, V = randomized_svd(
                W, n_components=wordvec_size, n_iter=5, random_state=None)
        except ImportError:
            # 느린 code
            U, S, V = np.linalg.svd(W)

        fig = px.scatter(x = U[:, 0], y = U[:, 1])
        for word, word_id in word_to_id.items():
            fig.add_annotation(text = word,
                               x = U[word_id, 0],
                               y = U[word_id, 1])
        st.plotly_chart(fig)