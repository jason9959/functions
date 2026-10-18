"""동적 포트폴리오 백테스트 화면 진입점."""

import streamlit as st


def render(page: str) -> None:
    if page != "dynamic_portfolio_conditions":
        st.session_state["current_page"] = "dynamic_portfolio_conditions"
        st.rerun()
        return

    st.title("🔄 동적 포트폴리오 백테스트")
    st.markdown(
        "<p class='step-caption'>시장 조건에 따라 자산 비중을 변경하는 전략을 구성합니다.</p>",
        unsafe_allow_html=True,
    )
    st.info("입력 조건과 계산 규칙은 다음 작업에서 연결할 예정입니다.")
    if st.button("뒤로", key="back_dynamic_portfolio", use_container_width=True):
        st.session_state["current_page"] = "feature"
        st.rerun()

