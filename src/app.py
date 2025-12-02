"""
혈당관리 마스터 - Streamlit 웹 애플리케이션
"""
import streamlit as st
import pandas as pd
import sys
import os

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_classifier import UserClassifier, FACTOR_MAPPING, get_user_type_info
from diet_analyzer import DietAnalyzer

# 페이지 설정
st.set_page_config(
    page_title="혈당관리 마스터",
    page_icon="🩸",
    layout="wide"
)

# 세션 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'survey_responses' not in st.session_state:
    st.session_state.survey_responses = {}


def load_survey_questions():
    """설문 문항 로드"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'survey_questions.csv')
    return pd.read_csv(data_path)


def home_page():
    """홈 페이지"""
    st.title("🩸 혈당관리 마스터")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 식습관 유형 분석")
        st.write("설문조사를 통해 나의 식습관 유형을 파악하고, 맞춤형 조언을 받아보세요.")
        if st.button("설문조사 시작", key="btn_survey", use_container_width=True):
            st.session_state.page = 'survey'
            st.rerun()

    with col2:
        st.subheader("🍽️ 식단 분석")
        st.write("오늘 먹은 음식을 입력하고, 혈당에 미치는 영향을 분석해보세요.")
        if st.button("식단 분석하기", key="btn_diet", use_container_width=True):
            st.session_state.page = 'diet'
            st.rerun()

    st.markdown("---")

    # 유형 결과가 있으면 표시
    if st.session_state.user_type:
        st.subheader("📊 내 식습관 유형")
        type_info = st.session_state.user_type
        st.success(f"**{type_info['name']}**")
        st.write(type_info['description'])

        with st.expander("맞춤 권장사항 보기"):
            for rec in type_info['recommendations']:
                st.write(f"• {rec}")


def survey_page():
    """설문조사 페이지"""
    st.title("📋 식습관 유형 분석 설문")
    st.markdown("각 문항에 대해 본인과 얼마나 일치하는지 선택해주세요.")
    st.markdown("---")

    questions = load_survey_questions()
    responses = {}

    # 척도 설명
    st.info("1: 전혀 아니다 | 2: 조금 아니다 | 3: 보통 | 4: 그렇다 | 5: 매우 그렇다")

    # 문항별 입력
    for idx, row in questions.iterrows():
        q_id = row['question_id']
        q_text = row['question_text']
        factor_name = row['factor_name']

        st.markdown(f"**{q_id}. {q_text}**")
        st.caption(f"요인: {factor_name}")

        responses[q_id] = st.slider(
            label=q_id,
            min_value=1,
            max_value=5,
            value=st.session_state.survey_responses.get(q_id, 3),
            key=f"slider_{q_id}",
            label_visibility="collapsed"
        )
        st.markdown("")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 홈으로", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()

    with col2:
        if st.button("결과 확인 →", type="primary", use_container_width=True):
            st.session_state.survey_responses = responses
            # 유형 분류
            classifier = UserClassifier()
            result = classifier.classify_user(responses)
            st.session_state.user_type = result
            st.session_state.page = 'result'
            st.rerun()


def result_page():
    """결과 페이지"""
    st.title("📊 식습관 유형 분석 결과")
    st.markdown("---")

    if not st.session_state.user_type:
        st.warning("설문을 먼저 완료해주세요.")
        if st.button("설문하러 가기"):
            st.session_state.page = 'survey'
            st.rerun()
        return

    type_info = st.session_state.user_type
    classifier = UserClassifier()

    # 유형 표시
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"🎯 당신의 유형: {type_info['name']}")
        st.write(type_info['description'])

        # 위험도 표시
        risk_colors = {
            'low': '🟢',
            'medium': '🟡',
            'medium-high': '🟠',
            'high': '🔴'
        }
        risk_labels = {
            'low': '낮음',
            'medium': '보통',
            'medium-high': '약간 높음',
            'high': '높음'
        }
        risk = type_info['risk_level']
        st.metric("혈당 위험도", f"{risk_colors.get(risk, '⚪')} {risk_labels.get(risk, '보통')}")

    with col2:
        # 위험 점수
        risk_score = classifier.get_risk_score(type_info['factor_scores'])
        st.metric("위험 점수", f"{risk_score:.0f}/100")

    st.markdown("---")

    # 요인별 점수
    st.subheader("📈 요인별 점수")
    factor_scores = type_info['factor_scores']

    cols = st.columns(5)
    factor_labels = {
        'Impulse': '충동적 섭취',
        'Social': '환경적 과식',
        'Self-control': '자기조절',
        'Stress': '스트레스 섭식',
        'Activity': '신체 활동'
    }

    for i, (factor, label) in enumerate(factor_labels.items()):
        with cols[i]:
            score = factor_scores.get(factor, 3)
            # Activity는 높을수록 좋음, 나머지는 낮을수록 좋음
            if factor == 'Activity':
                color = "normal" if score >= 3 else "inverse"
            else:
                color = "normal" if score <= 3 else "inverse"
            st.metric(label, f"{score:.1f}", delta=None)

    st.markdown("---")

    # 권장사항
    st.subheader("💡 맞춤 권장사항")
    for rec in type_info['recommendations']:
        st.write(f"✅ {rec}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← 홈으로", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    with col2:
        if st.button("다시 설문하기", use_container_width=True):
            st.session_state.survey_responses = {}
            st.session_state.page = 'survey'
            st.rerun()
    with col3:
        if st.button("식단 분석하기 →", type="primary", use_container_width=True):
            st.session_state.page = 'diet'
            st.rerun()


def diet_page():
    """식단 분석 페이지"""
    st.title("🍽️ 식단 분석")
    st.markdown("오늘 먹은 음식을 선택하고 혈당 영향을 분석해보세요.")
    st.markdown("---")

    try:
        analyzer = DietAnalyzer()
    except FileNotFoundError as e:
        st.error(f"데이터베이스 로드 실패: {e}")
        return

    # 카테고리별 음식 선택
    categories = analyzer.get_categories()
    category_labels = {
        '주식': '🍚 주식',
        '면류': '🍜 면류',
        '디저트': '🍰 디저트',
        '패스트푸드': '🍔 패스트푸드',
        '샐러드': '🥗 샐러드',
        '간식': '🍿 간식',
        '음료': '🥤 음료'
    }

    selected_foods = []

    st.subheader("음식 선택")

    # 탭으로 카테고리 구분
    tabs = st.tabs([category_labels.get(cat, cat) for cat in categories])

    for i, category in enumerate(categories):
        with tabs[i]:
            foods = analyzer.get_food_by_category(category)
            for _, food in foods.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(food['food_name'])
                with col2:
                    st.caption(f"GI: {food['gi_index']}")
                with col3:
                    st.caption(f"{food['calories']}kcal")
                with col4:
                    if st.checkbox("선택", key=f"food_{food['food_id']}", label_visibility="collapsed"):
                        selected_foods.append(food['food_id'])

    st.markdown("---")

    # 선택된 음식 표시
    if selected_foods:
        st.subheader(f"📝 선택한 음식 ({len(selected_foods)}개)")

        selected_names = []
        for fid in selected_foods:
            food = analyzer.get_food_by_id(fid)
            if food:
                selected_names.append(food['food_name'])

        st.write(", ".join(selected_names))

        if st.button("🔍 혈당 영향 분석", type="primary", use_container_width=True):
            # 영양 정보 계산
            nutrition = analyzer.calculate_meal_nutrition(selected_foods)

            # 혈당 영향 예측
            user_type_id = st.session_state.user_type.get('type_id') if st.session_state.user_type else None
            impact = analyzer.predict_glucose_impact(nutrition, user_type_id)

            # 피드백 생성
            feedback = analyzer.generate_feedback(nutrition, impact, st.session_state.user_type)

            st.markdown("---")
            st.subheader("📊 분석 결과")

            # 주요 지표
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("총 칼로리", f"{nutrition['calories']:.0f} kcal")
            with col2:
                gi_emoji = "🟢" if nutrition['gi_index'] < 55 else ("🟡" if nutrition['gi_index'] < 70 else "🔴")
                st.metric("평균 GI", f"{gi_emoji} {nutrition['gi_index']:.0f}")
            with col3:
                st.metric("혈당부하(GL)", f"{impact['glycemic_load']:.1f}")
            with col4:
                impact_emoji = "🟢" if impact['impact_score'] < 30 else ("🟡" if impact['impact_score'] < 50 else ("🟠" if impact['impact_score'] < 70 else "🔴"))
                st.metric("영향 점수", f"{impact_emoji} {impact['impact_score']:.0f}")

            # 영양소 상세
            st.markdown("---")
            st.subheader("🥗 영양 성분")

            nutr_col1, nutr_col2, nutr_col3, nutr_col4 = st.columns(4)
            with nutr_col1:
                st.metric("탄수화물", f"{nutrition['carbohydrate']:.1f}g")
                st.metric("당류", f"{nutrition['sugar']:.1f}g")
            with nutr_col2:
                st.metric("단백질", f"{nutrition['protein']:.1f}g")
                st.metric("지방", f"{nutrition['fat']:.1f}g")
            with nutr_col3:
                st.metric("식이섬유", f"{nutrition['fiber']:.1f}g")
            with nutr_col4:
                st.metric("나트륨", f"{nutrition['sodium']:.0f}mg")

            # 피드백
            st.markdown("---")
            st.subheader("💬 피드백")

            # 요약
            if impact['impact_score'] < 30:
                st.success(feedback['summary'])
            elif impact['impact_score'] < 50:
                st.info(feedback['summary'])
            elif impact['impact_score'] < 70:
                st.warning(feedback['summary'])
            else:
                st.error(feedback['summary'])

            # 긍정적 요소
            if feedback['positives']:
                st.markdown("**✅ 좋은 점:**")
                for pos in feedback['positives']:
                    st.write(f"  • {pos}")

            # 경고
            if feedback['warnings']:
                st.markdown("**⚠️ 주의사항:**")
                for warn in feedback['warnings']:
                    st.write(f"  • {warn}")

            # 제안
            if feedback['suggestions']:
                st.markdown("**💡 제안:**")
                for sug in feedback['suggestions']:
                    st.write(f"  • {sug}")

            # 대안 음식 추천
            st.markdown("---")
            st.subheader("🔄 더 나은 대안")

            for fid in selected_foods[:3]:  # 최대 3개 음식에 대해 대안 제시
                food = analyzer.get_food_by_id(fid)
                if food and food['gi_index'] >= 55:  # GI가 높은 음식만
                    alternatives = analyzer.get_alternative_foods(fid)
                    if alternatives:
                        st.write(f"**{food['food_name']}** 대신:")
                        for alt in alternatives:
                            st.write(f"  → {alt['food_name']} (GI: {alt['gi_index']}) - {alt['reason']}")

    else:
        st.info("음식을 선택해주세요.")

    st.markdown("---")
    if st.button("← 홈으로", use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()


def main():
    """메인 함수"""
    # 사이드바
    with st.sidebar:
        st.title("🩸 혈당관리 마스터")
        st.markdown("---")

        if st.button("🏠 홈", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()

        if st.button("📋 설문조사", use_container_width=True):
            st.session_state.page = 'survey'
            st.rerun()

        if st.button("🍽️ 식단 분석", use_container_width=True):
            st.session_state.page = 'diet'
            st.rerun()

        st.markdown("---")

        if st.session_state.user_type:
            st.subheader("내 유형")
            st.write(st.session_state.user_type['name'])

        st.markdown("---")
        st.caption("© 2024 혈당관리 마스터")

    # 페이지 라우팅
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'survey':
        survey_page()
    elif st.session_state.page == 'result':
        result_page()
    elif st.session_state.page == 'diet':
        diet_page()


if __name__ == "__main__":
    main()
