"""
혈당관리 마스터 - Streamlit 웹 애플리케이션
"""
import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import uuid

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_classifier import UserClassifier, FACTOR_MAPPING, get_user_type_info
from diet_analyzer import DietAnalyzer


def get_data_path(filename):
    """데이터 파일 경로 반환"""
    return os.path.join(os.path.dirname(__file__), '..', 'data', filename)


def save_survey_response(user_id, responses, result):
    """설문조사 응답을 CSV 파일에 저장"""
    csv_path = get_data_path('survey_responses.csv')

    # 새 레코드 생성
    record = {
        'response_id': str(uuid.uuid4())[:8],
        'user_id': user_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_type': result['type_id'],
        'user_type_name': result['name'],
        'risk_level': result['risk_level'],
        'factor_impulse': result['factor_scores'].get('Impulse', 0),
        'factor_social': result['factor_scores'].get('Social', 0),
        'factor_self_control': result['factor_scores'].get('Self-control', 0),
        'factor_stress': result['factor_scores'].get('Stress', 0),
        'factor_activity': result['factor_scores'].get('Activity', 0),
    }

    # 각 질문 응답 추가
    for q_id, value in responses.items():
        record[q_id] = value

    # DataFrame 생성
    new_df = pd.DataFrame([record])

    # 기존 파일이 있으면 추가, 없으면 새로 생성
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return record['response_id']


def save_diet_record(user_id, selected_foods, nutrition, impact, food_names):
    """식단 기록을 CSV 파일에 저장"""
    csv_path = get_data_path('diet_records.csv')

    # 새 레코드 생성
    record = {
        'record_id': str(uuid.uuid4())[:8],
        'user_id': user_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'food_ids': '|'.join(map(str, selected_foods)),
        'food_names': '|'.join(food_names),
        'food_count': len(selected_foods),
        'total_calories': nutrition['calories'],
        'avg_gi_index': nutrition['gi_index'],
        'total_carbohydrate': nutrition['carbohydrate'],
        'total_protein': nutrition['protein'],
        'total_fat': nutrition['fat'],
        'total_sugar': nutrition['sugar'],
        'total_fiber': nutrition['fiber'],
        'total_sodium': nutrition['sodium'],
        'glycemic_load': impact['glycemic_load'],
        'impact_score': impact['impact_score'],
        'impact_level': impact['level']
    }

    # DataFrame 생성
    new_df = pd.DataFrame([record])

    # 기존 파일이 있으면 추가, 없으면 새로 생성
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return record['record_id']

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
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]


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

            # 설문 응답 저장
            save_survey_response(st.session_state.user_id, responses, result)

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

            # 식단 기록 저장
            save_diet_record(
                st.session_state.user_id,
                selected_foods,
                nutrition,
                impact,
                selected_names
            )

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


def show_detail_view(survey_path, diet_path, has_survey_data, has_diet_data):
    """상세 보기 페이지"""
    view_type = st.session_state.detail_view
    filter_value = st.session_state.detail_filter

    # 뒤로가기 버튼
    if st.button("← 대시보드로 돌아가기", use_container_width=True):
        st.session_state.detail_view = None
        st.session_state.detail_filter = None
        st.rerun()

    st.markdown("---")

    risk_labels = {'low': '낮음', 'medium': '보통', 'medium-high': '약간 높음', 'high': '높음'}
    impact_labels = {'low': '낮음', 'medium': '보통', 'medium-high': '약간 높음', 'high': '높음'}

    # 위험 수준별 상세
    if view_type == 'risk_level' and has_survey_data:
        survey_df = pd.read_csv(survey_path)
        filtered = survey_df[survey_df['risk_level'] == filter_value]

        st.title(f"⚠️ 위험 수준: {risk_labels.get(filter_value, filter_value)}")
        st.markdown(f"### 총 {len(filtered)}명의 고객")
        st.markdown("---")

        # 요약 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("평균 충동적 섭취", f"{filtered['factor_impulse'].mean():.1f}")
        with col2:
            st.metric("평균 스트레스 섭식", f"{filtered['factor_stress'].mean():.1f}")
        with col3:
            st.metric("평균 신체 활동", f"{filtered['factor_activity'].mean():.1f}")

        st.markdown("---")
        st.subheader("📋 고객 목록")

        # 테이블로 표시
        display_df = filtered[['user_id', 'timestamp', 'user_type_name', 'factor_impulse', 'factor_stress', 'factor_activity']].copy()
        display_df.columns = ['고객 ID', '응답 일시', '유형', '충동적 섭취', '스트레스 섭식', '신체 활동']
        display_df = display_df.sort_values('응답 일시', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 혈당 영향 수준별 상세
    elif view_type == 'impact_level' and has_diet_data:
        diet_df = pd.read_csv(diet_path)
        filtered = diet_df[diet_df['impact_level'] == filter_value]

        st.title(f"🩸 혈당 영향: {impact_labels.get(filter_value, filter_value)}")
        st.markdown(f"### 총 {len(filtered)}건의 식단 기록")
        st.markdown("---")

        # 요약 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("평균 칼로리", f"{filtered['total_calories'].mean():.0f}kcal")
        with col2:
            st.metric("평균 GI", f"{filtered['avg_gi_index'].mean():.1f}")
        with col3:
            st.metric("평균 탄수화물", f"{filtered['total_carbohydrate'].mean():.1f}g")
        with col4:
            st.metric("평균 영향 점수", f"{filtered['impact_score'].mean():.1f}")

        st.markdown("---")
        st.subheader("📋 식단 기록 목록")

        for _, row in filtered.sort_values('timestamp', ascending=False).iterrows():
            with st.expander(f"🍽️ {row['food_names'].replace('|', ', ')} - {row['timestamp'][:10]}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**고객:** {row['user_id']}")
                    st.write(f"**음식:** {row['food_names'].replace('|', ', ')}")
                    st.write(f"**음식 수:** {row['food_count']}개")
                with col2:
                    st.write(f"**칼로리:** {row['total_calories']:.0f}kcal")
                    st.write(f"**GI 지수:** {row['avg_gi_index']:.1f}")
                    st.write(f"**영향 점수:** {row['impact_score']:.1f}")

                st.markdown("**영양 성분:**")
                st.write(f"탄수화물: {row['total_carbohydrate']:.1f}g | 단백질: {row['total_protein']:.1f}g | 지방: {row['total_fat']:.1f}g | 당류: {row['total_sugar']:.1f}g | 식이섬유: {row['total_fiber']:.1f}g")

    # 유형별 상세
    elif view_type == 'user_type' and has_survey_data:
        survey_df = pd.read_csv(survey_path)
        filtered = survey_df[survey_df['user_type_name'] == filter_value]

        st.title(f"👥 유형: {filter_value}")
        st.markdown(f"### 총 {len(filtered)}명의 고객")
        st.markdown("---")

        # 유형 특성 요약
        st.subheader("📊 유형 특성 평균")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("충동적 섭취", f"{filtered['factor_impulse'].mean():.1f}")
        with col2:
            st.metric("환경적 과식", f"{filtered['factor_social'].mean():.1f}")
        with col3:
            st.metric("자기조절", f"{filtered['factor_self_control'].mean():.1f}")
        with col4:
            st.metric("스트레스 섭식", f"{filtered['factor_stress'].mean():.1f}")
        with col5:
            st.metric("신체 활동", f"{filtered['factor_activity'].mean():.1f}")

        st.markdown("---")
        st.subheader("📋 고객 목록")

        for _, row in filtered.sort_values('timestamp', ascending=False).iterrows():
            with st.expander(f"👤 {row['user_id']} - {row['timestamp'][:10]}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**위험 수준:** {risk_labels.get(row['risk_level'], row['risk_level'])}")
                    st.write(f"**응답 일시:** {row['timestamp']}")
                with col2:
                    st.write(f"**충동적 섭취:** {row['factor_impulse']:.1f}")
                    st.write(f"**스트레스 섭식:** {row['factor_stress']:.1f}")
                    st.write(f"**신체 활동:** {row['factor_activity']:.1f}")

    # 특정 음식 포함 식단
    elif view_type == 'food' and has_diet_data:
        diet_df = pd.read_csv(diet_path)
        filtered = diet_df[diet_df['food_names'].str.contains(filter_value, na=False)]

        st.title(f"🍽️ '{filter_value}' 포함 식단")
        st.markdown(f"### 총 {len(filtered)}건의 기록")
        st.markdown("---")

        # 통계
        if len(filtered) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 칼로리", f"{filtered['total_calories'].mean():.0f}kcal")
            with col2:
                st.metric("평균 GI", f"{filtered['avg_gi_index'].mean():.1f}")
            with col3:
                st.metric("평균 영향 점수", f"{filtered['impact_score'].mean():.1f}")

            st.markdown("---")

            # 함께 많이 먹는 음식
            st.subheader("🔗 함께 많이 선택되는 음식")
            paired_foods = []
            for _, row in filtered.iterrows():
                foods = row['food_names'].split('|')
                for f in foods:
                    if f != filter_value:
                        paired_foods.append(f)

            if paired_foods:
                paired_counts = pd.Series(paired_foods).value_counts().head(5)
                for food, count in paired_counts.items():
                    st.write(f"• {food}: {count}회")

            st.markdown("---")
            st.subheader("📋 식단 기록")

            for _, row in filtered.sort_values('timestamp', ascending=False).iterrows():
                impact_emoji = {'low': '🟢', 'medium': '🟡', 'medium-high': '🟠', 'high': '🔴'}
                with st.expander(f"{impact_emoji.get(row['impact_level'], '⚪')} {row['food_names'].replace('|', ' + ')} - {row['timestamp'][:10]}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**고객:** {row['user_id']}")
                        st.write(f"**칼로리:** {row['total_calories']:.0f}kcal")
                    with col2:
                        st.write(f"**GI 지수:** {row['avg_gi_index']:.1f}")
                        st.write(f"**영향 점수:** {row['impact_score']:.1f}")

    # 특정 조합 상세
    elif view_type == 'combo' and has_diet_data:
        diet_df = pd.read_csv(diet_path)
        filtered = diet_df[diet_df['food_names'] == filter_value]

        foods_display = filter_value.replace('|', ' + ')
        st.title(f"🍱 식사 조합 상세")
        st.markdown(f"### {foods_display}")
        st.markdown(f"**총 {len(filtered)}회 선택됨**")
        st.markdown("---")

        if len(filtered) > 0:
            # 영양 정보
            st.subheader("📊 영양 정보")
            row = filtered.iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("칼로리", f"{row['total_calories']:.0f}kcal")
            with col2:
                gi_emoji = "🟢" if row['avg_gi_index'] < 55 else ("🟡" if row['avg_gi_index'] < 70 else "🔴")
                st.metric("GI 지수", f"{gi_emoji} {row['avg_gi_index']:.1f}")
            with col3:
                st.metric("혈당 부하", f"{row['glycemic_load']:.1f}")
            with col4:
                impact_emoji = {'low': '🟢', 'medium': '🟡', 'medium-high': '🟠', 'high': '🔴'}
                st.metric("영향 점수", f"{impact_emoji.get(row['impact_level'], '⚪')} {row['impact_score']:.1f}")

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🥗 영양 성분")
                st.write(f"• 탄수화물: {row['total_carbohydrate']:.1f}g")
                st.write(f"• 단백질: {row['total_protein']:.1f}g")
                st.write(f"• 지방: {row['total_fat']:.1f}g")
            with col2:
                st.subheader("📝 기타")
                st.write(f"• 당류: {row['total_sugar']:.1f}g")
                st.write(f"• 식이섬유: {row['total_fiber']:.1f}g")
                st.write(f"• 나트륨: {row['total_sodium']:.0f}mg")

            st.markdown("---")
            st.subheader("📅 선택 이력")
            for _, r in filtered.sort_values('timestamp', ascending=False).iterrows():
                st.write(f"• {r['user_id']} - {r['timestamp']}")

    else:
        st.warning("데이터를 찾을 수 없습니다.")


def dashboard_page():
    """대시보드 페이지 - 통계 및 분석"""

    # 상세 보기 세션 상태 초기화
    if 'detail_view' not in st.session_state:
        st.session_state.detail_view = None
    if 'detail_filter' not in st.session_state:
        st.session_state.detail_filter = None

    # 데이터 로드
    survey_path = get_data_path('survey_responses.csv')
    diet_path = get_data_path('diet_records.csv')

    has_survey_data = os.path.exists(survey_path)
    has_diet_data = os.path.exists(diet_path)

    # 상세 보기 모드인 경우
    if st.session_state.detail_view:
        show_detail_view(survey_path, diet_path, has_survey_data, has_diet_data)
        return

    st.title("📊 대시보드")
    st.markdown("고객 데이터 통계 및 분석 현황을 확인하세요.")
    st.markdown("---")

    if not has_survey_data and not has_diet_data:
        st.warning("아직 수집된 데이터가 없습니다. 설문조사와 식단 분석을 먼저 진행해주세요.")
        if st.button("← 홈으로", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        return

    # 탭으로 구분
    tab1, tab2, tab3 = st.tabs(["📈 전체 현황", "👥 유형 분석", "🍽️ 식단 베스트"])

    with tab1:
        st.subheader("📈 전체 현황")

        col1, col2, col3, col4 = st.columns(4)

        # 설문 통계
        if has_survey_data:
            survey_df = pd.read_csv(survey_path)
            with col1:
                st.metric("총 설문 응답", f"{len(survey_df)}건")
            with col2:
                unique_users = survey_df['user_id'].nunique()
                st.metric("참여 고객 수", f"{unique_users}명")
        else:
            with col1:
                st.metric("총 설문 응답", "0건")
            with col2:
                st.metric("참여 고객 수", "0명")

        # 식단 통계
        if has_diet_data:
            diet_df = pd.read_csv(diet_path)
            with col3:
                st.metric("총 식단 기록", f"{len(diet_df)}건")
            with col4:
                avg_calories = diet_df['total_calories'].mean()
                st.metric("평균 칼로리", f"{avg_calories:.0f}kcal")
        else:
            with col3:
                st.metric("총 식단 기록", "0건")
            with col4:
                st.metric("평균 칼로리", "-")

        st.markdown("---")

        # 위험도 분포
        if has_survey_data:
            st.subheader("⚠️ 위험 수준 분포 (클릭하여 상세보기)")
            risk_counts = survey_df['risk_level'].value_counts()
            risk_order = ['low', 'medium', 'medium-high', 'high']
            risk_labels = {'low': '낮음', 'medium': '보통', 'medium-high': '약간 높음', 'high': '높음'}

            cols = st.columns(4)
            for i, level in enumerate(risk_order):
                with cols[i]:
                    count = risk_counts.get(level, 0)
                    pct = (count / len(survey_df) * 100) if len(survey_df) > 0 else 0
                    if st.button(f"**{risk_labels[level]}**\n{count}명 ({pct:.1f}%)", key=f"risk_{level}", use_container_width=True):
                        st.session_state.detail_view = 'risk_level'
                        st.session_state.detail_filter = level
                        st.rerun()

        st.markdown("---")

        # 혈당 영향 수준 분포
        if has_diet_data:
            st.subheader("🩸 식단 혈당 영향 분포 (클릭하여 상세보기)")
            impact_counts = diet_df['impact_level'].value_counts()
            impact_order = ['low', 'medium', 'medium-high', 'high']
            impact_labels = {'low': '낮음', 'medium': '보통', 'medium-high': '약간 높음', 'high': '높음'}

            cols = st.columns(4)
            for i, level in enumerate(impact_order):
                with cols[i]:
                    count = impact_counts.get(level, 0)
                    pct = (count / len(diet_df) * 100) if len(diet_df) > 0 else 0
                    if st.button(f"**{impact_labels[level]}**\n{count}건 ({pct:.1f}%)", key=f"impact_{level}", use_container_width=True):
                        st.session_state.detail_view = 'impact_level'
                        st.session_state.detail_filter = level
                        st.rerun()

    with tab2:
        st.subheader("👥 식습관 유형 분석")

        if has_survey_data:
            survey_df = pd.read_csv(survey_path)

            # 유형별 분포
            st.markdown("### 🏆 유형 베스트 (클릭하여 상세보기)")
            type_counts = survey_df['user_type_name'].value_counts()

            for i, (type_name, count) in enumerate(type_counts.items(), 1):
                pct = count / len(survey_df) * 100
                col1, col2, col3 = st.columns([2.5, 1, 0.5])
                with col1:
                    st.progress(pct / 100, text=f"{i}위. {type_name}")
                with col2:
                    st.write(f"**{count}명** ({pct:.1f}%)")
                with col3:
                    if st.button("→", key=f"type_{type_name}", help=f"{type_name} 상세보기"):
                        st.session_state.detail_view = 'user_type'
                        st.session_state.detail_filter = type_name
                        st.rerun()

            st.markdown("---")

            # 요인별 평균 점수
            st.markdown("### 📊 요인별 평균 점수")
            factor_cols = {
                'factor_impulse': '충동적 섭취',
                'factor_social': '환경적 과식',
                'factor_self_control': '자기조절 어려움',
                'factor_stress': '스트레스 섭식',
                'factor_activity': '신체 활동'
            }

            cols = st.columns(5)
            for i, (col_name, label) in enumerate(factor_cols.items()):
                with cols[i]:
                    avg_score = survey_df[col_name].mean()
                    # Activity는 높을수록 좋음
                    if col_name == 'factor_activity':
                        status = "🟢" if avg_score >= 3.5 else ("🟡" if avg_score >= 2.5 else "🔴")
                    else:
                        status = "🟢" if avg_score <= 2.5 else ("🟡" if avg_score <= 3.5 else "🔴")
                    st.metric(label, f"{status} {avg_score:.1f}")

            st.markdown("---")

            # 최근 응답 현황
            st.markdown("### 📅 최근 설문 응답")
            recent = survey_df.sort_values('timestamp', ascending=False).head(5)
            for _, row in recent.iterrows():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"👤 {row['user_id']}")
                with col2:
                    st.write(f"🏷️ {row['user_type_name']}")
                with col3:
                    st.caption(row['timestamp'][:10])
        else:
            st.info("설문조사 데이터가 없습니다.")

    with tab3:
        st.subheader("🍽️ 식단 메뉴 베스트 20")

        if has_diet_data:
            diet_df = pd.read_csv(diet_path)

            # 모든 음식 추출 및 집계
            all_foods = []
            for _, row in diet_df.iterrows():
                foods = row['food_names'].split('|')
                all_foods.extend(foods)

            food_counts = pd.Series(all_foods).value_counts().head(20)

            st.markdown("### 🏆 가장 많이 선택된 음식 TOP 20 (클릭하여 상세보기)")

            for i, (food_name, count) in enumerate(food_counts.items(), 1):
                col1, col2, col3, col4 = st.columns([0.5, 2.5, 0.7, 0.3])
                with col1:
                    if i <= 3:
                        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                        st.write(medals[i])
                    else:
                        st.write(f"**{i}**")
                with col2:
                    pct = count / len(diet_df) * 100
                    st.progress(min(pct / 50, 1.0), text=food_name)
                with col3:
                    st.write(f"**{count}회**")
                with col4:
                    if st.button("→", key=f"food_{i}_{food_name}", help=f"{food_name} 포함 식단 보기"):
                        st.session_state.detail_view = 'food'
                        st.session_state.detail_filter = food_name
                        st.rerun()

            st.markdown("---")

            # 식사 조합 베스트
            st.markdown("### 🍱 인기 식사 조합 TOP 10 (클릭하여 상세보기)")
            combo_counts = diet_df['food_names'].value_counts().head(10)

            for i, (combo, count) in enumerate(combo_counts.items(), 1):
                foods = combo.replace('|', ' + ')
                col1, col2, col3 = st.columns([3.5, 0.7, 0.3])
                with col1:
                    st.write(f"**{i}.** {foods}")
                with col2:
                    st.write(f"{count}회")
                with col3:
                    if st.button("→", key=f"combo_{i}", help="이 조합 상세보기"):
                        st.session_state.detail_view = 'combo'
                        st.session_state.detail_filter = combo
                        st.rerun()

            st.markdown("---")

            # GI 지수별 통계
            st.markdown("### 📈 GI 지수 통계")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_gi = diet_df['avg_gi_index'].mean()
                st.metric("평균 GI", f"{avg_gi:.1f}")
            with col2:
                low_gi_count = len(diet_df[diet_df['avg_gi_index'] < 55])
                st.metric("저GI 식단", f"{low_gi_count}건")
            with col3:
                high_gi_count = len(diet_df[diet_df['avg_gi_index'] >= 70])
                st.metric("고GI 식단", f"{high_gi_count}건")

            st.markdown("---")

            # 최근 식단 기록
            st.markdown("### 📅 최근 식단 기록")
            recent_diet = diet_df.sort_values('timestamp', ascending=False).head(5)
            for _, row in recent_diet.iterrows():
                foods = row['food_names'].replace('|', ', ')
                impact_emoji = {'low': '🟢', 'medium': '🟡', 'medium-high': '🟠', 'high': '🔴'}
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"🍽️ {foods}")
                with col2:
                    st.write(f"{impact_emoji.get(row['impact_level'], '⚪')} {row['impact_score']:.0f}점")
                with col3:
                    st.caption(row['timestamp'][:10])
        else:
            st.info("식단 기록 데이터가 없습니다.")

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

        if st.button("📊 대시보드", use_container_width=True):
            st.session_state.page = 'dashboard'
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
    elif st.session_state.page == 'dashboard':
        dashboard_page()


if __name__ == "__main__":
    main()
