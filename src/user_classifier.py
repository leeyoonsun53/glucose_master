"""
사용자 유형 분류 모델
설문조사 결과를 바탕으로 사용자의 식습관 유형을 분류합니다.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 요인별 문항 매핑
FACTOR_MAPPING = {
    'Impulse': ['Q3', 'Q4', 'Q5', 'Q7'],      # 충동적 섭취
    'Social': ['Q6', 'Q8'],                    # 환경적 과식
    'Self-control': ['Q1'],                    # 자기조절
    'Stress': ['Q2'],                          # 스트레스 섭식
    'Activity': ['Q9', 'Q10']                  # 신체 활동
}

# 유형 정의 (주유형 + 부유형 조합)
USER_TYPES = {
    'impulse_low_activity': {
        'name': '충동형-저활동',
        'description': '감정에 따른 충동적 섭취가 많고 신체 활동이 적은 유형',
        'risk_level': 'high',
        'recommendations': [
            '식사 전 5분 대기 규칙 실천',
            '감정 일기 작성으로 감정 인식',
            '가벼운 산책부터 시작하는 운동 습관',
            '건강한 간식 미리 준비'
        ]
    },
    'impulse_active': {
        'name': '충동형-활동적',
        'description': '충동적 섭취 경향이 있으나 활동량으로 일부 상쇄하는 유형',
        'risk_level': 'medium',
        'recommendations': [
            '운동 전후 적절한 영양 섭취',
            '충동적 간식 대신 단백질 위주 선택',
            '규칙적인 식사 시간 유지'
        ]
    },
    'social_low_control': {
        'name': '환경형-저조절',
        'description': '사회적 상황에서 과식하며 자기조절이 어려운 유형',
        'risk_level': 'high',
        'recommendations': [
            '외식 전 건강한 간식 섭취',
            '메뉴 미리 결정하기',
            '작은 접시 사용 습관',
            '주간 식단 계획 세우기'
        ]
    },
    'social_controlled': {
        'name': '환경형-조절형',
        'description': '사회적 상황에서 과식하나 평소 자기조절이 되는 유형',
        'risk_level': 'medium',
        'recommendations': [
            '외식 빈도 조절',
            '사회적 상황에서의 식사량 인식',
            '동반자와 건강한 메뉴 선택 공유'
        ]
    },
    'stress_eater': {
        'name': '스트레스형',
        'description': '스트레스 상황에서 음식으로 해소하려는 유형',
        'risk_level': 'medium-high',
        'recommendations': [
            '스트레스 해소를 위한 대안 활동 찾기',
            '명상이나 심호흡 연습',
            '규칙적인 수면 습관',
            '카페인과 당분 섭취 줄이기'
        ]
    },
    'balanced': {
        'name': '균형형',
        'description': '전반적으로 균형 잡힌 식습관을 가진 유형',
        'risk_level': 'low',
        'recommendations': [
            '현재 좋은 습관 유지',
            '다양한 영양소 균형 섭취',
            '정기적인 건강 체크'
        ]
    }
}


class UserClassifier:
    def __init__(self, model_path=None):
        self.scaler = StandardScaler()
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), '..', 'models', 'user_classifier.joblib'
        )

    def calculate_factor_scores(self, responses: dict) -> dict:
        """설문 응답으로부터 요인별 점수 계산"""
        factor_scores = {}
        for factor, questions in FACTOR_MAPPING.items():
            scores = [responses.get(q, 3) for q in questions]
            factor_scores[factor] = np.mean(scores)
        return factor_scores

    def classify_user(self, responses: dict) -> dict:
        """사용자 유형 분류"""
        factor_scores = self.calculate_factor_scores(responses)

        # 주유형 결정 (가장 높은 부정적 요인)
        impulse_score = factor_scores['Impulse']
        social_score = factor_scores['Social']
        stress_score = factor_scores['Stress']
        control_score = factor_scores['Self-control']
        activity_score = factor_scores['Activity']

        # Activity는 역산 (높을수록 좋음, Q9, Q10은 긍정문)
        # 나머지는 높을수록 문제

        primary_issues = {
            'impulse': impulse_score,
            'social': social_score,
            'stress': stress_score
        }

        max_issue = max(primary_issues, key=primary_issues.get)
        max_score = primary_issues[max_issue]

        # 유형 결정 로직
        if max_score < 2.5:  # 모든 부정적 요인이 낮음
            user_type = 'balanced'
        elif max_issue == 'impulse':
            if activity_score >= 3.5:
                user_type = 'impulse_active'
            else:
                user_type = 'impulse_low_activity'
        elif max_issue == 'social':
            if control_score >= 3.5:  # 자기조절 어려움이 높음 = 조절 안됨
                user_type = 'social_low_control'
            else:
                user_type = 'social_controlled'
        else:  # stress
            user_type = 'stress_eater'

        type_info = USER_TYPES[user_type].copy()
        type_info['type_id'] = user_type
        type_info['factor_scores'] = factor_scores

        return type_info

    def get_risk_score(self, factor_scores: dict) -> float:
        """혈당 위험도 점수 계산 (0-100)"""
        # 가중치 적용
        weights = {
            'Impulse': 0.3,
            'Social': 0.2,
            'Self-control': 0.2,
            'Stress': 0.15,
            'Activity': -0.15  # 활동량은 역방향
        }

        score = 0
        for factor, weight in weights.items():
            factor_value = factor_scores.get(factor, 3)
            if weight > 0:
                score += weight * (factor_value / 5) * 100
            else:
                score += abs(weight) * ((5 - factor_value) / 5) * 100

        return min(max(score, 0), 100)

    def generate_training_data(self, n_samples=500) -> pd.DataFrame:
        """학습용 샘플 데이터 생성"""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            # 다양한 유형의 데이터 생성
            profile = np.random.choice(['impulse', 'social', 'stress', 'balanced', 'active'])

            if profile == 'impulse':
                base = {'Q1': 3, 'Q2': 3, 'Q3': 4, 'Q4': 4, 'Q5': 4,
                       'Q6': 3, 'Q7': 4, 'Q8': 3, 'Q9': 2, 'Q10': 2}
            elif profile == 'social':
                base = {'Q1': 3, 'Q2': 3, 'Q3': 3, 'Q4': 3, 'Q5': 3,
                       'Q6': 5, 'Q7': 3, 'Q8': 4, 'Q9': 3, 'Q10': 3}
            elif profile == 'stress':
                base = {'Q1': 3, 'Q2': 5, 'Q3': 3, 'Q4': 3, 'Q5': 4,
                       'Q6': 3, 'Q7': 3, 'Q8': 3, 'Q9': 2, 'Q10': 2}
            elif profile == 'balanced':
                base = {'Q1': 2, 'Q2': 2, 'Q3': 2, 'Q4': 2, 'Q5': 2,
                       'Q6': 2, 'Q7': 2, 'Q8': 2, 'Q9': 4, 'Q10': 4}
            else:  # active
                base = {'Q1': 3, 'Q2': 3, 'Q3': 3, 'Q4': 3, 'Q5': 3,
                       'Q6': 3, 'Q7': 3, 'Q8': 3, 'Q9': 5, 'Q10': 5}

            # 노이즈 추가
            row = {}
            for q, v in base.items():
                noise = np.random.randint(-1, 2)
                row[q] = max(1, min(5, v + noise))

            # 유형 분류 결과 추가
            result = self.classify_user(row)
            row['user_type'] = result['type_id']
            row['risk_score'] = self.get_risk_score(result['factor_scores'])

            data.append(row)

        return pd.DataFrame(data)

    def save_model(self, model_data: dict):
        """모델 저장"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model_data, self.model_path)

    def load_model(self) -> dict:
        """모델 로드"""
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None


def get_user_type_info(type_id: str) -> dict:
    """유형 ID로 유형 정보 조회"""
    return USER_TYPES.get(type_id, USER_TYPES['balanced'])
