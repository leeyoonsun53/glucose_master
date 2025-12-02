"""
샘플 데이터 생성 스크립트
설문조사 응답 50개, 식단 기록 50개 생성
"""
import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime, timedelta
import random

# 경로 설정
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# 유형별 응답 패턴 정의
USER_TYPE_PATTERNS = {
    'balanced': {
        'name': '균형잡힌 식습관형',
        'risk_level': 'low',
        'Q1': (2, 3), 'Q2': (1, 3), 'Q3': (1, 3), 'Q4': (2, 3), 'Q5': (1, 3),
        'Q6': (2, 3), 'Q7': (1, 3), 'Q8': (2, 3), 'Q9': (4, 5), 'Q10': (4, 5)
    },
    'stress_eater': {
        'name': '스트레스성 섭취형',
        'risk_level': 'medium-high',
        'Q1': (3, 4), 'Q2': (4, 5), 'Q3': (3, 4), 'Q4': (3, 5), 'Q5': (4, 5),
        'Q6': (3, 4), 'Q7': (3, 5), 'Q8': (3, 4), 'Q9': (2, 3), 'Q10': (2, 3)
    },
    'impulse_eater': {
        'name': '충동적 섭취형',
        'risk_level': 'high',
        'Q1': (4, 5), 'Q2': (3, 4), 'Q3': (4, 5), 'Q4': (4, 5), 'Q5': (4, 5),
        'Q6': (3, 5), 'Q7': (4, 5), 'Q8': (4, 5), 'Q9': (1, 3), 'Q10': (2, 3)
    },
    'social_eater': {
        'name': '환경영향형',
        'risk_level': 'medium',
        'Q1': (3, 4), 'Q2': (2, 3), 'Q3': (2, 4), 'Q4': (3, 4), 'Q5': (2, 4),
        'Q6': (4, 5), 'Q7': (3, 4), 'Q8': (4, 5), 'Q9': (2, 4), 'Q10': (2, 4)
    },
    'inactive': {
        'name': '활동부족형',
        'risk_level': 'medium',
        'Q1': (2, 4), 'Q2': (2, 4), 'Q3': (2, 4), 'Q4': (3, 4), 'Q5': (2, 4),
        'Q6': (2, 4), 'Q7': (2, 4), 'Q8': (3, 4), 'Q9': (1, 2), 'Q10': (1, 2)
    }
}

# 식사 패턴별 음식 조합 정의
MEAL_PATTERNS = {
    'healthy_breakfast': {
        'foods': [(2, '현미밥'), (43, '닭가슴살샐러드'), (70, '요거트(플레인)')],
        'impact_level': 'low'
    },
    'regular_lunch': {
        'foods': [(1, '흰쌀밥'), (92, '비빔밥')],
        'impact_level': 'medium'
    },
    'fast_food': {
        'foods': [(31, '햄버거(불고기)'), (34, '감자튀김(중)'), (71, '콜라')],
        'impact_level': 'high'
    },
    'noodle_meal': {
        'foods': [(11, '자장면'), (72, '사이다')],
        'impact_level': 'high'
    },
    'healthy_dinner': {
        'foods': [(4, '잡곡밥'), (44, '연어샐러드'), (86, '녹차')],
        'impact_level': 'low'
    },
    'snack_time': {
        'foods': [(53, '견과류믹스'), (64, '과일(사과)')],
        'impact_level': 'low'
    },
    'dessert': {
        'foods': [(21, '초콜릿 케이크'), (78, '카페라떼')],
        'impact_level': 'high'
    },
    'korean_meal': {
        'foods': [(91, '김밥'), (15, '우동')],
        'impact_level': 'medium'
    },
    'light_meal': {
        'foods': [(5, '오트밀'), (65, '과일(바나나)'), (90, '두유(무가당)')],
        'impact_level': 'low'
    },
    'heavy_meal': {
        'foods': [(17, '라면'), (58, '떡볶이'), (71, '콜라')],
        'impact_level': 'high'
    }
}

# 음식 데이터 로드
food_df = pd.read_csv(os.path.join(DATA_DIR, 'food_database.csv'))


def generate_survey_responses(n=50):
    """설문조사 응답 샘플 데이터 생성"""
    records = []
    types = list(USER_TYPE_PATTERNS.keys())

    base_date = datetime.now() - timedelta(days=60)

    for i in range(n):
        user_type = random.choice(types)
        pattern = USER_TYPE_PATTERNS[user_type]

        # 응답 생성
        responses = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']:
            min_val, max_val = pattern[q]
            responses[q] = random.randint(min_val, max_val)

        # 요인별 점수 계산
        factor_impulse = np.mean([responses['Q3'], responses['Q4'], responses['Q5'], responses['Q7']])
        factor_social = np.mean([responses['Q6'], responses['Q8']])
        factor_self_control = responses['Q1']
        factor_stress = responses['Q2']
        factor_activity = np.mean([responses['Q9'], responses['Q10']])

        record = {
            'response_id': str(uuid.uuid4())[:8],
            'user_id': f'user_{i+1:03d}',
            'timestamp': (base_date + timedelta(days=random.randint(0, 60),
                                                 hours=random.randint(8, 22),
                                                 minutes=random.randint(0, 59))).strftime('%Y-%m-%d %H:%M:%S'),
            'user_type': user_type,
            'user_type_name': pattern['name'],
            'risk_level': pattern['risk_level'],
            'factor_impulse': round(factor_impulse, 2),
            'factor_social': round(factor_social, 2),
            'factor_self_control': factor_self_control,
            'factor_stress': factor_stress,
            'factor_activity': round(factor_activity, 2),
        }

        # 질문별 응답 추가
        for q, val in responses.items():
            record[q] = val

        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def calculate_nutrition(food_ids):
    """선택된 음식의 영양 정보 계산"""
    selected = food_df[food_df['food_id'].isin(food_ids)]

    if selected.empty:
        return None

    nutrition = {
        'calories': selected['calories'].sum(),
        'gi_index': selected['gi_index'].mean(),
        'carbohydrate': selected['carbohydrate'].sum(),
        'protein': selected['protein'].sum(),
        'fat': selected['fat'].sum(),
        'sugar': selected['sugar'].sum(),
        'fiber': selected['fiber'].sum(),
        'sodium': selected['sodium'].sum()
    }

    return nutrition


def calculate_impact(nutrition):
    """혈당 영향 계산"""
    gi = nutrition['gi_index']
    carb = nutrition['carbohydrate']

    glycemic_load = (gi * carb) / 100

    # 영향 점수 계산
    impact_score = min(100, glycemic_load * 1.5 + (nutrition['sugar'] * 0.5))

    if impact_score < 30:
        level = 'low'
    elif impact_score < 50:
        level = 'medium'
    elif impact_score < 70:
        level = 'medium-high'
    else:
        level = 'high'

    return {
        'glycemic_load': round(glycemic_load, 2),
        'impact_score': round(impact_score, 2),
        'level': level
    }


def generate_diet_records(n=50):
    """식단 기록 샘플 데이터 생성"""
    records = []
    patterns = list(MEAL_PATTERNS.keys())

    base_date = datetime.now() - timedelta(days=60)

    for i in range(n):
        pattern_name = random.choice(patterns)
        pattern = MEAL_PATTERNS[pattern_name]

        # 음식 ID와 이름 추출
        food_ids = [f[0] for f in pattern['foods']]
        food_names = [f[1] for f in pattern['foods']]

        # 영양 정보 계산
        nutrition = calculate_nutrition(food_ids)
        if nutrition is None:
            continue

        # 혈당 영향 계산
        impact = calculate_impact(nutrition)

        record = {
            'record_id': str(uuid.uuid4())[:8],
            'user_id': f'user_{random.randint(1, 50):03d}',
            'timestamp': (base_date + timedelta(days=random.randint(0, 60),
                                                 hours=random.choice([7, 8, 12, 13, 18, 19, 20]),
                                                 minutes=random.randint(0, 59))).strftime('%Y-%m-%d %H:%M:%S'),
            'food_ids': '|'.join(map(str, food_ids)),
            'food_names': '|'.join(food_names),
            'food_count': len(food_ids),
            'total_calories': round(nutrition['calories'], 1),
            'avg_gi_index': round(nutrition['gi_index'], 1),
            'total_carbohydrate': round(nutrition['carbohydrate'], 1),
            'total_protein': round(nutrition['protein'], 1),
            'total_fat': round(nutrition['fat'], 1),
            'total_sugar': round(nutrition['sugar'], 1),
            'total_fiber': round(nutrition['fiber'], 1),
            'total_sodium': round(nutrition['sodium'], 1),
            'glycemic_load': impact['glycemic_load'],
            'impact_score': impact['impact_score'],
            'impact_level': impact['level']
        }

        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    """샘플 데이터 생성 및 저장"""
    print("Sample data generation...")

    # 설문조사 응답 생성
    survey_df = generate_survey_responses(50)
    survey_path = os.path.join(DATA_DIR, 'survey_responses.csv')
    survey_df.to_csv(survey_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Survey responses: {len(survey_df)} records saved")

    # 식단 기록 생성
    diet_df = generate_diet_records(50)
    diet_path = os.path.join(DATA_DIR, 'diet_records.csv')
    diet_df.to_csv(diet_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Diet records: {len(diet_df)} records saved")

    print("\nSample data generation completed!")

    # 데이터 요약 출력
    print("\n=== Survey Data Summary ===")
    print(f"Total responses: {len(survey_df)}")
    print("User type distribution:")
    print(survey_df['user_type'].value_counts().to_string())

    print("\n=== Diet Record Summary ===")
    print(f"Total records: {len(diet_df)}")
    print("Impact level distribution:")
    print(diet_df['impact_level'].value_counts().to_string())


if __name__ == "__main__":
    main()
