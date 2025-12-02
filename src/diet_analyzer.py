"""
식단 분석 및 혈당 예측 모델
사용자의 식단 정보를 분석하여 혈당 영향을 예측하고 피드백을 제공합니다.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class DietAnalyzer:
    def __init__(self, food_db_path=None, model_path=None):
        self.food_db_path = food_db_path or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'food_database.csv'
        )
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), '..', 'models', 'glucose_predictor.joblib'
        )
        self.food_db = None
        self.model = None
        self.scaler = StandardScaler()
        self.load_food_database()

    def load_food_database(self):
        """음식 데이터베이스 로드"""
        if os.path.exists(self.food_db_path):
            self.food_db = pd.read_csv(self.food_db_path)
        else:
            raise FileNotFoundError(f"음식 데이터베이스를 찾을 수 없습니다: {self.food_db_path}")

    def get_food_by_category(self, category: str) -> pd.DataFrame:
        """카테고리별 음식 목록 조회"""
        return self.food_db[self.food_db['category'] == category]

    def get_food_by_id(self, food_id: int) -> dict:
        """ID로 음식 정보 조회"""
        food = self.food_db[self.food_db['food_id'] == food_id]
        if len(food) > 0:
            return food.iloc[0].to_dict()
        return None

    def get_food_by_name(self, name: str) -> dict:
        """이름으로 음식 정보 조회"""
        food = self.food_db[self.food_db['food_name'] == name]
        if len(food) > 0:
            return food.iloc[0].to_dict()
        return None

    def search_food(self, keyword: str) -> pd.DataFrame:
        """키워드로 음식 검색"""
        return self.food_db[self.food_db['food_name'].str.contains(keyword, na=False)]

    def calculate_meal_nutrition(self, food_ids: list, portions: list = None) -> dict:
        """식단의 영양 정보 계산"""
        if portions is None:
            portions = [1.0] * len(food_ids)

        total_nutrition = {
            'calories': 0,
            'gi_index': 0,
            'sugar': 0,
            'carbohydrate': 0,
            'protein': 0,
            'fat': 0,
            'fiber': 0,
            'sodium': 0
        }

        gi_weights = []
        foods_info = []

        for food_id, portion in zip(food_ids, portions):
            food = self.get_food_by_id(food_id)
            if food:
                foods_info.append({
                    'name': food['food_name'],
                    'category': food['category'],
                    'portion': portion
                })
                total_nutrition['calories'] += food['calories'] * portion
                total_nutrition['sugar'] += food['sugar'] * portion
                total_nutrition['carbohydrate'] += food['carbohydrate'] * portion
                total_nutrition['protein'] += food['protein'] * portion
                total_nutrition['fat'] += food['fat'] * portion
                total_nutrition['fiber'] += food['fiber'] * portion
                total_nutrition['sodium'] += food['sodium'] * portion
                gi_weights.append((food['gi_index'], food['carbohydrate'] * portion))

        # 가중 평균 GI 계산
        if gi_weights:
            total_carbs = sum(w[1] for w in gi_weights)
            if total_carbs > 0:
                total_nutrition['gi_index'] = sum(gi * carb for gi, carb in gi_weights) / total_carbs
            else:
                total_nutrition['gi_index'] = 0

        total_nutrition['foods'] = foods_info
        return total_nutrition

    def predict_glucose_impact(self, meal_nutrition: dict, user_type: str = None) -> dict:
        """혈당 영향 예측"""
        # GI 기반 혈당 부하(GL) 계산
        gl = (meal_nutrition['gi_index'] * meal_nutrition['carbohydrate']) / 100

        # 혈당 영향 점수 계산 (0-100)
        base_impact = min(100, (gl / 50) * 100)

        # 섬유소 보정 (섬유소가 많으면 혈당 상승 완화)
        fiber_factor = max(0.7, 1 - (meal_nutrition['fiber'] / 30))

        # 단백질/지방 보정 (혈당 상승 속도 완화)
        protein_fat_factor = max(0.8, 1 - (meal_nutrition['protein'] + meal_nutrition['fat']) / 100)

        adjusted_impact = base_impact * fiber_factor * protein_fat_factor

        # 유형별 추가 보정
        type_multipliers = {
            'impulse_low_activity': 1.2,
            'impulse_active': 0.9,
            'social_low_control': 1.15,
            'social_controlled': 1.0,
            'stress_eater': 1.1,
            'balanced': 0.85
        }

        if user_type and user_type in type_multipliers:
            adjusted_impact *= type_multipliers[user_type]

        adjusted_impact = min(100, max(0, adjusted_impact))

        # 예상 혈당 상승폭 추정 (mg/dL)
        estimated_spike = 20 + (adjusted_impact * 1.5)

        return {
            'glycemic_load': round(gl, 1),
            'impact_score': round(adjusted_impact, 1),
            'estimated_spike': round(estimated_spike, 1),
            'risk_level': self._get_risk_level(adjusted_impact),
            'gi_category': self._get_gi_category(meal_nutrition['gi_index'])
        }

    def _get_risk_level(self, impact_score: float) -> str:
        """위험도 레벨 반환"""
        if impact_score < 30:
            return 'low'
        elif impact_score < 50:
            return 'medium'
        elif impact_score < 70:
            return 'high'
        else:
            return 'very_high'

    def _get_gi_category(self, gi: float) -> str:
        """GI 카테고리 반환"""
        if gi < 55:
            return 'low'
        elif gi < 70:
            return 'medium'
        else:
            return 'high'

    def generate_feedback(self, meal_nutrition: dict, glucose_impact: dict,
                         user_type_info: dict = None) -> dict:
        """식단에 대한 피드백 생성"""
        feedback = {
            'summary': '',
            'positives': [],
            'warnings': [],
            'suggestions': []
        }

        gi = meal_nutrition['gi_index']
        gl = glucose_impact['glycemic_load']
        calories = meal_nutrition['calories']
        sugar = meal_nutrition['sugar']
        fiber = meal_nutrition['fiber']
        sodium = meal_nutrition['sodium']
        impact = glucose_impact['impact_score']

        # 요약
        if impact < 30:
            feedback['summary'] = '혈당 관리에 좋은 식단입니다!'
        elif impact < 50:
            feedback['summary'] = '적절한 수준의 식단이지만, 개선의 여지가 있습니다.'
        elif impact < 70:
            feedback['summary'] = '혈당 상승이 우려되는 식단입니다. 조절이 필요합니다.'
        else:
            feedback['summary'] = '혈당에 큰 영향을 줄 수 있는 식단입니다. 주의가 필요합니다.'

        # 긍정적 요소
        if gi < 55:
            feedback['positives'].append('낮은 GI 지수로 혈당 상승이 완만합니다.')
        if fiber >= 5:
            feedback['positives'].append(f'식이섬유({fiber:.1f}g)가 풍부하여 혈당 조절에 도움됩니다.')
        if meal_nutrition['protein'] >= 15:
            feedback['positives'].append('단백질이 충분하여 포만감이 오래 지속됩니다.')
        if sugar < 10:
            feedback['positives'].append('당류 함량이 낮아 혈당 급상승을 방지합니다.')

        # 경고
        if gi >= 70:
            feedback['warnings'].append(f'높은 GI 지수({gi:.0f})로 혈당이 빠르게 상승할 수 있습니다.')
        if gl >= 20:
            feedback['warnings'].append(f'혈당부하(GL: {gl:.1f})가 높습니다.')
        if sugar >= 25:
            feedback['warnings'].append(f'당류({sugar:.1f}g)가 많습니다. 혈당 급상승 주의!')
        if sodium >= 1500:
            feedback['warnings'].append(f'나트륨({sodium:.0f}mg)이 높습니다.')
        if calories >= 700:
            feedback['warnings'].append(f'칼로리({calories:.0f}kcal)가 높은 편입니다.')

        # 제안
        if gi >= 55:
            feedback['suggestions'].append('채소나 샐러드를 함께 섭취하면 GI를 낮출 수 있습니다.')
        if fiber < 3:
            feedback['suggestions'].append('식이섬유가 부족합니다. 현미밥이나 채소를 추가해보세요.')
        if meal_nutrition['protein'] < 10:
            feedback['suggestions'].append('단백질 섭취를 늘리면 혈당 안정에 도움됩니다.')

        # 유형별 맞춤 제안
        if user_type_info:
            type_id = user_type_info.get('type_id', '')
            if type_id == 'impulse_low_activity':
                feedback['suggestions'].append('식후 10분 산책은 혈당 조절에 큰 도움이 됩니다.')
            elif type_id == 'stress_eater':
                feedback['suggestions'].append('천천히 음미하며 드시면 포만감이 더 빨리 옵니다.')
            elif type_id in ['social_low_control', 'social_controlled']:
                feedback['suggestions'].append('식사량을 미리 정해두면 과식을 예방할 수 있습니다.')

        return feedback

    def get_alternative_foods(self, food_id: int, max_results: int = 3) -> list:
        """더 나은 대안 음식 추천"""
        original = self.get_food_by_id(food_id)
        if not original:
            return []

        category = original['category']
        original_gi = original['gi_index']

        # 같은 카테고리에서 GI가 낮은 음식 찾기
        same_category = self.food_db[
            (self.food_db['category'] == category) &
            (self.food_db['gi_index'] < original_gi) &
            (self.food_db['food_id'] != food_id)
        ].sort_values('gi_index')

        alternatives = []
        for _, food in same_category.head(max_results).iterrows():
            gi_reduction = original_gi - food['gi_index']
            alternatives.append({
                'food_id': food['food_id'],
                'food_name': food['food_name'],
                'gi_index': food['gi_index'],
                'gi_reduction': gi_reduction,
                'calories': food['calories'],
                'reason': f"GI가 {gi_reduction:.0f} 낮아 혈당 상승이 완만합니다."
            })

        return alternatives

    def train_prediction_model(self, training_data: pd.DataFrame = None):
        """혈당 예측 모델 학습"""
        if training_data is None:
            training_data = self._generate_training_data()

        features = ['gi_index', 'carbohydrate', 'sugar', 'fiber', 'protein', 'fat', 'calories']
        X = training_data[features]
        y = training_data['glucose_impact']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        score = self.model.score(X_test_scaled, y_test)

        # 모델 저장
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'score': score
        }
        self.save_model(model_data)

        return score

    def _generate_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """학습용 샘플 데이터 생성"""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            gi = np.random.uniform(20, 100)
            carb = np.random.uniform(10, 100)
            sugar = np.random.uniform(0, carb * 0.5)
            fiber = np.random.uniform(0, 15)
            protein = np.random.uniform(5, 40)
            fat = np.random.uniform(2, 35)
            calories = carb * 4 + protein * 4 + fat * 9

            # 혈당 영향 시뮬레이션
            gl = (gi * carb) / 100
            base_impact = min(100, (gl / 50) * 100)
            fiber_effect = max(0.7, 1 - fiber / 30)
            protein_effect = max(0.8, 1 - protein / 80)
            noise = np.random.normal(0, 5)

            glucose_impact = max(0, min(100, base_impact * fiber_effect * protein_effect + noise))

            data.append({
                'gi_index': gi,
                'carbohydrate': carb,
                'sugar': sugar,
                'fiber': fiber,
                'protein': protein,
                'fat': fat,
                'calories': calories,
                'glucose_impact': glucose_impact
            })

        return pd.DataFrame(data)

    def save_model(self, model_data: dict):
        """모델 저장"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model_data, self.model_path)

    def load_model(self):
        """모델 로드"""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            return True
        return False

    def get_categories(self) -> list:
        """음식 카테고리 목록 조회"""
        return self.food_db['category'].unique().tolist()

    def get_all_foods(self) -> pd.DataFrame:
        """전체 음식 목록 조회"""
        return self.food_db
