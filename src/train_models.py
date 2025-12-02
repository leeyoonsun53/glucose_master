"""
모델 학습 스크립트
사용자 유형 분류 및 혈당 예측 모델을 학습합니다.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_classifier import UserClassifier
from diet_analyzer import DietAnalyzer


def train_user_classifier():
    """사용자 분류 모델 학습 및 샘플 데이터 생성"""
    print("=" * 50)
    print("사용자 유형 분류 모델 학습")
    print("=" * 50)

    classifier = UserClassifier()

    # 학습 데이터 생성
    print("\n학습 데이터 생성 중...")
    training_data = classifier.generate_training_data(n_samples=500)

    # 데이터 저장
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_training_data.csv')
    training_data.to_csv(data_path, index=False, encoding='utf-8-sig')
    print(f"학습 데이터 저장 완료: {data_path}")
    print(f"샘플 수: {len(training_data)}")

    # 유형별 분포 출력
    print("\n유형별 분포:")
    print(training_data['user_type'].value_counts())

    print("\n사용자 분류 모델 준비 완료!")


def train_diet_analyzer():
    """식단 분석 및 혈당 예측 모델 학습"""
    print("\n" + "=" * 50)
    print("혈당 예측 모델 학습")
    print("=" * 50)

    analyzer = DietAnalyzer()

    print("\n모델 학습 중...")
    score = analyzer.train_prediction_model()

    print(f"\n모델 학습 완료!")
    print(f"테스트 정확도 (R² score): {score:.4f}")


def main():
    """메인 함수"""
    print("\n" + "=" * 50)
    print("   혈당관리 마스터 - 모델 학습")
    print("=" * 50)

    train_user_classifier()
    train_diet_analyzer()

    print("\n" + "=" * 50)
    print("모든 모델 학습 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
