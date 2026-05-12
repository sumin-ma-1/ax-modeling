# 대청호 조류경보 대시보드 실행 방법

## 1. 실행 명령

작업 폴더에서 아래 명령을 실행합니다.

```powershell
python -m streamlit run app.py
```

실행 후 브라우저에서 아래 주소를 엽니다.

```text
http://localhost:8501
```

## 2. 화면 구성

- 대시보드: 지점별 `T+1`, `T+3`, `T+7`, `T+10` 위험도와 heatmap
- 예측 실행: 운영 누적 데이터 기준 예측, 결과 다운로드, 최근 관측값 확인
- API 수집: 기상청 ASOS API 수집, 기관별 JSON API 테스트, 당일 관측 템플릿 생성
- 위험 원인: SHAP top feature와 SHAP summary plot
- 대응 시나리오: 리드타임별 권장 조치, 담당 부서, 긴급도
- 모델 성능: best model 성능표, 전체 모델 비교, ROC/PR, Confusion Matrix

사이드바의 `예측 모델 선택`에서 두 모델을 전환할 수 있습니다.

```text
환경·수문 기반 사전예측 모델
- 조류 세포수 계열 제외
- 조류 모니터링 값이 아직 없을 때 사용하는 주 모델

조류 모니터링 포함 보조 모델
- log_cyano, cyano lag/rolling, 회남 cyano lag 포함
- 조류 세포수 측정값이 확보된 이후 단기 위험 판단에 사용하는 보조 모델
```

## 3. 입력 데이터

운영 입력은 `operational_data.csv`입니다.

앱을 처음 실행했을 때 `operational_data.csv`가 없으면 기존 `final_data.csv`를 복사해 운영 원장을 자동 생성합니다.

새 관측값을 반영할 때는 UI 왼쪽 사이드바의 `신규 관측 CSV 누적 반영`에 CSV를 업로드한 뒤 `운영 데이터셋에 추가/갱신` 버튼을 누릅니다.

업로드 파일은 오늘 관측값만 들어 있어도 됩니다. 다만 컬럼 구조는 기존 `final_data.csv`와 동일한 것이 가장 좋고, 최소한 `조사일`, `채수위치`와 모델 입력 피처 생성에 필요한 기상·수문·조류 컬럼이 포함되어야 합니다.

중복 데이터는 `조사일 + 채수위치` 기준으로 처리합니다. 같은 날짜와 지점이 이미 있으면 새로 업로드한 값으로 갱신됩니다.

운영 흐름은 다음과 같습니다.

```text
final_data.csv
→ 최초 실행 시 operational_data.csv 생성
→ 신규 관측 CSV 업로드
→ operational_data.csv에 누적/갱신
→ 누적 이력으로 lag/rolling/CHD 피처 재계산
→ T+1/T+3/T+7/T+10 예측
```

## 4. 예측 해석

예측확률이 모델별 threshold 이상이면 `고위험`으로 표시됩니다.

현재 threshold는 다음 파일에서 관리됩니다.

```text
outputs/modeling_env/tables/best_model_summary.csv
```

운영 단계는 다음 기준으로 표시됩니다.

```text
낮음: 0.30 미만
관심: 0.30 이상
주의: 0.60 이상
고위험: 리드타임별 threshold 이상
```

## 5. API 연동

### 기상청 ASOS 일자료

`API 수집` 탭에서 기상청 공공데이터포털 `serviceKey`를 입력하고 `기상자료 불러오기`를 누르면 선택한 기준일의 대전·청주·보은 ASOS 일자료를 수집합니다.

수집된 기상값은 당일 관측 템플릿에 자동 병합되며, 병합된 CSV를 내려받아 조류·수질 실측값을 채운 뒤 사이드바에서 운영 데이터셋에 누적 반영하면 됩니다.

서비스키를 매번 입력하지 않으려면 아래 파일을 만들 수 있습니다.

```text
.streamlit/secrets.toml
```

예시는 다음 파일에 있습니다.

```text
.streamlit/secrets.toml.example
```

### K-water 또는 내부 수질 API

기관별 API는 응답 구조가 다를 수 있어 `Generic JSON API 호출기`를 제공합니다.

입력 항목은 다음입니다.

```text
API URL
API Key 또는 serviceKey
레코드 경로 예: response.body.items.item
추가 query params(JSON)
```

API 응답을 확인한 뒤 CSV로 다운로드하고, 컬럼명을 `final_data.csv` 구조에 맞춘 다음 운영 데이터셋에 누적 반영합니다.

## 6. 주요 산출물

대시보드는 아래 파일을 자동으로 읽습니다.

```text
outputs/modeling_env/models/best_model_Tplus*.pkl
outputs/modeling_env/tables/best_model_summary.csv
outputs/modeling_env/tables/model_results.csv
outputs/modeling_env/tables/shap_top_features.csv
outputs/modeling_env/tables/scenario_recommendation.csv
outputs/modeling_env/figures/*.png
```

현재 대시보드는 과제 목적에 맞춘 `outputs/modeling_env` 모델을 기본으로 사용합니다. 이 모델은 `total_cyano` 및 유해남조류 4종(`microcystis`, `anabaena`, `oscillatoria`, `aphanizomenon`)과 이들의 lag/rolling/공간 선행 피처를 입력변수에서 제외한 사전예측 모델입니다.
