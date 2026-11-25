# K-CURE 암 공공 라이브러리 결합 스키마 가이드

## 1. 5대 결합 기관 데이터 구조

### (1) 중앙암등록본부 (국립암센터) - `CANCER_REGISTRY`
* `ID`: 가명 결합키 (환자 고유 식별자)
* `FDX`: 최초진단일자 (YYYYMMDD)
* `AGE`: 암 진단 시 연령
* `SEX`: 성별 (1: 남성, 2: 여성)
* `ICD_10`: 원발암 분류코드 (예: C16 위암, C18~C20 대장암, C22 간암, C34 폐암, C50 유방암 등)
* `MCODE`: 종양학 국제질병분류 형태코드 (ICD-O-3 Morphology)
* `SEER`: 요약병기 (1: 국한 Localized, 2: 국소 Regional, 3: 원격전이 Distant, 9: 미상)
* `SURGERY_YN`, `CHEMO_YN`, `RT_YN`: 최초 등록 시 수술, 항암, 방사선 치료 여부 (Y/N)

---

### (2) 국민건강보험공단 (NHIS) - `NHIS_SCREENING_ELIG`
* `ID`: 가명 결합키
* `INCOME_DECILE`: 건강보험료 기준 소득분위 (1~10분위, 0: 의료급여)
* `REGION`: 거주 지역 구분 (수도권 / 비수도권)
* `HEIGHT`, `WEIGHT`, `BMI`: 체질량지수 (kg/m²)
* `SBP`, `DBP`: 수축기 / 이완기 혈압 (mmHg)
* `FBS`: 공복혈당 (mg/dL)
* `CHOL_TOT`, `HDL`, `LDL`, `TG`: 지질 프로파일
* `EGFR`: 신사구체여과율 (신장 기능)
* `AST`, `ALT`, `GGT`: 간기능 검사 수치
* `SMOKING_STATUS`: 흡연 여부 (0: 비흡연, 1: 과거흡연, 2: 현재흡연) & 갑년(Pack-years)
* `ALCOHOL_FREQ`: 음주 빈도 (주당 횟수)
* `PHYSICAL_ACT`: 주당 중고강도 신체활동 일수

---

### (3) 건강보험심사평가원 (HIRA) - `HIRA_CLAIMS`
* `ID`: 가명 결합키
* `CLAIM_ID`: 명세서 일련번호
* `RECU_FR_DT`: 요양개시일자 (YYYYMMDD)
* `MAIN_SICK`, `SUB_SICK`: 주상병 / 부상병 코드 (ICD-10)
  * 심혈관질환(CVD): I20~I25 (허혈성 심질환), I50 (심부전), I60~I64 (뇌졸중)
  * 대사질환: E11 (제2형 당뇨병), E78 (지질대사 장애)
  * 신질환: N18 (만성 신장병)
* `DRUG_CODE`: 원외처방 약제 코드
  * 심독성 유발 항암제: Anthracycline(Doxorubicin 등), Trastuzumab, 5-FU, Cisplatin
  * 심혈관/대사 약제: 항고혈압제, 혈당강하제(인슐린, 메트포르민), 스타틴
* `CCI`: Charlson Comorbidity Index (기저 동반질환 지수) 산출

---

### (4) 통계청 (KOSTAT) - `DEATH_REGISTRY`
* `ID`: 가명 결합키
* `DEATH_YN`: 사망 여부 (1: 사망, 0: 생존)
* `DEATH_DT`: 사망일자 (YYYYMMDD)
* `DEATH_CAUSE`: 사망 원인 질병분류코드 (KCD)
  * C00~C97: 암 사망
  * I00~I99: 순환기계 질환(심혈관) 사망
  * 기타 질환 사망

---

## 2. 분석 대상 코호트 및 결과변수 설계
1. **Index Date**: 암 진단일(`FDX`)로부터 5년이 경과한 시점 (5-Year Cancer Survivors)
2. **Follow-up 종료 시점**:
   * 만성질환 최초 진단일
   * 사망일자
   * 추적관찰 종료일 (Censoring)
3. **경쟁 위험 (Competing Risk)**: 만성질환 발생 전 암 재발로 사망한 경우를 경쟁 사건(Competing Event)으로 정의.
