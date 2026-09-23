# 통합 포트폴리오 대시보드 - 수정본

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
`app.py`와 `requirements.txt`를 같은 저장소/브랜치에 올리고, `features/` 폴더 전체를 함께 커밋해야 합니다.
기존 Streamlit Secrets의 `DART_API_KEY`, `SEC_USER_AGENT`는 그대로 사용합니다.

## 중요
- 6~10번 기능은 `features/featureXX_.../page.py`에서 렌더링됩니다.
- 각 폴더의 `__init__.py`도 포함되어 있습니다.
- 7번 공개 데이터는 첫 실행 시 `feature07_inflation/data/`에 준비합니다.

## 기능 폴더 구조

1~5번은 기존 화면을 보존하는 어댑터를 먼저 `features/feature01~05`에 연결했습니다.
실제 계산·차트·보고서 코드는 검증을 거치며 각 패키지로 단계적으로 이동할 수 있습니다.
6~10번 기능은 번호가 각각 7~11번으로 이동했으며, 기존 `f06~f10` 내부 상태 키는 호환성을 위해 유지합니다.
- 10번 라오어 V2.2는 일봉 OHLC를 이용한 근사 백테스트이며 쿼터손절은 옵션입니다.
