# 🔮 AI 관상가

얼굴 사진을 업로드하면 OpenAI 이미지 분석 모델이 전통 관상학 관점에서 풀이해 주는 Streamlit 웹앱입니다.

> 본 결과는 재미를 위한 것이며 과학적 근거가 없습니다.

## 로컬 실행

1. 의존성 설치

   ```bash
   pip install -r requirements.txt
   ```

2. `.env.example`을 복사해 `.env`를 만들고 API 키를 입력합니다.

   ```
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini   # 선택, 비우면 기본값 사용
   ```

3. 앱 실행

   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud 배포

1. GitHub에 저장소를 올립니다. (`.env`는 `.gitignore`에 포함되어 업로드되지 않습니다.)
2. [share.streamlit.io](https://share.streamlit.io)에서 저장소를 연결하고 Main file path를 `app.py`로 지정합니다.
3. **Advanced settings → Secrets**에 아래 내용을 입력합니다. Streamlit Cloud는 `.env` 파일을 읽지 않습니다.

   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   ```

4. Deploy를 누릅니다.

## 설정 우선순위

`st.secrets` → `.env`(환경변수) 순서로 읽습니다. 둘 다 없으면 앱 화면에 설정 안내가 표시됩니다.

## 개인정보

업로드한 사진은 분석을 위해 OpenAI로 전송되며 앱에 저장되지 않습니다.
