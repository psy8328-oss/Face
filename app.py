"""AI 관상가 - Streamlit 앱."""

import base64
import io
import logging
import os

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from PIL import Image, ImageOps, UnidentifiedImageError

from prompts import CANNOT_READ_MARKER, NO_FACE_MARKER, SYSTEM_PROMPT, USER_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 1024
MAX_PIXELS = 50_000_000  # 5000만 화소 (최신 폰 고화소 모드까지 허용)
MAX_COMPLETION_TOKENS = 1500
TEMPERATURE = 0.8
REQUEST_TIMEOUT = 60

DISCLAIMER = "⚠️ 본 결과는 재미를 위한 것이며 과학적 근거가 없습니다."
PRIVACY_NOTICE = (
    "🔒 업로드한 사진은 분석을 위해 OpenAI로 전송되며 앱에 저장되지 않습니다. "
    "본인 사진 또는 동의를 받은 사진만 사용해 주세요."
)


class AnalysisError(Exception):
    """사용자에게 보여줄 한국어 메시지를 담은 예외."""

    def __init__(self, message: str, level: str = "error"):
        super().__init__(message)
        self.message = message
        self.level = level  # "error" 또는 "warning"


# ---------------------------------------------------------------- 설정


def _read_secret(name: str) -> str | None:
    """st.secrets에서 값을 읽는다. secrets 파일이 없으면 None을 반환한다."""
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value).strip() if value else None


def get_config() -> dict:
    """설정을 읽는다. 우선순위: st.secrets → .env(환경변수)."""
    load_dotenv()

    api_key = _read_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").strip()
    model = (
        _read_secret("OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL", "").strip()
        or DEFAULT_MODEL
    )

    if not api_key:
        st.error(
            "OpenAI API 키가 설정되지 않았습니다.\n\n"
            "- **로컬 실행**: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=...`를 추가하세요.\n"
            "- **Streamlit Cloud**: App settings → Secrets에 `OPENAI_API_KEY = \"...\"`를 등록하세요."
        )
        st.stop()

    return {"api_key": api_key, "model": model}


# ---------------------------------------------------------------- 이미지 처리


def prepare_image(raw: bytes) -> bytes:
    """업로드 이미지를 검증하고 전송용 JPEG로 변환한다. 메모리에서만 처리한다."""
    if not raw:
        raise AnalysisError("빈 파일입니다. 다른 사진을 올려주세요.")
    if len(raw) > MAX_FILE_BYTES:
        raise AnalysisError("파일이 너무 큽니다. 10MB 이하의 사진을 올려주세요.")

    unreadable = "이미지를 읽을 수 없습니다. 손상되지 않은 JPG/PNG/WEBP 사진을 올려주세요."
    too_large = "사진 해상도가 너무 큽니다. 5000만 화소 이하의 사진을 올려주세요."

    try:
        img = Image.open(io.BytesIO(raw))  # 헤더만 읽음 (픽셀은 아직 디코딩하지 않음)
    except Image.DecompressionBombError:
        raise AnalysisError(too_large)
    except (UnidentifiedImageError, OSError, ValueError):
        raise AnalysisError(unreadable)

    with img:
        # 압축률이 높은 이미지는 파일이 작아도 디코딩하면 GB 단위 메모리를 쓰므로 먼저 막는다.
        if img.width * img.height > MAX_PIXELS:
            raise AnalysisError(too_large)
        try:
            # 원본 크기로 복사본을 만들지 않도록 먼저 줄인다 (JPEG는 축소 디코딩됨).
            if img.mode == "P":
                img = img.convert("RGBA")
            elif img.mode not in ("RGB", "RGBA", "L", "LA", "CMYK"):
                img = img.convert("RGB")
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            img = ImageOps.exif_transpose(img)

            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGBA")
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.getchannel("A"))
                img = background
            else:
                img = img.convert("RGB")

            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85)
            return out.getvalue()
        except (OSError, ValueError, Image.DecompressionBombError):
            raise AnalysisError(unreadable)


def to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


# ---------------------------------------------------------------- OpenAI 호출


def _call_model(client: OpenAI, model: str, data_url: str, with_temperature: bool):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    if with_temperature:
        kwargs["temperature"] = TEMPERATURE
    return client.chat.completions.create(**kwargs)


def analyze_face(jpeg_bytes: bytes, config: dict) -> str:
    """OpenAI 비전 모델로 관상 풀이를 받아온다. 실패하면 AnalysisError를 던진다."""
    client = OpenAI(api_key=config["api_key"], timeout=REQUEST_TIMEOUT, max_retries=1)
    data_url = to_data_url(jpeg_bytes)

    # 예외 메시지에 키 일부가 들어갈 수 있으므로 화면·로그에는 예외 종류만 남긴다.
    try:
        try:
            response = _call_model(client, config["model"], data_url, with_temperature=True)
        except BadRequestError as e:
            # temperature를 지원하지 않는 모델이면 빼고 한 번 더 시도한다.
            if "temperature" not in str(e):
                raise
            response = _call_model(client, config["model"], data_url, with_temperature=False)
    except AuthenticationError:
        raise AnalysisError("API 키가 올바르지 않습니다. 설정된 OPENAI_API_KEY를 확인해 주세요.")
    except PermissionDeniedError:
        raise AnalysisError("이 API 키로는 해당 모델을 사용할 권한이 없습니다.")
    except RateLimitError:
        raise AnalysisError("요청이 너무 많거나 사용 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.")
    except APITimeoutError:
        raise AnalysisError("분석 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.")
    except APIConnectionError:
        raise AnalysisError("OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요.")
    except BadRequestError:
        logger.warning("OpenAI BadRequestError (model=%s)", config["model"])
        raise AnalysisError(
            f"요청이 거부되었습니다. 모델({config['model']})이 이미지 분석을 지원하는지 확인해 주세요."
        )
    except APIStatusError as e:
        logger.warning("OpenAI APIStatusError status=%s", e.status_code)
        raise AnalysisError("OpenAI 서버에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
    except Exception as e:
        logger.warning("Unexpected error during analysis: %s", type(e).__name__)
        raise AnalysisError("알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")

    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise AnalysisError(
            "AI가 이 사진의 분석을 거절했습니다. 얼굴이 잘 보이는 다른 사진으로 시도해 주세요.",
            level="warning",
        )

    content = (message.content or "").strip()
    if not content:
        raise AnalysisError("AI 응답이 비어 있습니다. 다시 시도해 주세요.")
    if NO_FACE_MARKER in content:
        raise AnalysisError(
            "사진에서 얼굴을 찾지 못했습니다. 얼굴이 정면으로 잘 보이는 밝은 사진을 다시 올려주세요.",
            level="warning",
        )
    if CANNOT_READ_MARKER in content:
        raise AnalysisError(
            "이 사진으로는 관상을 풀이하기 어렵습니다. 다른 사진으로 다시 시도해 주세요.",
            level="warning",
        )
    return content


# ---------------------------------------------------------------- 상태 관리


def init_state() -> None:
    defaults = {"image_id": None, "result": None, "error": None, "analyzing": False}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_if_new_image(image_id) -> None:
    """다른 이미지가 들어오면 이전 결과를 지운다."""
    if st.session_state.image_id != image_id:
        st.session_state.image_id = image_id
        st.session_state.result = None
        st.session_state.error = None
        st.session_state.analyzing = False


def _start_analysis() -> None:
    st.session_state.analyzing = True
    st.session_state.result = None
    st.session_state.error = None


def _restart() -> None:
    """'다시 하기': 결과를 지우고 입력 위젯 key를 바꿔 업로드/촬영 사진도 비운다."""
    st.session_state.input_nonce = st.session_state.get("input_nonce", 0) + 1
    st.session_state.image_id = None
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.analyzing = False


# ---------------------------------------------------------------- 화면

# 색은 테마(.streamlit/config.toml)를 따르고, 여기서는 배치와 크기만 다룬다.
# 금색 장식선(#C9A227)만 고정색이며 라이트/다크 배경 모두에서 보인다.
CUSTOM_CSS = """
<style>
.hero { text-align: center; padding: 0.5rem 0 1.25rem; }
.hero-icon { font-size: 3rem; line-height: 1; }
.hero h1 { margin: 0.4rem 0 0.25rem; padding: 0; font-size: 2.6rem; letter-spacing: 0.08em; }
.hero p { margin: 0; opacity: 0.75; font-size: 1.05rem; }
.hero-line { width: 72px; height: 2px; margin: 1rem auto 0; background: #C9A227; border-radius: 2px; }
.footer { text-align: center; font-size: 0.8rem; opacity: 0.65; line-height: 1.6; padding-top: 0.5rem; }
@media (max-width: 480px) {
  .hero h1 { font-size: 2rem; }
  .hero p { font-size: 0.95rem; }
}
</style>
"""

PHOTO_TIPS = """
- **정면**을 바라보는 사진이 가장 정확해요.
- **밝은 조명**에서 그림자 없이 찍어 주세요.
- 이마부터 턱까지 **얼굴 전체**가 보이게 해 주세요. 앞머리나 마스크로 가리지 않는 게 좋아요.
- 선글라스, 강한 필터, 여러 명이 함께 나온 사진은 피해 주세요.
"""


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <div class="hero-icon">🔮</div>
          <h1>AI 관상가</h1>
          <p>사진 한 장으로 보는 나의 관상</p>
          <div class="hero-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("📸 좋은 사진 고르는 팁"):
        st.markdown(PHOTO_TIPS)


def render_footer() -> None:
    st.divider()
    st.markdown(
        f'<div class="footer">{DISCLAIMER}<br>{PRIVACY_NOTICE}</div>',
        unsafe_allow_html=True,
    )


def get_image_input():
    """업로드 또는 카메라 촬영으로 이미지를 받는다. 없으면 None.

    탭은 열린 쪽만 실행해서, 카메라 탭을 열기 전에는 카메라 권한을 요청하지 않는다.
    """
    nonce = st.session_state.get("input_nonce", 0)
    upload_tab, camera_tab = st.tabs(
        ["📁 사진 올리기", "📷 카메라로 찍기"], key="input_tab", on_change="rerun"
    )

    if upload_tab.open:
        with upload_tab:
            return st.file_uploader(
                "얼굴이 잘 보이는 정면 사진을 올려주세요 (JPG·PNG·WEBP, 최대 10MB)",
                type=ALLOWED_TYPES,
                key=f"uploader_{nonce}",
            )
    if camera_tab.open:
        with camera_tab:
            return st.camera_input("카메라를 보고 촬영해 주세요", key=f"camera_{nonce}")
    return None


def split_sections(markdown: str) -> list[str]:
    """결과 마크다운을 '## ' 제목 기준으로 나눈다. 제목이 없으면 통째로 하나."""
    sections, current = [], []
    for line in markdown.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if s]


def render_result() -> None:
    error = st.session_state.error
    if error:
        message, level = error
        (st.warning if level == "warning" else st.error)(message)

    result = st.session_state.result
    if result:
        st.markdown("#### 📜 관상 풀이")
        for section in split_sections(result):
            if section.startswith("## "):
                # 카드 안에서는 '##'가 너무 커서 한 단계 작은 제목으로 표시한다 (다운로드 원문은 그대로).
                section = "####" + section[2:]
            with st.container(border=True):
                st.markdown(section)
        # 결과가 길면 하단 안내가 화면 밖으로 밀리므로 결과 바로 아래에도 표시한다 (PRD F8).
        st.caption(DISCLAIMER)
        st.download_button(
            "📥 결과 저장하기",
            data=result,
            file_name="관상_결과.md",
            mime="text/markdown",
            width="stretch",
        )

    if error or result:
        st.button("🔄 다시 하기", on_click=_restart, width="stretch")


def render_body(config: dict) -> None:
    image_file = get_image_input()
    if image_file is None:
        reset_if_new_image(None)
        return

    reset_if_new_image(image_file.file_id)

    try:
        jpeg_bytes = prepare_image(image_file.getvalue())
    except AnalysisError as e:
        st.error(e.message)
        return

    with st.container(horizontal_alignment="center"):
        st.image(jpeg_bytes, caption="올려주신 사진", width=280)

    # 결과가 나온 뒤에는 버튼을 숨겨 같은 사진의 반복 유료 요청을 막는다.
    # 다시 보려면 '다시 하기' 또는 새 사진 업로드로 시작한다. (오류 후에는 재시도 가능)
    if st.session_state.result is None:
        st.button(
            "🔮 관상 보기",
            type="primary",
            disabled=st.session_state.analyzing,
            on_click=_start_analysis,
            width="stretch",
        )

    if st.session_state.analyzing:
        try:
            with st.spinner("관상가가 얼굴을 찬찬히 살펴보고 있습니다..."):
                st.session_state.result = analyze_face(jpeg_bytes, config)
        except AnalysisError as e:
            st.session_state.error = (e.message, e.level)
        finally:
            st.session_state.analyzing = False
        st.rerun()

    render_result()


def main() -> None:
    st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")
    init_state()

    render_header()
    config = get_config()
    render_body(config)
    render_footer()


if __name__ == "__main__":
    main()
