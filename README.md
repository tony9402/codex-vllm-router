# Codex Responses Router

## 개요
`codex_responses_router.py`는 **FastAPI** 기반 서비스로, **Codex Responses** API 형식을 downstream LLM이 사용하는 **OpenAI Chat Completions** 요청 형식으로 변환합니다. 페이로드를 정규화하고 헤더를 복사하며 함수 호출 방식을 변환해 기존 Codex 클라이언트가 OpenAI 호환 엔드포인트와 통신할 수 있게 합니다.

## 빠른 시작
1. **필요한 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```
2. **환경 변수 설정**
   ```bash
   export UPSTREAM_BASE_URL="http://<upstream-host>:<port>/v1"   # 필수
   export LLM_API_KEY="..."                                 # 선택, 없으면 자동으로 Bearer 로 접두
   # 필요 시 헤더 이름 지정
   export LLM_KEY_HEADER="Authorization"   # 기본값
   export REQUEST_TIMEOUT_SECONDS="600"    # 기본 타임아웃(초)
   ```
3. **서버 실행** — 제공된 `start.sh` 스크립트를 이용하거나 직접 `uvicorn`을 실행합니다.
   ```bash
   # 권장: start.sh 사용
   ./start.sh

   # 직접 실행
   uvicorn codex_responses_router:app \
     --host 127.0.0.1 \
     --port 8787
   ```

   서버는 **http://127.0.0.1:8787** 에서 요청을 받습니다.

## 동작 원리
* **헤더 전달** — 들어오는 `Authorization`(또는 `LLM_KEY_HEADER`에 지정된 헤더)을 upstream 요청에 복사하고, 없을 경우 `LLM_API_KEY`를 사용합니다.
* **페이로드 변환** — **Responses** 본문의 메시지에 ID, 상태, `output_text` 주석을 추가하고 Chat‑Completions 형식으로 변환합니다.
* **함수 매핑** — Responses 스타일 함수 정의와 호출을 OpenAI 호환 `function` 도구 형식으로 변환합니다.

## 개발 및 테스트
* 로컬에서 서버를 실행한 뒤 `/v1/responses` 엔드포인트에 POST 요청을 보냅니다(`curl`, Postman 등 사용).
* 라우터는 `UPSTREAM_BASE_URL`에 지정된 URL로 요청을 전달하고, 스트리밍 응답을 클라이언트에 반환합니다.

## 라이선스
본 프로젝트는 저장소 최상위 `LICENSE` 파일에 정의된 라이선스를 따릅니다.

## 설정 파일 업데이트
`codex_config_example.toml` 은 예시 설정 파일이며, 실제 사용을 위해서는 이 파일의 내용을 **`~/.codex/config.toml`** 로 복사한 뒤 각 항목을 실제 값으로 교체해야 합니다. 민감한 정보(모델명, API 키, IP, 포트 등)는 예시가 아닌 실제 환경에 맞는 값으로 바꾸세요.

```toml
# 예시 설정 (실제 값으로 교체)
model = "<your-model-name>"
model_provider = "<your-provider-name>"

[model_providers.<your-provider-name>]
name = "<Provider Display Name>"
base_url = "http://<host>:<port>/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false

request_max_retries = 4
stream_max_retries = 5
stream_idle_timeout_ms = 600000
```

위와 같이 수정한 뒤 `~/.codex/config.toml` 에 저장하면 Codex가 해당 설정을 사용합니다.
