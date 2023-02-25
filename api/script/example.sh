WORK_DIR="${HOME}/LLM4RS"
cd ${WORK_DIR}

TASK_NAME="example"
if [ ! -d "data/${TASK_NAME}" ]; then
  mkdir "data/${TASK_NAME}"
fi
REQUESTS_FILEPATH="data/${TASK_NAME}/request.jsonl"
SAVE_FILEPATH="data/${TASK_NAME}/response.jsonl"

REQUESTS_URL="https://api.openai.com/v1/completions"

API_KEY="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"

TOKEN_ENCODING_NAME="p50k_base"

MAX_ATTEMPTS=5

python api/api_request_parallel_processor.py \
    --requests_filepath ${REQUESTS_FILEPATH} \
    --save_filepath ${SAVE_FILEPATH} \
    --request_url ${REQUESTS_URL} \
    --api_key ${API_KEY} \
    --max_requests_per_minute 1500 \
    --max_tokens_per_minute 6250000 \
    --token_encoding_name ${TOKEN_ENCODING_NAME} \
    --max_attempts ${MAX_ATTEMPTS} \
    --logging_level 20