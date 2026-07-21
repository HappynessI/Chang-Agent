#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python"
CSV="$ROOT/默认业务空间-apiKey-6214720.csv"
NODE="${NODE:-}"
BAILIAN_NETWORK_MODE="${BAILIAN_NETWORK_MODE:-}"

if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE is required" >&2
  exit 2
fi
if [[ "$BAILIAN_NETWORK_MODE" != "direct" && "$BAILIAN_NETWORK_MODE" != "proxy" ]]; then
  echo "BAILIAN_NETWORK_MODE must be direct or proxy" >&2
  exit 2
fi
if ! sinfo -N -n "$NODE" -h -o '%T' | grep -Eq 'idle|mix|alloc'; then
  echo "node is not available: $NODE" >&2
  exit 2
fi

proxy_fingerprint() {
  env | LC_ALL=C awk -F= 'tolower($1) ~ /^(http_proxy|https_proxy|all_proxy|no_proxy)$/ {print}' \
    | LC_ALL=C sort | sha256sum | awk '{print $1}'
}

MANAGEMENT_PROXY_FINGERPRINT=$(proxy_fingerprint)
TMP_PREFIX="/tmp/CA_0721_bailian_probe"
SHARED_LOG_DIR="${PROBE_LOG_DIR:-/guisongxia01/pangchao/wangyihan/wyh}"
PROBE_PY=$(printf '%q' "$PYTHON")
ROOT_Q=$(printf '%q' "$ROOT")
CSV_Q=$(printf '%q' "$CSV")
MANAGEMENT_FP_Q=$(printf '%q' "$MANAGEMENT_PROXY_FINGERPRINT")

if [[ "$BAILIAN_NETWORK_MODE" == "direct" ]]; then
  NETWORK_PREFIX='env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy'
else
  NETWORK_PREFIX='env'
fi

WRAP="set -eu; proxy_fingerprint() { env | LC_ALL=C awk -F= 'tolower(\$1) ~ /^(http_proxy|https_proxy|all_proxy|no_proxy)$/ {print}' | LC_ALL=C sort | sha256sum | awk '{print \$1}'; }; test -d ${ROOT_Q}; test -x ${PROBE_PY}; test -f /guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt; test -f /guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth; ${PROBE_PY} -c 'import numpy, PIL; print(\"python_dependencies=ok\")'; nvidia-smi -L >/dev/null; echo gpu_visible=ok; echo worker_proxy_fingerprint=\$(proxy_fingerprint); if [ ${BAILIAN_NETWORK_MODE} = proxy ] && [ \"\$(proxy_fingerprint)\" != ${MANAGEMENT_FP_Q} ]; then echo proxy_fingerprint_mismatch >&2; exit 3; fi; ${NETWORK_PREFIX} ${PROBE_PY} ${ROOT_Q}/tools/run_with_bailian_csv.py ${CSV_Q} /usr/bin/env bash -c 'test -n \"\$DASHSCOPE_BASE_URL\"; curl --silent --show-error --max-time 20 --output /dev/null --write-out \"endpoint_http=%{http_code}\\n\" \"\$DASHSCOPE_BASE_URL\"'; echo endpoint_reachable=ok"

JOB=$(sbatch --parsable --job-name=CA21Probe --nodelist="$NODE" --gres=gpu:1 \
  --cpus-per-task=1 --mem=4G --time=00:05:00 \
  --output="${SHARED_LOG_DIR}/.ca_0721_bailian_probe-%j.out" \
  --error="${SHARED_LOG_DIR}/.ca_0721_bailian_probe-%j.err" --wrap="$WRAP")
printf 'job=%s\nnode=%s\nnetwork_mode=%s\nmanagement_proxy_fingerprint=%s\nstdout=%s/.ca_0721_bailian_probe-%s.out\nstderr=%s/.ca_0721_bailian_probe-%s.err\n' \
  "$JOB" "$NODE" "$BAILIAN_NETWORK_MODE" "$MANAGEMENT_PROXY_FINGERPRINT" \
  "$SHARED_LOG_DIR" "$JOB" "$SHARED_LOG_DIR" "$JOB"
