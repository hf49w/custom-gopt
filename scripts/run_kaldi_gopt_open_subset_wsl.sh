#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

PREPARED_DATASET_ROOT="${PREPARED_DATASET_ROOT:-}"
KALDI_GOP_ROOT="${KALDI_GOP_ROOT:-${WORKSPACE_ROOT}/kaldi/egs/gop_speechocean762/s5}"
LIBRISPEECH_EG_ROOT="${LIBRISPEECH_EG_ROOT:-${WORKSPACE_ROOT}/kaldi/egs/librispeech/s5}"
RUN_TAG="${RUN_TAG:-gopt_open_streaming_test}"
NJ="${NJ:-8}"
KALDI_CMD="${KALDI_CMD:-run.pl}"

if [[ -z "${PREPARED_DATASET_ROOT}" ]]; then
  echo "PREPARED_DATASET_ROOT is required" >&2
  exit 1
fi

if [[ ! -d "${PREPARED_DATASET_ROOT}" ]]; then
  echo "Missing PREPARED_DATASET_ROOT: ${PREPARED_DATASET_ROOT}" >&2
  exit 1
fi

MODEL_DIR="${LIBRISPEECH_EG_ROOT}/exp/chain_cleaned/tdnn_1d_sp"
IVECTOR_EXTRACTOR_DIR="${LIBRISPEECH_EG_ROOT}/exp/nnet3_cleaned/extractor"
LANG_DIR="${LIBRISPEECH_EG_ROOT}/data/lang_test_tgsmall"

TEST_PART="${RUN_TAG}_test"
LOCAL_TAG_DIR="data/local/${RUN_TAG}"
DICT_DIR="data/local/${RUN_TAG}_dict_nosp"
LANG_NOSP_DIR="data/${RUN_TAG}_lang_nosp"
ALI_DIR="exp/${RUN_TAG}_ali_${TEST_PART}"
PROB_DIR="exp/${RUN_TAG}_probs_${TEST_PART}"
GOP_DIR="exp/${RUN_TAG}_gop_${TEST_PART}"

cd "${KALDI_GOP_ROOT}"
. ./cmd.sh
. ./path.sh
cmd="${KALDI_CMD}"

for d in "${MODEL_DIR}" "${IVECTOR_EXTRACTOR_DIR}" "${LANG_DIR}"; do
  [[ -d "${d}" ]] || { echo "Missing path: ${d}" >&2; exit 1; }
done

rm -rf "data/${TEST_PART}" "${LOCAL_TAG_DIR}" "${DICT_DIR}" "${LANG_NOSP_DIR}" "${ALI_DIR}" "${PROB_DIR}" "${GOP_DIR}"
mkdir -p "data/${TEST_PART}" "${LOCAL_TAG_DIR}"
mkdir -p "${ALI_DIR}/log" "${GOP_DIR}/log"

cp "${PREPARED_DATASET_ROOT}/test/wav.scp" "data/${TEST_PART}/wav.scp"
cp "${PREPARED_DATASET_ROOT}/test/text" "data/${TEST_PART}/text"
cp "${PREPARED_DATASET_ROOT}/test/utt2spk" "data/${TEST_PART}/utt2spk"
cp "${PREPARED_DATASET_ROOT}/test/spk2utt" "data/${TEST_PART}/spk2utt"
cp "${PREPARED_DATASET_ROOT}/resource/lexicon.txt" "${LOCAL_TAG_DIR}/lexicon.txt"
cp "${PREPARED_DATASET_ROOT}/resource/text-phone" "${LOCAL_TAG_DIR}/text-phone"
cp "${PREPARED_DATASET_ROOT}/resource/scores.json" "${LOCAL_TAG_DIR}/scores.json"

utils/validate_data_dir.sh --no-feats "data/${TEST_PART}"

steps/make_mfcc.sh --nj "${NJ}" --mfcc-config conf/mfcc_hires.conf --cmd "${cmd}" "data/${TEST_PART}"
steps/compute_cmvn_stats.sh "data/${TEST_PART}"
utils/fix_data_dir.sh "data/${TEST_PART}"

steps/online/nnet2/extract_ivectors_online.sh --cmd "${cmd}" --nj "${NJ}" \
  "data/${TEST_PART}" "${IVECTOR_EXTRACTOR_DIR}" "data/${TEST_PART}/ivectors"

steps/nnet3/compute_output.sh --cmd "${cmd}" --nj "${NJ}" \
  --online-ivector-dir "data/${TEST_PART}/ivectors" \
  "data/${TEST_PART}" "${MODEL_DIR}" "${PROB_DIR}"

local/prepare_dict.sh "${LOCAL_TAG_DIR}/lexicon.txt" "${DICT_DIR}"
utils/prepare_lang.sh --phone-symbol-table "${LANG_DIR}/phones.txt" \
  "${DICT_DIR}" "<UNK>" "data/local/${RUN_TAG}_lang_tmp_nosp" "${LANG_NOSP_DIR}"

utils/split_data.sh "data/${TEST_PART}" "${NJ}"
for i in $(seq 1 "${NJ}"); do
  utils/sym2int.pl -f 2- "${LANG_NOSP_DIR}/words.txt" \
    "data/${TEST_PART}/split${NJ}/${i}/text" \
    > "data/${TEST_PART}/split${NJ}/${i}/text.int"
done
utils/sym2int.pl -f 2- "${LANG_NOSP_DIR}/phones.txt" \
  "${LOCAL_TAG_DIR}/text-phone" > "${LOCAL_TAG_DIR}/text-phone.int"

${cmd} JOB=1:${NJ} "${ALI_DIR}/log/mk_align_graph.JOB.log" \
  compile-train-graphs-without-lexicon \
    --read-disambig-syms="${LANG_NOSP_DIR}/phones/disambig.int" \
    "${MODEL_DIR}/tree" "${MODEL_DIR}/final.mdl" \
    "ark,t:data/${TEST_PART}/split${NJ}/JOB/text.int" \
    "ark,t:${LOCAL_TAG_DIR}/text-phone.int" \
    "ark:|gzip -c > ${ALI_DIR}/fsts.JOB.gz"
echo "${NJ}" > "${ALI_DIR}/num_jobs"

steps/align_mapped.sh --cmd "${cmd}" --nj "${NJ}" --graphs "${ALI_DIR}" \
  "data/${TEST_PART}" "${PROB_DIR}" "${LANG_DIR}" "${MODEL_DIR}" "${ALI_DIR}"

local/remove_phone_markers.pl "${LANG_DIR}/phones.txt" \
  "${LANG_NOSP_DIR}/phones-pure.txt" "${LANG_NOSP_DIR}/phone-to-pure-phone.int"

${cmd} JOB=1:${NJ} "${ALI_DIR}/log/ali_to_phones.JOB.log" \
  ali-to-phones --per-frame=true "${MODEL_DIR}/final.mdl" \
    "ark,t:gunzip -c ${ALI_DIR}/ali.JOB.gz|" \
    "ark,t:|gzip -c >${ALI_DIR}/ali-phone.JOB.gz"

${cmd} JOB=1:${NJ} "${GOP_DIR}/log/compute_gop.JOB.log" \
  compute-gop --phone-map="${LANG_NOSP_DIR}/phone-to-pure-phone.int" \
    --skip-phones-string=0:1:2 \
    "${MODEL_DIR}/final.mdl" \
    "ark,t:gunzip -c ${ALI_DIR}/ali.JOB.gz|" \
    "ark,t:gunzip -c ${ALI_DIR}/ali-phone.JOB.gz|" \
    "ark:${PROB_DIR}/output.JOB.ark" \
    "ark,scp:${GOP_DIR}/gop.JOB.ark,${GOP_DIR}/gop.JOB.scp" \
    "ark,scp:${GOP_DIR}/feat.JOB.ark,${GOP_DIR}/feat.JOB.scp"

cat "${GOP_DIR}"/feat.*.scp > "${GOP_DIR}/feat.scp"
cat "${GOP_DIR}"/gop.*.scp > "${GOP_DIR}/gop.scp"

cat <<EOF
{
  "kaldi_gop_root": "${KALDI_GOP_ROOT}",
  "run_tag": "${RUN_TAG}",
  "test_part": "${TEST_PART}",
  "feature_scp": "${KALDI_GOP_ROOT}/${GOP_DIR}/feat.scp",
  "gop_scp": "${KALDI_GOP_ROOT}/${GOP_DIR}/gop.scp",
  "phones_pure_txt": "${KALDI_GOP_ROOT}/${LANG_NOSP_DIR}/phones-pure.txt",
  "pseudo_scores_json": "${KALDI_GOP_ROOT}/${LOCAL_TAG_DIR}/scores.json"
}
EOF
