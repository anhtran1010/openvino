# Ternary-Bonsai 8B on the OpenVINO GPU plugin with the TernOCL int2 kernels — step by step

From an empty directory to Ternary-Bonsai 8B decoding on an Intel Xe2 GPU (discrete
or integrated) through OpenVINO GenAI, with performance from `llm_bench`, accuracy
from `who_what_bench`, and the exact list of GEMM shapes the two TernOCL kernel
families execute. How the int2 `FullyConnected` path works is described in
[ternocl_int2_architecture.md](ternocl_int2_architecture.md); the 27B guide is
[ternocl_int2_bonsai2_from_scratch.md](ternocl_int2_bonsai2_from_scratch.md).

Ternary-Bonsai 8B is a plain Qwen3-8B (`Qwen3ForCausalLM`, 36 layers, hidden 4096,
intermediate 12288, 32 query / 8 KV heads of 128, untied 151669-row `lm_head`) whose
linear weights are trained ternary. Unlike Bonsai 2 27B it has no Hadamard rotation
and no vision tower, so it needs no GGUF converter and no template IR: export the
published fp16 checkpoint, rewrite its weights as `u2`, run it with `LLMPipeline`.

The fp16 checkpoint `prism-ml/Ternary-Bonsai-8B-unpacked` is exactly ternary with
one scale per 128-group along K in every linear layer, `lm_head` and the embedding
included (about 35% zeros), so the `u2` rewrite in step 3 reproduces it bit for bit.

---

## 0. What you need

| | |
|---|---|
| GPU | Intel Xe2, discrete (Arc Pro B-series) or integrated (Lunar Lake Arc 140V) |
| GPU runtime | Intel compute runtime with OpenCL — the int2 path runs on the plugin's **OpenCL** runtime |
| Compiler | GCC (tested with 14.3); Intel oneAPI `icx`/`icpx` also works, see the note in step 2 |
| Tools | CMake ≥ 3.26, Ninja, git, Python 3.11+ |
| Disk | ~75 GB: OpenVINO + GenAI build ~35 GB, fp16 checkpoint 16.4 GB, fp16 IR 16.4 GB, final IR 2.1 GB |
| RAM | ~32 GB on the machine that runs the export (step 3 loads the 8B in fp16); add swap if you have less |

The environment every command needs (put it in a file and `source` it):

```bash
# env.sh
export WORK=$HOME/ov-bonsai8b
export OV_ROOT=$WORK/openvino
export TOOLS=$OV_ROOT/src/plugins/intel_gpu/tools/int2
export ZE_AFFINITY_MASK=0                    # multi-GPU hosts: which GPU to use
```

and, for the GPU runs of steps 4-5, the OpenVINO + GenAI build of step 2:

```bash
# genai.sh
source $WORK/env.sh
source $OV_ROOT/install/setupvars.sh
source $WORK/bench-venv/bin/activate
```

---

## 1. Download the sources

```bash
mkdir -p $WORK && cd $WORK
git clone https://github.com/openvinotoolkit/openvino.git openvino
cd openvino && git submodule update --init --recursive && cd ..      # includes thirdparty/TernOCL
git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git openvino.genai   # includes openvino_tokenizers
```

The OpenCL kernels come from [TernOCL](https://github.com/libxsmm/TernOCL), pinned
as the submodule `src/plugins/intel_gpu/thirdparty/TernOCL`. If you already have a
TernOCL checkout at the pinned commit (`git -C openvino submodule status
src/plugins/intel_gpu/thirdparty/TernOCL`), you can skip the submodule and pass
`-DTERNOCL_ROOT=<that checkout>` in step 2. The same checkout builds the standalone
drivers of section 6.4.

## 2. Build OpenVINO with GenAI, and the benchmark environment

Identical to step 2 of the [27B guide](ternocl_int2_bonsai2_from_scratch.md#2-build-openvino-with-genai-and-the-benchmark-environment),
with debug caps switched on so that step 6 can list the shapes (`-DENABLE_DEBUG_CAPS=ON`
only adds logging; leave it off for a production build).

```bash
source env.sh && cd $WORK
python3 -m venv bench-venv
for t in llm_bench who_what_benchmark; do
  bench-venv/bin/pip install -r openvino.genai/tools/$t/requirements.txt
done
bench-venv/bin/pip install --no-deps -e openvino.genai/tools/who_what_benchmark
bench-venv/bin/pip uninstall -y openvino openvino-tokenizers openvino-genai   # use this build's packages instead
```

```bash
cd $OV_ROOT
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$OV_ROOT/install \
  -DOPENVINO_EXTRA_MODULES=$WORK/openvino.genai \
  -DENABLE_PYTHON=ON -DPython3_EXECUTABLE=$WORK/bench-venv/bin/python \
  -DGPU_RT_TYPE=OCL -DENABLE_SYSTEM_OPENCL=OFF -DENABLE_DEBUG_CAPS=ON \
  -DENABLE_INTEL_CPU=ON -DENABLE_INTEL_NPU=OFF \
  -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF \
  -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=OFF \
  -DENABLE_OV_TF_FRONTEND=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF
  # optional: -DTERNOCL_ROOT=<existing TernOCL checkout>
cmake --build build -j $(nproc) && cmake --install build
```

With `icx`/`icpx`, add the flags from the 27B guide (`-fp-model=precise`,
`-DENABLE_XGRAMMAR=OFF`).

```bash
source $WORK/genai.sh
python -c "import openvino as ov, openvino_genai; print(ov.get_version(), ov.Core().available_devices)"   # ... ['CPU', 'GPU']
```

Python side of the model preparation (step 3; no GPU, no OpenVINO build needed — it
uses the pip wheels in a separate venv):

```bash
cd $WORK
python3 -m venv venv && ./venv/bin/pip install openvino openvino-tokenizers numpy
./venv/bin/pip install "optimum-intel[openvino]==2.1.0" "transformers==5.2.0"
```

## 3. Download Ternary-Bonsai 8B and build its IR

```bash
cd $WORK
./venv/bin/hf download prism-ml/Ternary-Bonsai-8B-unpacked --local-dir tbonsai8b-hf     # 16.4 GB, 4 safetensors shards

# fp16 export; a Qwen3 checkpoint is a plain LLM, so the task is text-generation-with-past
./venv/bin/optimum-cli export openvino --model tbonsai8b-hf \
    --task text-generation-with-past --weight-format fp16 tbonsai8b-fp16

# rewrite the 253 ternary MatMuls (36 layers x 7 + lm_head) as u2 codes + fp16 group scales
./venv/bin/python $TOOLS/quantize_ir_ternary.py \
    --in  tbonsai8b-fp16/openvino_model.xml \
    --out tbonsai8b-u2/openvino_model.xml
# expected: "rewriting 253 MatMul weights" ... "weights 14.09 GiB -> 1.87 GiB"

cp tbonsai8b-fp16/*.json tbonsai8b-fp16/openvino_*tokenizer.* tbonsai8b-u2/
cp tbonsai8b-hf/chat_template.jinja tbonsai8b-u2/ 2>/dev/null || true
```

`tbonsai8b-u2/` is the GenAI model directory. The token embedding is a `Gather`, not
a `MatMul`, so it stays fp16 (1.16 GiB); the TernOCL path only covers
`FullyConnected`. Keep `tbonsai8b-fp16/` if you want the fp16 reference of step 5.2;
otherwise delete it and `tbonsai8b-hf/`.

## 4. Run

```bash
source $WORK/genai.sh
```

### 4.1 Performance (`llm_bench`)

Decode and a short prompt:

```bash
cd $WORK/openvino.genai/tools/llm_bench
python benchmark.py -m $WORK/tbonsai8b-u2 -d GPU \
    -p "Explain photosynthesis in two sentences." --disable_prompt_permutation -ic 256 -n 3
```

That prompt is about 8 tokens, so its prefill also runs on the GEMV kernel (M ≤ 8,
section 6). To measure prefill on the M-tiled kernel, give it a long prompt:

```bash
python - <<'EOF'
import json
text = " ".join(["The quick brown fox jumps over the lazy dog."] * 100)    # ~1k tokens
json.dump({"prompt": text}, open("prompt_1k.jsonl", "w"))
EOF
python benchmark.py -m $WORK/tbonsai8b-u2 -d GPU -pf prompt_1k.jsonl \
    --disable_prompt_permutation -ic 128 -n 3
# again with the int2 x int8 prefill kernel:
OV_GPU_TERNOCL_INT2_INT8_PREFILL=1 python benchmark.py -m $WORK/tbonsai8b-u2 -d GPU -pf prompt_1k.jsonl \
    --disable_prompt_permutation -ic 128 -n 3
```

The warm-up iteration builds the OpenCL programs; the average is over the following
iterations. Reference point from the 27B guide: 168 tokens/s decode on an Arc Pro B70.
Batch-1 decode is partly host-bound; compare builds on one machine.

### 4.2 Accuracy (`who_what_bench`)

The fp16 model is the natural reference, since the `u2` weights are exactly the fp16
ones:

```bash
cd $WORK/openvino.genai/tools/who_what_benchmark
wwb --base-model $WORK/tbonsai8b-fp16 --gt-data gt_fp16.csv --model-type text --genai --device GPU
wwb --target-model $WORK/tbonsai8b-u2 --gt-data gt_fp16.csv --model-type text --genai --device GPU
```

On an integrated GPU the 16 GB fp16 model may not fit next to everything else in
shared memory. Then use the same `u2` model on the stock kernels as the reference;
the score then isolates what the TernOCL kernels change (the stock path runs `u2` on
the reference FC kernel, at a few tokens/s):

```bash
OV_GPU_ENABLE_TERNOCL_INT2=0 wwb --base-model $WORK/tbonsai8b-u2 --gt-data gt_stock.csv \
    --model-type text --genai --device GPU
wwb --target-model $WORK/tbonsai8b-u2 --gt-data gt_stock.csv --model-type text --genai --device GPU
```

Reported on an Arc Pro B70: 0.988 similarity between TernOCL and the stock kernels.

## 5. The `lm_head` is not on TernOCL as exported

`lm_head` is `[151669, 4096]`. 151669 is not a multiple of 16, and the impl rejects
any FC with `N % 16 != 0` (`fully_connected_ternocl_int2.hpp`, "N is not a multiple of
16"). The 4096 × 151680 entries in the tile tables and in TernOCL's `bench.sh` are the
benchmark stand-in, rounded up to 16, not the shape the plugin sees. As exported, the
`u2` `lm_head` therefore falls through to the only stock FC kernel that reads `u2`,
`fully_connected_gpu_bfyx_ref`. The output is still correct, but the reference kernel
is slow on the largest layer of the model (0.15 GiB of `u2` weights per token).

Because of GenAI's slice-before-matmul transformation, `lm_head` always runs with
M = 1 (batch 1), in prefill and in decode alike.

Options, until the impl accepts a ragged N:

| Option | How | Cost |
|---|---|---|
| keep it (default above) | nothing | `lm_head` on the reference `u2` kernel |
| dense fp16 `lm_head` | `--skip-n 151669` on `quantize_ir_ternary.py` | runs on the tuned fp16 FC kernel, but reads 1.16 GiB per token instead of 0.15 GiB (decode is bandwidth-bound; this costs most on integrated GPUs) |
| pad N to 151680 | zero rows 151669..151679 and mask their logits | needs graph surgery the GenAI pipeline still recognises (`MatMul → Add → Result`, with the 11 pad logits forced to −inf by the Add), and the bias must have the FC output type to fold into the epilogue (`POSTOP 3`); not validated |
| accept N % 16 ≠ 0 in the impl | plugin change; the kernels already clip partial tiles | development work |

Check which way it went in a run with `OV_VERBOSE=4` (section 6.3): the reject line
names the reason.

## 6. Shapes the TernOCL kernels execute

### 6.1 FullyConnected layers after the plugin's fusions

The plugin merges FCs before choosing implementations: q/k/v share an input, so
`FullyConnectedHorizontalFusion` merges them into one FC (N = 4096 + 1024 + 1024);
gate/up are always merged for `u2` models on the TernOCL path (N = 2 × 12288), with
SwiGLU as a separate primitive after the merged FC (FC+SwiGLU fusion is limited to
`u4`/`i4` on non-systolic GPUs). The residual adds after `o_proj` and `down_proj`
fold into the epilogue.

| FC | K | N | Epilogue (`POSTOP`) | Out | Per forward | `u2`+scales per call | TernOCL |
|---|---:|---:|---|---|---:|---:|---|
| q/k/v (merged) | 4096 | 6144 | 0 none | f16 | 36 | 6.7 MB | yes |
| o_proj | 4096 | 4096 | 2 residual add | f16 | 36 | 4.5 MB | yes |
| gate/up (merged) | 4096 | 24576 | 0 none, SwiGLU follows | f16 | 36 | 26.7 MB | yes |
| down_proj | 12288 | 4096 | 2 residual add | f16 | 36 | 13.4 MB | yes |
| lm_head | 4096 | 151669 | — | f32 | 1 (M = 1) | 165 MB | **no**, section 5 |

So 144 FCs per forward run on TernOCL, over four distinct (K, N) shapes, and decode
streams 1.85 GB of TernOCL weights per token. All K are multiples of 128, so every
group scale lines up; there is no Hadamard transform (`hadamard_fwht_1024` never runs).

### 6.2 Which kernel runs for which M

M is the number of token rows in the call: the prompt length in prefill, 1 in
batch-1 decode. The impl picks the launch class from M and builds each class's
program on first use:

| Kernel family | Kernel | M | When |
|---|---|---|---|
| int2 x f16 up-convert | `int2_fp16_upcvt_gemm` (GEMV) | 1 / 2 / 3-4 / 5-8 (`SGM` = 1/2/4/8) | every decode step; prefill of prompts ≤ 8 tokens |
| int2 x f16 up-convert | `int2_fp16_upcvt_gemm_mt` | 9-16 / 17-32 / 33-63 / ≥ 64 | prefill (default) |
| int2 x int8 DPAS | `quant_a` + `int2_int8_gemm_mt` | 9-16 / 17-32 / 33-63 / ≥ 64 | prefill with `OV_GPU_TERNOCL_INT2_INT8_PREFILL=1` |

The complete set of kernel invocations is the product of the four shapes in 6.1 with
these M classes:

| (K, N) | GEMV tile `wgn/ls/u`, B70 · Arc 140V | `gemm_mt` tile, M < 64 (`mt_m/mt_n/wg_m/wg_n`), B70 · Arc 140V | `gemm_mt` tile, M ≥ 64, B70 · Arc 140V | int8 tile |
|---|---|---|---|---|
| (4096, 6144) | 16/4/1 · 64/1/1 | narrow band · i_narrow band | 64/32/4/4 · default 128/16/2/4 | default per band |
| (4096, 4096) | 32/8/1 · 32/2/1 | narrow · i_narrow | 64/16/2/4 · default 128/16/2/4 | default per band |
| (4096, 24576) | 64/1/1 · 16/2/1 | wide · i_wide | 64/16/2/2 · default 128/16/2/4 | default per band |
| (12288, 4096) | 32/8/1 · 32/2/1 | narrow · i_narrow | 64/32/4/2 · default 128/16/2/4 | default per band |

The GEMV tiles are tuned for these 8B shapes on both GPUs. The M < 64 tiles are
shared per output-width class (narrow N ≤ 8192, wide N < 65536). The M ≥ 64 tiles are
tuned for 8B only on the B70; the Arc 140V and the int8 kernel use the default tile
of the launch class (correct, but not tuned for 8B). See section 6 of the
architecture doc for how to tune them.

### 6.3 Confirming the list on a run

With the debug-caps build of step 2:

```bash
source $WORK/genai.sh
cd $WORK/openvino.genai/tools/llm_bench
OV_VERBOSE=4 python benchmark.py -m $WORK/tbonsai8b-u2 -d GPU -pf prompt_1k.jsonl \
    --disable_prompt_permutation -ic 4 -n 1 2>&1 | grep '\[ternocl-int2\]' > ternocl.log

grep ' create '  ternocl.log | sed 's/.* N=\([0-9]*\) K=\([0-9]*\) postop=\([0-9]\) out_f32=\([01]\).*/K=\2 N=\1 postop=\3 out_f32=\4/' | sort | uniq -c
grep ' reject '  ternocl.log | sed 's/^.*: //' | sort | uniq -c     # expect the lm_head: "N is not a multiple of 16"
grep 'M-class'   ternocl.log | sed 's/:.*//' | sort -u               # one line per (K, N, M class, [int8]) program built
```

Expected `create` summary (counts are per program build; dynamic-shape impls are
cloned, so only the distinct lines matter):

```
K=4096 N=6144 postop=0 out_f32=0
K=4096 N=4096 postop=2 out_f32=0
K=4096 N=24576 postop=0 out_f32=0
K=12288 N=4096 postop=2 out_f32=0
```

and `M-class 0` (decode) plus `M-class 7` (the 1k-token prefill) for each of the four
shapes, or `M-class 7 int8` with `OV_GPU_TERNOCL_INT2_INT8_PREFILL=1`.

### 6.4 Reproducing the shapes on the standalone kernels

Each shape and M can be timed and validated outside OpenVINO with the TernOCL drivers
(any checkout at the pinned commit):

```bash
cd <TernOCL checkout>
make -C int2_fp16_upcvt CXX=g++ && make -C int2_via_int2_x_int8_dpas CXX=g++
for s in 4096:6144:0 4096:4096:2 4096:24576:0 12288:4096:2; do
  IFS=: read K N P <<< $s
  ./int2_fp16_upcvt/build/int2_fp16_upcvt_ocl --m 1    --k $K --n $N --postop $P --iters 50   # decode GEMV
  ./int2_fp16_upcvt/build/int2_fp16_upcvt_ocl --m 1024 --k $K --n $N --postop $P --iters 20   # prefill gemm_mt
  ./int2_via_int2_x_int8_dpas/build/int2_int8_dpas_ocl --m 1024 --k $K --n $N --postop $P --iters 20
done
```

Pass the tile flags (`--wgn/--ls/--u`, `--mt-m/--mt-n/--wg-m/--wg-n`) from 6.2 to time
the tile the plugin actually uses; without them the driver picks its own default.
The drivers require N % 16 == 0, so the real `lm_head` (N = 151669) cannot be run there
either.

## 7. Knobs and troubleshooting

The knobs (`GPU_TERNOCL_INT2_INT8_PREFILL`, `GPU_ENABLE_TERNOCL_INT2`,
`GPU_ENABLE_RMS_ROPE_FUSION`) and the troubleshooting table of the
[27B guide](ternocl_int2_bonsai2_from_scratch.md#6-knobs-that-matter) apply unchanged.
`GPU_ENABLE_HADAMARD_FC_FUSION` has no effect on this model. `GPU_ENABLE_RMS_ROPE_FUSION`
does: Qwen3-8B rotates the whole 128-wide head, so the per-head Q/K RMSNorm folds into
RoPE by default.

| Symptom | Cause / fix |
|---|---|
| optimum export killed (OOM) | the fp16 export holds the 16 GB model plus tracing buffers; add swap or export on a bigger machine — the IR is portable |
| `rewriting 0 MatMul weights` | wrong input: pass `openvino_model.xml` of the fp16 export, not a `--weight-format int4/int8` export |
| fluent nonsense output | tokenizer files missing or from another model: copy `openvino_*tokenizer.*` and the `*.json` from the fp16 export |
