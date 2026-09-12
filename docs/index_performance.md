# Index performance measurement

Run the readiness check with the interpreter intended for indexing:

```powershell
python scripts/benchmark_index.py --check
```

The default command is also `--check`. It reads installed distribution metadata and
local model-cache paths only; it does not import Torch, instantiate an encoder,
contact the network, install dependencies, or download weights. Exit status `2`
means dependencies or cached model files were not detected. File presence does not
prove a complete cache or compatible runtime. The real run supplies that check.
Use `--model C:\models\bge-m3` for existing local weights, or select an interpreter
that already has the required dependencies. No installation is performed by this script.

Explicit real run on a bounded email JSON sample:

```powershell
python scripts/benchmark_index.py --run --input data/emails.json --output work/index-benchmark.json --embedding-batch-size 32 --cpu-threads 0
```

Model/device/revision defaults come from the current process environment, falling
back to `BAAI/bge-m3`, `cpu`, and no revision pin. This launcher deliberately does
not load `.env` during readiness checks; pass model options explicitly if your
application's `.env` differs. CPU threads `0` leaves Torch's existing default;
positive values request that number of threads. Encoder batch size must be 1–512. The encoder batch size and Chroma
write batch size (`--write-batch-size`, default `64`) are independent.

The real run sets Hugging Face and Transformers offline flags before child-process
imports. An incomplete cache fails with a recorded error. Only the separate
`--allow-download` option permits a model download; existing offline environment
flags are still respected. This option is unnecessary for an already cached model.

The input is copied once, in bounded blocks, into a private temporary directory.
Its SHA-256 and byte count are recorded. The default maximum is 32 MiB, 10,000 emails,
and 10,000 chunks; corresponding `--max-input-mb`, `--max-emails`, and `--max-chunks`
options can explicitly raise those limits. Per-process timeout defaults to 1,800
seconds. Input must be supplied explicitly. Production index paths are overridden
before configuration imports, and every run owns fresh scratch index directories.
Scratch input and indexes are removed on completion; only the report persists.

## Scenarios and interpretation

Two fresh child processes each execute an initial build and an unchanged repeat:

| Process | Initial build | Unchanged repeat |
| --- | --- | --- |
| `forced_baseline` | Fresh index, `force_reembed=True` | Recalculate all embeddings |
| `incremental` | Fresh index, `force_reembed=False` | Reuse eligible vectors |

The baseline is the **same final code with forced recalculation**, not a historical
implementation. Both initial builds must compute the first corpus. These scenarios
do not demonstrate a first-build cache speedup. The meaningful reuse comparison is
the forced unchanged repeat versus the incremental unchanged repeat. No small-change
scenario is included in this initial benchmark.

Reuse is scoped to the same email with exactly matching embedding text; it is not
a cache shared across unrelated emails. Unchanged updates still scan and verify
the full base generation, so that work remains O(N) even when embedding calls are
zero. The benchmark therefore reports total time alongside encoder savings.

Initial builds start with a cold **process model cache**, while unchanged repeats
use the model loaded in that process. Disk-cached/downloaded model files and OS page
cache are separate: the second child can benefit from OS cache, and this script does
not flush it. Times are single observations; run multiple times under comparable
machine load before making performance claims. Initial and repeat phases share one
frozen input, with checked chunk counts.

Each phase reports real `embedding_calls` and `embedding_texts` by observing
`embed_texts`; these count encoder API invocations and input texts, not internal
Torch minibatches. The core metrics report provides `stages_seconds` for
`preprocessing`, `model_load`, `embedding`, `write`, and `verify`, plus staging,
comparison, copying, and publication when applicable. Nested stage timers account
for exclusive time; `unaccounted_seconds` and total wall time cover remaining work.
Preprocessing validates/cleans input and counts chunks; the repeatable chunk plan's
second iteration is included in source spooling. An absent stage means it did not
execute, not an estimated duration. `model_load` includes actual initialization and
any permitted cache/download work. Download and model initialization durations
cannot automatically be separated: no zero or estimated download timing is supplied.
The report labels model files as `cached_only_offline` or
`download_allowed_unmeasured`; allowing downloads does not prove one occurred.
`total_seconds` includes phase work; report
`orchestration_seconds` also includes frozen input copying and process startup.

The JSON includes the effective indexing configuration, dependency versions,
input digest, phase counts, timings, outcome, and errors. Writes use a temporary
file plus atomic replacement. Each worker prints concise phase start/end progress
to stderr. Failure exits `2` and preserves completed phases plus available failed
phase timings/counts. Hard process termination or timeout can preserve only the
last checkpoint; an unfinished phase is labelled `running`, with unavailable
timings omitted. Exceptions are recorded by type/code, without provider or input
message text. The report contains no email bodies but does include a corpus hash
and model configuration. No real BGE-M3 throughput is claimed until this script has
successfully run with the actual local runtime and weights.
