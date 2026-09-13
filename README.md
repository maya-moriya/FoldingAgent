# FoldingAgent: Inferring Parametric Origami Procedures from Demonstration Videos

**SIGGRAPH Asia 2026**

Maya Moriya<sup>1</sup>,
Sigal Raab<sup>1</sup>,
Yael Vinker<sup>2</sup>,
Tali Dekel<sup>1</sup>

<sup>1</sup>Weizmann Institute of Science &nbsp;&nbsp; <sup>2</sup>MIT

[**Paper**](https://arxiv.org/abs/2609.00377) ·
[**Project page**](https://maya-moriya.github.io/origami-page/) ·
[**Dataset**](https://huggingface.co/datasets/mayaweiz/PurelandFold) ·
[**Results**](https://maya-moriya.github.io/origami-page/FoldingAgentResults/overview.html)

> Given an instructional origami video, FoldingAgent reconstructs the full folding
> process: from a sequence of keyframes to a sequence of 3D geometric states, enabling
> re-rendering, editing, and analysis.

## Abstract

We present *FoldingAgent*, an agentic framework for inferring explicit parametric folding programs directly from Origami demonstration videos. Our framework leverages the reasoning power of a pre-trained Vision-Language Model (VLM) equipped with a suite of specialized tools that enable the agent to simulate geometric transitions, verify physical plausibility, retrieve and compare visual content, and evaluate its own predictions. To translate visual content into folding programs, we define a parametric space that consists of the paper’s geometry and a set of parametric folding actions. Unlike models that predict static crease patterns, our agent operates sequentially and possesses the ability to re-plan its actions, effectively mitigating the compounding errors inherent in multi-step folding. Our approach takes a step toward closing the gap between human origami knowledge, which is primarily shared through unstructured visual demonstrations, and computational methods, which typically rely on structured, parametric representations such as a crease pattern or an executable parametric plan. We evaluate our approach on PurelandFold, a newly curated benchmark of diverse Pureland origami videos with ground-truth geometry and action labels. Our results demonstrate that by combining VLM reasoning with a set of specialized tools and physical simulation, we can successfully transform unstructured visual demonstrations into executable, physically plausible folding procedures. Project page: https://maya-moriya.github.io/origami-page.

## How it works

A pre-trained VLM agent reconstructs the folding sequence one keyframe transition at a
time, in a ReAct-style loop of reasoning and tool calls:

| Component | In the paper | In this code |
| --- | --- | --- |
| **Agent** | The VLM that plans each transition and calls tools | `main.py` — `run_agent()` and the backends in `backends/` |
| **Simulator** | Executes parametric actions — `add_vertex`, `fold`, `unfold`, `rotate`, `flip` — on the paper's geometry and layer structure | `simulator.py`, over [FoldingAgentSimulator](https://github.com/maya-moriya/FoldingAgentSimulator) |
| **Critic** | A separate VLM that compares a four-image grid of photos and renders, returning *Match*, *Mismatch* or *Extreme Divergence* | `critic.py`, plus `overview_critic.py` for the whole sequence |
| **Controller** | The deterministic layer that dispatches tool calls and keeps the simulator state and history in sync | `controller.py` |
| **Checkpoints and rollback** | Save a verified state; restore it to re-plan a transition | `memory.py` |
| **Selector** | When a keyframe's attempt budget (5) is exhausted, a lightweight agent picks the best attempt | `comparator.py` |

## Dataset: PurelandFold

[PurelandFold](https://huggingface.co/datasets/mayaweiz/PurelandFold) is the benchmark
introduced in the paper: **27** Pureland origami folding sequences, averaging 12 keyframes
(5–21), **337** keyframes in total. Each keyframe pairs a real photograph with the exact
geometric state of the paper at that step — the crease pattern in FOLD format (vertices,
faces, edges, mountain/valley assignment, fold angles and face stacking order) and as SVG.
It is distributed as Parquet under **CC-BY-4.0**.

## Install

Requires **Python 3.11+**.

```bash
git clone https://github.com/maya-moriya/FoldingAgent.git
cd FoldingAgent
uv sync
```

`uv sync` installs everything, including the folding engine — which lives in a separate
repository, [FoldingAgentSimulator](https://github.com/maya-moriya/FoldingAgentSimulator)
(distribution `foldingagent-simulator`, imported as `origami`). `uv.lock` pins it to an
exact commit, so a clone reproduces the same engine.

## API keys

The provider is inferred from the model id, and only that provider's key is read — from the
environment or from a `.env` file in the repository root.

| Model prefix | Provider | Environment variable |
| --- | --- | --- |
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-*`, `o1`, `o3`, `o4-*` | OpenAI | `OPENAI_API_KEY` |
| anything else (e.g. `gemini-*`) | Google | `GOOGLE_API_KEY` |

The critic, overview critic and comparator all use the same model and key as the agent.

## Run

```bash
export GOOGLE_API_KEY=...     # or ANTHROPIC_API_KEY / OPENAI_API_KEY, or put it in .env

python -m foldingagent.main data/heart/keyframes.json
```

That is the whole invocation. The manifest supplies the frames *and* the paper colours;
everything else is a setting, not an argument.

| Argument | Default | Meaning |
| --- | --- | --- |
| `keyframes` | *(required)* | Path to a `keyframes.json` manifest. |
| `--resume` | *(none)* | An existing run directory to continue in place. |

From Python it is the same two:

```python
from foldingagent.main import run_agent

checkpoints = run_agent("data/heart/keyframes.json")
```

### Settings

The model, the run limits and where output goes live in
[config.py](src/foldingagent/config.py) — edit them there rather than passing them per
run:

| Constant | Default | Meaning |
| --- | --- | --- |
| `DEFAULT_MODEL` | `gemini-3.1-pro-preview` | Model id; selects the provider and key. |
| `OUT_DIR` | `out` | Root for run output; each run creates `<OUT_DIR>/run_<timestamp>_<model>_<sequence>/`. |
| `MAX_ITERATIONS` | `300` | Hard limit on LLM round-trips. |
| `MAX_ATTEMPTS_PER_FRAME` | `5` | Attempts at one frame before the comparator picks the best. |
| `MAX_CONSECUTIVE_NO_TOOL_RESPONSES` | `3` | Tool-less replies tolerated before the context is rewound. |
| `DEFAULT_FRONT_COLOR` / `DEFAULT_BACK_COLOR` | `white` / `lightblue` | Fallbacks for a manifest that omits its colours. |

## Data layout

```
data/heart/
  keyframes.json          the manifest: frame directory, paper colours, keyframe indices
  frames/                 frames named by global index alone: 000004.jpg
```

`keyframes.json` is the single source of truth for a sequence:

```json
{
  "name": "heart",
  "frames": "frames",
  "front_color": "white",
  "back_color": "red",
  "keyframes": [4, 10, 38, 50, ...]
}
```

`frames` is resolved relative to the manifest, and each index names the file in that
directory. Keyframes are not duplicated into a second folder — they are the frames at
those indices. A manifest that omits its colours falls back to the defaults in `config.py`.

## Output

Each run directory is named `run_<timestamp>_<model>_<sequence>/`, so a listing
says which model folded which sequence:

| Path | Contents |
| --- | --- |
| `metadata.json` | What the run was and how it went: model, sequence, outcome, tokens, and the settings in force. A resumed run keeps the earlier attempt under `previous`. |
| `checkpoints.json` | Final geometry for every solved frame. |
| `frames.json` | The resolved frame list, so a run can be resumed in place. |
| `logs/agent_log.jsonl` | The full event stream — the only input `viewers/viewer.html` needs. |
| `logs/history_log.json` | Chronological record of checkpoints, critic verdicts and rollbacks. |
| `logs/attempt_tree.json` | The branching tree of attempts per frame. |
| `figs/iter_<n>_<tool>.png` | Every figure, named for the turn and the tool that drew it. |
| `llm_calls/` | Every request/response pair, for offline inspection. |

Open `viewers/viewer.html` in a browser and load a run's `logs/agent_log.jsonl` to step
through the reconstruction.

## When a request fails

Any error from any provider — a 503, a dropped connection, a bad key, anything the SDK
raises — ends the run rather than escaping as a traceback. The failure is identified
(backend, model, exception type, HTTP status, message), printed, and written to
`agent_log.jsonl` as a `backend_error` event; the run finishes with
`stop_reason = "backend_error:<type>"`.

Frames solved so far are checkpointed as usual, so nothing is lost — continue the same
run directory in place:

```bash
python -m foldingagent.main data/heart/keyframes.json --resume out/run_20260902_101010_gemini-3.1-pro-preview_heart
```

## Package layout

```
src/foldingagent/
  main.py             run_agent() — the agent loop — and the CLI
  controller.py       OrigamiController: the tool implementations, and dispatch()
  tools.py            the tool schema the model is given (Anthropic format, canonical)
  simulator.py        the geometry model and the engine that transforms it
  memory.py           checkpoints, history log, attempt tree
  logger.py           agent_log.jsonl + LLM call recording
  rendering.py        geometry -> PNG, and the paper colours it uses
  critic.py           per-step visual critic + its 2x2 grid
  overview_critic.py  whole-sequence critic + its overview grid
  comparator.py       picks the best attempt for a frame
  prompts.py          every string sent to a model (text only)
  prompt_assembler.py fills the prompt templates with runtime values
  config.py           every default constant
  backends/           one module per provider behind a shared LLMBackend ABC
```

## `observe_movement` and the frame subset

The `observe_movement` tool shows the agent what happened *between* two keyframes, by
sampling 5 evenly spaced frames from the raw video dump in `data/heart/frames/`.

Shipping all 541 frames would add 41 MB to the repository, so it ships **65** — exactly
the ones the tool samples for the keyframes in `keyframes.json` (5.0 MB). The
sampler takes 5 evenly spaced frames from each step's range and returns the range whole
when it holds 5 or fewer, so a directory pruned to precisely those frames yields exactly
the same filmstrips. `scripts/check_movement_frames.py` verifies this.

The subset is tied to the shipped keyframes. A manifest listing different indices spans
different ranges, so `observe_movement` will have fewer frames to sample — it degrades to
whatever falls in range, and errors if that is fewer than two. Ask the authors for the
full dump to run `observe_movement` on a different selection.

## Citation

```bibtex
@inproceedings{moriya2026foldingagent,
  title     = {FoldingAgent: Inferring Parametric Origami Procedures from Demonstration Videos},
  author    = {Moriya, Maya and Raab, Sigal and Vinker, Yael and Dekel, Tali},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
