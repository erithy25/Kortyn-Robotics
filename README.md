# Kortyn

Kortyn measures how network latency and a shared GPU server affect the success of learned robot policies. A robot
(the *Agent*) executes chunks of actions from a policy and requests the next chunk while the current one runs; the
policy runs on a server (the *Brain*) that may serve many robots. Kortyn emulates the network and the Brain's
queue, runs the real policy in simulation (LeRobot policies on PushT, Meta-World and LIBERO) and records success,
either in real time or in a validated simulation-time mode that needs only a CPU.

**Status:** research code with a complete technical report. The project started as a business idea (one GPU
"Brain" for a fleet of robots); after phase 4 it is not pursued as a product. The tool and the data are public
under the Apache-2.0 license.

## Results in one paragraph

SmolVLA (`lerobot/smolvla_libero`) on two long-horizon LIBERO tasks keeps its success up to about 200 ms of
end-to-end latency and then drops steeply: 47% at 0 ms, 30% at 300 ms, 5% at 1,000 ms, far below the 1,250 ms its
action queue could bridge. LeRobot's real-time chunking (RTC) with guidance over up to 25 actions did not stretch
this curve (mean gain over the budget −1.6 pp [−6.1, +3.4] pp) and failed the check that it matches the
synchronous reference at zero latency. With guidance limited to the frozen actions plus 4, RTC passed a
pre-registered zero-latency gate (45.5% against 49.0%) but did not stretch the curve demonstrably either
(+2.6 pp [−1.7, +5.6] pp). Under an assumed service time of 220 ms per plan, one consumer GPU serves 3 robots
without RTC and 4 robots with short guidance within a 3-pp tolerance, and 2 robots by a conservative estimate; a
fleet validation ran below its prediction (−9.7 pp pooled) after we fixed a rendering fault that had invalidated
two earlier validations. The GPU service time remains an assumption until the browser measurement below has been
run. All five phases were pre-registered; the negative results are reported in full in
**[`paper/report.md`](paper/report.md)**.

## Quickstart

Requirements: Linux x86_64 or macOS arm64, [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 and the
pinned dependencies from `uv.lock` (CPU PyTorch on Linux).

```bash
git clone https://github.com/erithy25/Kortyn-Robotics && cd Kortyn-Robotics
uv sync                      # Python 3.12 + pinned packages
uv run pytest                # tests
```

**Capacity of a GPU for a robot fleet** (one command; dose-response curve + network profile + GPU):

```bash
uv run kortyn-capacity --config examples/capacity/example.yaml
uv run kortyn-capacity --config examples/capacity/example.yaml --gpu path/to/your_gpu_measurement.json
```

It simulates fleets of 1, 2, 3, … robots sharing the GPU (request rules, office Wi-Fi, batching schedulers),
predicts every robot's success from the curve at the latencies it saw, and prints per scheduling strategy the
capacity (largest fleet with at most 1% late plans and the worst robot at most 3 pp below a single robot), the
latency quantiles and the expected success. The example config documents every input; the curve file
(`examples/capacity/smolvla_libero_long.json`) holds success counts per task and latency, and
`python -m kortyn.dose_export` writes such a file from your own rollouts. With the example inputs the tool
reproduces the pre-registered phase-4 capacities exactly (`tests/test_capacity.py`).

**Measure your GPU in the browser** (free Colab or Kaggle GPU, no local install): open
[`notebooks/gpu_benchmark.ipynb`](notebooks/gpu_benchmark.ipynb) and follow
[`docs/gpu_benchmark_browser.md`](docs/gpu_benchmark_browser.md) (German, for non-programmers). The notebook
writes a JSON file that `kortyn-capacity --gpu` reads directly.

**Run policy rollouts in simulation time.** LIBERO needs its own environment (MuJoCo < 3.9):

```bash
export UV_PROJECT_ENVIRONMENT=.venv-libero
uv sync --no-group metaworld --group libero
uv run --no-sync python -m kortyn.run_virtual --config kortyn/experiments/phase4/libero_dose.yaml --share-model
```

**Rebuild the report's tables, figures and number provenance** from `results/`:

```bash
uv run python -m kortyn.paper
```

## What is in the repository

| Path | Content |
|---|---|
| `kortyn/agent.py`, `kortyn/brain.py`, `kortyn/netem_proxy.py` | real-time mode: robot agent, gRPC policy server, network emulator |
| `kortyn/virtual_time.py`, `kortyn/virtual_agent.py`, `kortyn/run_virtual.py` | simulation-time mode (discrete-event simulation of the agent clock) |
| `kortyn/fleet.py`, `kortyn/fleet_brain.py` | fleet queueing model: batching Brain, schedulers FIFO/EDF, admission control |
| `kortyn/dose_curve.py`, `kortyn/capacity.py` | dose-response curve model, capacity tool (`kortyn-capacity`) |
| `kortyn/gpu_benchmark.py`, `kortyn/notebook_bench.py` | GPU measurement file (reader, empirical service model) and the notebook's script |
| `kortyn/experiments/` | every experiment configuration (YAML) |
| `results/` | raw results of all phases (episode and request logs, evaluations) |
| `reports/` | phase reports and pre-registrations (German) |
| `paper/` | technical report (English), figures, provenance of every number |
| `docs/phases_de.md` | the previous README: phase summaries and every reproduction command (German) |

## Reports

- **[`paper/report.md`](paper/report.md)**: technical report over phases 1–5 (English).
- Phase reports (German): [`reports/phase1.md`](reports/phase1.md), [`reports/phase2.md`](reports/phase2.md),
  [`reports/phase3a.md`](reports/phase3a.md), [`reports/phase3.md`](reports/phase3.md),
  [`reports/phase4.md`](reports/phase4.md).
- Pre-registrations: [`reports/phase3.md`](reports/phase3.md) (section 0),
  [`reports/phase4_preregistration.md`](reports/phase4_preregistration.md),
  [`reports/phase5_preregistration.md`](reports/phase5_preregistration.md).

## License

[Apache License 2.0](LICENSE). We chose it because it is the license of the ecosystem Kortyn builds on (LeRobot and
the SmolVLA weights are Apache-2.0; LIBERO, robosuite and MuJoCo use compatible permissive licenses), it allows
research and commercial use without copyleft obligations, and it contains an explicit patent grant, which matters
to companies in robotics that want to evaluate or extend the tool. Third-party models and datasets keep their own
licenses.
