"""Typed configuration for all Kortyn components.

Everything that can be tuned lives in YAML files under ``kortyn/experiments/``:

* ``base.yaml``      – ports, paths, Brain profiles, Agent/Proxy parameters, the phase-1 experiment suite.
* ``<scenario>.yaml`` – one network scenario (delay, jitter, bandwidth, stalls) and the agent mode.
* ``phase2/*.yaml``   – phase-2 experiments (freshness strategies); they ``extends: ../base.yaml`` so the
  environment, checkpoint, Brain profiles and seeds are inherited unchanged.
* ``phase3a/*.yaml``  – simulation-time experiments (``python -m kortyn.run_virtual``): hardware profiles
  (service-time distributions per Brain profile) and timing models (latency model or replay) per run;
  ``smolvla_*.yaml`` switch to SmolVLA on Meta-World (``env.kind: metaworld``).

YAML is decoded with ``draccus.decode`` (the config library LeRobot uses), which rejects unknown keys and
wrongly typed values. Semantic checks (ranges, consistency) live in ``__post_init__`` of each dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import draccus
import yaml

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
DEFAULT_BASE_CONFIG = EXPERIMENTS_DIR / "base.yaml"


class ConfigError(ValueError):
    """Raised for missing files, malformed YAML or invalid values."""


# --------------------------------------------------------------------------------------------- network
@dataclass
class StallConfig:
    """Periodic link outage: every ``interval_s`` the link freezes for ``duration_ms`` in both directions."""

    interval_s: float = 10.0
    duration_ms: float = 300.0

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ConfigError(f"stall.interval_s must be > 0, got {self.interval_s}")
        if self.duration_ms <= 0:
            raise ConfigError(f"stall.duration_ms must be > 0, got {self.duration_ms}")
        if self.duration_ms / 1000.0 >= self.interval_s:
            raise ConfigError("stall.duration_ms must be shorter than stall.interval_s")


@dataclass
class NetworkConfig:
    """Network impairment between Agent and Brain.

    ``rtt_ms`` / ``rtt_jitter_ms`` describe the *round trip* (what ``ping`` shows). The proxy splits both
    evenly over the two directions, so with uniform jitter the RTT stays within ``rtt_ms ± rtt_jitter_ms``.
    """

    rtt_ms: float = 0.0
    rtt_jitter_ms: float = 0.0
    jitter_distribution: Literal["uniform", "normal"] = "uniform"
    # Per-direction serialization limit in Mbit/s; ``None`` disables the limit.
    bandwidth_mbps: float | None = None
    stall: StallConfig | None = None
    # Reads that arrive closer together than this belong to one burst and share one delay sample, so a
    # message split over several TCP reads is not delayed by the maximum of several samples.
    burst_gap_ms: float = 2.0

    def __post_init__(self) -> None:
        if self.rtt_ms < 0:
            raise ConfigError(f"rtt_ms must be >= 0, got {self.rtt_ms}")
        if self.rtt_jitter_ms < 0:
            raise ConfigError(f"rtt_jitter_ms must be >= 0, got {self.rtt_jitter_ms}")
        if self.bandwidth_mbps is not None and self.bandwidth_mbps <= 0:
            raise ConfigError(f"bandwidth_mbps must be > 0 or null, got {self.bandwidth_mbps}")
        if self.burst_gap_ms < 0:
            raise ConfigError(f"burst_gap_ms must be >= 0, got {self.burst_gap_ms}")

    @property
    def one_way_delay_ms(self) -> float:
        return self.rtt_ms / 2.0

    @property
    def one_way_jitter_ms(self) -> float:
        return self.rtt_jitter_ms / 2.0

    def to_dict(self) -> dict[str, Any]:
        encoded = draccus.encode(self)
        if not isinstance(encoded, dict):
            raise ConfigError("NetworkConfig did not encode to a mapping")
        return encoded


@dataclass
class ScenarioConfig:
    """One measurement condition: a network impairment plus the agent's execution mode."""

    name: str
    description: str = ""
    # realtime: the simulation runs at the env's control rate and never waits for the Brain.
    # sync:     the simulation waits for every chunk (LeRobot-eval semantics); reference only.
    mode: Literal["realtime", "sync"] = "realtime"
    network: NetworkConfig = field(default_factory=NetworkConfig)


# --------------------------------------------------------------------------------------------- brain
@dataclass
class CheckpointConfig:
    repo_id: str = "lerobot/diffusion_pusht"
    # Pinned Hub revision; the checkpoint is stored in the pre-0.4 LeRobot format and migrated locally.
    revision: str = "84a7c23178445c6bbf7e1a884ff497017910f653"
    # Where the migrated checkpoint is written (relative paths are relative to the working directory).
    local_dir: str = "checkpoints/diffusion_pusht_migrated"


@dataclass
class BrainProfile:
    """Sampler settings of the diffusion policy. Profiles are selected per experiment suite."""

    noise_scheduler_type: Literal["DDPM", "DDIM"] = "DDIM"
    num_inference_steps: int = 10
    # Actions returned per request. ``None`` = maximum the checkpoint supports (horizon - n_obs_steps + 1).
    n_action_steps: int | None = None

    def __post_init__(self) -> None:
        if self.num_inference_steps < 1:
            raise ConfigError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}")
        if self.n_action_steps is not None and self.n_action_steps < 1:
            raise ConfigError(f"n_action_steps must be >= 1, got {self.n_action_steps}")

    @property
    def sampler_label(self) -> str:
        return f"{self.noise_scheduler_type}-{self.num_inference_steps}"


@dataclass
class BrainConfig:
    host: str = "127.0.0.1"
    port: int = 50051
    # "auto" = mps -> cuda -> cpu. Explicit values: "cpu", "mps", "cuda", "cuda:0", ...
    device: str = "auto"
    # Intra-op threads for CPU inference; 0 = (number of CPUs - 1), at least 1.
    torch_threads: int = 0
    # Linux only: CPU cores the Brain process is pinned to. Empty = no pinning.
    cpu_affinity: list[int] = field(default_factory=list)
    warmup_inferences: int = 3
    grpc_max_workers: int = 8
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        _check_port("brain.port", self.port)
        if self.torch_threads < 0:
            raise ConfigError(f"brain.torch_threads must be >= 0, got {self.torch_threads}")
        if self.warmup_inferences < 0:
            raise ConfigError(f"brain.warmup_inferences must be >= 0, got {self.warmup_inferences}")
        if self.grpc_max_workers < 1:
            raise ConfigError(f"brain.grpc_max_workers must be >= 1, got {self.grpc_max_workers}")

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def resolved_torch_threads(self) -> int:
        if self.torch_threads > 0:
            return self.torch_threads
        return max(1, (os.cpu_count() or 2) - 1)


# --------------------------------------------------------------------------------------------- proxy
@dataclass
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 50052
    control_port: int = 50053
    read_chunk_bytes: int = 65536
    # Upper bound for data buffered inside the proxy per direction (back-pressure beyond this).
    max_buffer_bytes: int = 16 * 1024 * 1024
    # Linux only: CPU cores the proxy process is pinned to. Empty = no pinning.
    cpu_affinity: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check_port("proxy.port", self.port)
        _check_port("proxy.control_port", self.control_port)
        if self.port == self.control_port:
            raise ConfigError("proxy.port and proxy.control_port must differ")
        if self.read_chunk_bytes < 1024:
            raise ConfigError(f"proxy.read_chunk_bytes must be >= 1024, got {self.read_chunk_bytes}")
        if self.max_buffer_bytes < self.read_chunk_bytes:
            raise ConfigError("proxy.max_buffer_bytes must be >= proxy.read_chunk_bytes")

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


# --------------------------------------------------------------------------------------------- agent
@dataclass
class MetaWorldEnvConfig:
    """One Meta-World (v3) task, built like LeRobot's ``MetaworldEnv`` (phase 3a)."""

    task: str = "push-v3"
    camera_name: str = "corner2"
    # Language instruction sent with every request; None = LeRobot's task description (its evaluation).
    instruction: str | None = None

    def __post_init__(self) -> None:
        if not self.task.endswith("-v3"):
            raise ConfigError(f"metaworld.task must be a v3 task name like 'push-v3', got {self.task!r}")


@dataclass
class EnvConfig:
    """Simulation environment. Defaults mirror LeRobot's ``PushtEnv`` config (checked by a unit test).

    ``kind: metaworld`` (phase 3a) uses the ``metaworld`` section instead of the gym fields.
    """

    kind: Literal["gym", "metaworld"] = "gym"
    gym_id: str = "gym_pusht/PushT-v0"
    # Module that registers ``gym_id`` with gymnasium on import.
    package: str = "gym_pusht"
    fps: float = 10.0
    max_episode_steps: int = 300
    gym_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"obs_type": "pixels_agent_pos", "render_mode": "rgb_array"}
    )
    # Raw env observation key -> LeRobot feature key.
    features_map: dict[str, str] = field(
        default_factory=lambda: {"pixels": "observation.image", "agent_pos": "observation.state"}
    )
    # LeRobot key whose value is a valid "stay where you are" action (PushT: the agent position, because
    # actions are absolute position targets). Used before the first chunk in cold-start mode. None = the
    # zero action (Meta-World: actions are end-effector displacements).
    hold_action_key: str | None = "observation.state"
    metaworld: MetaWorldEnvConfig | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ConfigError(f"env.fps must be > 0, got {self.fps}")
        if self.max_episode_steps < 1:
            raise ConfigError(f"env.max_episode_steps must be >= 1, got {self.max_episode_steps}")
        if self.hold_action_key is not None and self.hold_action_key not in self.features_map.values():
            raise ConfigError(f"env.hold_action_key {self.hold_action_key!r} is not a mapped feature key")
        if self.kind == "metaworld" and self.metaworld is None:
            raise ConfigError("env.kind 'metaworld' needs an env.metaworld section")


@dataclass
class ImageTransportConfig:
    """How camera images travel to the Brain (phase 3a). The default sends raw frames at camera resolution.

    ``height`` / ``width`` scale the frames down before sending (the Brain scales them back to the policy's
    input size, so the policy sees a blurrier image); ``codec`` compresses them (``jpeg`` with ``quality``,
    lossless ``png``) or sends raw bytes.
    """

    codec: Literal["raw", "jpeg", "png"] = "raw"
    quality: int = 90
    height: int | None = None
    width: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.quality <= 100:
            raise ConfigError(f"image_transport.quality must be in [1, 100], got {self.quality}")
        for name, value in (("height", self.height), ("width", self.width)):
            if value is not None and value < 8:
                raise ConfigError(f"image_transport.{name} must be >= 8 or null, got {value}")

    @property
    def is_identity(self) -> bool:
        """Raw frames at camera resolution: the phase-1 wire format."""
        return self.codec == "raw" and self.height is None and self.width is None


@dataclass
class AgentConfig:
    client_id: str = "kortyn-agent-0"
    # Actions requested per chunk (<= what the Brain can return).
    actions_per_chunk: int = 15
    # Request a new chunk when at most this many actions remain after executing the current tick's action.
    refill_threshold: int = 6
    # gRPC deadline per request. Only a failure detector: a late chunk is still merged (stale part dropped).
    request_timeout_s: float = 5.0
    # Episode start: "warm" = first chunk is fetched before the episode clock starts (robot waits at the
    # start line, excluded from statistics); "cold" = the clock starts at reset and the robot holds its
    # position until the first chunk arrives.
    start_mode: Literal["warm", "cold"] = "warm"
    warmup_timeout_s: float = 60.0
    # Actions executed per chunk in sync (reference) mode; LeRobot eval uses the policy's n_action_steps.
    sync_actions_per_chunk: int = 8
    # Linux only: CPU cores the agent process is pinned to. Empty = no pinning.
    cpu_affinity: list[int] = field(default_factory=list)
    # Validity gate per episode (measurement hygiene, not robot performance).
    max_tick_lateness_ms: float = 50.0
    max_proxy_lateness_ms: float = 20.0
    max_cpu_steal_fraction: float = 0.2
    # Phase 3a: camera images on the wire (resolution, compression); the default is the phase-1 format.
    image_transport: ImageTransportConfig = field(default_factory=ImageTransportConfig)

    def __post_init__(self) -> None:
        if self.actions_per_chunk < 1:
            raise ConfigError(f"agent.actions_per_chunk must be >= 1, got {self.actions_per_chunk}")
        if not 0 <= self.refill_threshold < self.actions_per_chunk:
            raise ConfigError("agent.refill_threshold must be in [0, actions_per_chunk)")
        if self.request_timeout_s <= 0 or self.warmup_timeout_s <= 0:
            raise ConfigError("agent timeouts must be > 0")
        if not 1 <= self.sync_actions_per_chunk <= self.actions_per_chunk:
            raise ConfigError("agent.sync_actions_per_chunk must be in [1, actions_per_chunk]")


# --------------------------------------------------------------------------------------------- freshness
@dataclass
class LatencyEstimatorConfig:
    """Online estimate of the end-to-end latency of one robot: median and MAD over a sliding window."""

    # Completed requests the estimate is based on (median/MAD: robust against isolated stalls).
    window: int = 16
    # Prior before the first measurement; the warm-up request of every episode replaces it.
    initial_ms: float = 400.0
    # Spread assumed while fewer than three samples exist, and a floor for the robust spread afterwards.
    initial_spread_ms: float = 50.0
    min_spread_ms: float = 5.0

    def __post_init__(self) -> None:
        if self.window < 3:
            raise ConfigError(f"latency.window must be >= 3, got {self.window}")
        if self.initial_ms <= 0:
            raise ConfigError(f"latency.initial_ms must be > 0, got {self.initial_ms}")
        if self.initial_spread_ms < 0 or self.min_spread_ms < 0:
            raise ConfigError("latency spreads must be >= 0")


@dataclass
class RequestConfig:
    """When the agent asks the Brain for a new plan."""

    # threshold: request when at most ``refill_threshold`` actions remain after the current tick (phase 1).
    # adaptive:  request so early that the next plan arrives in time (latency quantile + margin) AND so
    #            often that the expected mean plan age stays at or below ``target_age_s``.
    mode: Literal["threshold", "adaptive"] = "threshold"
    # threshold mode; None = ``agent.refill_threshold`` (6 in phase 1).
    refill_threshold: int | None = None
    target_age_s: float = 0.7
    # Latency used for the deadline: median + z * robust spread (1.645 ~ 95 % for normal noise).
    latency_quantile_z: float = 1.645
    # Extra ticks of lead time on top of the latency quantile.
    safety_margin_ticks: int = 1

    def __post_init__(self) -> None:
        if self.refill_threshold is not None and self.refill_threshold < 0:
            raise ConfigError(f"request.refill_threshold must be >= 0, got {self.refill_threshold}")
        if self.target_age_s <= 0:
            raise ConfigError(f"request.target_age_s must be > 0, got {self.target_age_s}")
        if self.latency_quantile_z < 0:
            raise ConfigError(f"request.latency_quantile_z must be >= 0, got {self.latency_quantile_z}")
        if self.safety_margin_ticks < 0:
            raise ConfigError(f"request.safety_margin_ticks must be >= 0, got {self.safety_margin_ticks}")


@dataclass
class SmoothingConfig:
    """Transition between consecutive plans."""

    # none:    the new plan simply replaces the old one where they overlap (phase 1).
    # inpaint: the Brain inpaints the new plan: actions that are executed while it computes are frozen to
    #          the old plan, the rest of the overlap is pulled towards the old plan with decaying weights
    #          (gradient-free variant of real-time chunking, see kortyn/inpainting.py).
    mode: Literal["none", "inpaint"] = "none"
    # Weights over the overlap after the frozen part, as in LeRobot's RTC: linear | exp | zeros (= frozen
    # part only, "hard" inpainting).
    schedule: Literal["linear", "exp", "zeros"] = "linear"
    # Longest soft overlap (actions of the old plan after the frozen part that guide the new plan).
    max_overlap_steps: int = 10

    def __post_init__(self) -> None:
        if self.max_overlap_steps < 0:
            raise ConfigError(f"smoothing.max_overlap_steps must be >= 0, got {self.max_overlap_steps}")


@dataclass
class GuardConfig:
    """What the robot does when no usable action is left (budget violation = emergency)."""

    # repeat_last: repeat the last executed action (phase 1); hold: stop (PushT: target = current position).
    mode: Literal["repeat_last", "hold"] = "repeat_last"
    # Actions older than this (observation -> execution) count as unusable: emergency, the robot holds.
    max_plan_age_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_plan_age_s is not None:
            if self.max_plan_age_s <= 0:
                raise ConfigError(f"guard.max_plan_age_s must be > 0, got {self.max_plan_age_s}")
            if self.mode != "hold":
                raise ConfigError(
                    "guard.max_plan_age_s needs guard.mode 'hold' (repeating the last action would execute "
                    "an even older plan)"
                )


@dataclass
class StrategyConfig:
    """One freshness strategy. ``StrategyConfig()`` is exactly the phase-1 behaviour (the baseline)."""

    description: str = ""
    request: RequestConfig = field(default_factory=RequestConfig)
    # timestep: actions are keyed by control step; steps that already passed are skipped on arrival.
    # shift:    naive client: the plan starts executing at its arrival tick (stale actions are executed).
    alignment: Literal["timestep", "shift"] = "timestep"
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    latency: LatencyEstimatorConfig = field(default_factory=LatencyEstimatorConfig)

    def to_dict(self) -> dict[str, Any]:
        encoded = draccus.encode(self)
        if not isinstance(encoded, dict):
            raise ConfigError("StrategyConfig did not encode to a mapping")
        return encoded


BASELINE_STRATEGY = "baseline"


# --------------------------------------------------------------------------------------------- virtual time
EmpiricalQuantity = Literal[
    "server_total_ms",
    "server_inference_ms",
    "send_offset_ms",
    "network_ms",
    "e2e_ms",
    "tick_lateness_ms",
]


@dataclass
class EmpiricalSourceConfig:
    """Samples of one logged quantity, read from earlier runs (``results/<profile>/<run>/``).

    ``paths`` are run directories or glob patterns (``results/kalibriert/*``); runs in sync mode are skipped,
    and only the final attempt of every seed counts (as in the reports). Request quantities use the
    real-time requests (not the warm-up); ``tick_lateness_ms`` uses every control tick.

    ``per_seed``: an episode draws only from the values logged in the episode with the same seed (pooled
    over the listed runs). That calibrates the sample to the state of the logging machine during that
    episode (session drift, drift within a session); a seed without logged values is an error.
    """

    paths: list[str] = field(default_factory=list)
    quantity: EmpiricalQuantity = "server_total_ms"
    per_seed: bool = False

    def __post_init__(self) -> None:
        if not self.paths:
            raise ConfigError("empirical.paths must name at least one run directory or pattern")


@dataclass
class DistributionConfig:
    """A random duration in milliseconds (latency component of the virtual-time mode).

    ``constant``: ``value_ms``. ``uniform``: ``[low_ms, high_ms]``. ``normal`` / ``lognormal``: mean and
    standard deviation **of the duration itself** (``mean_ms``, ``std_ms``). ``empirical``: resampled from
    logged runs. Every sample is multiplied by ``scale`` (sensitivity analysis) and clipped to
    ``[min_ms, max_ms]``.
    """

    kind: Literal["constant", "uniform", "normal", "lognormal", "empirical"] = "constant"
    value_ms: float = 0.0
    low_ms: float = 0.0
    high_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    empirical: EmpiricalSourceConfig | None = None
    scale: float = 1.0
    min_ms: float = 0.0
    max_ms: float | None = None

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ConfigError(f"distribution scale must be > 0, got {self.scale}")
        if self.min_ms < 0:
            raise ConfigError(f"distribution min_ms must be >= 0, got {self.min_ms}")
        if self.max_ms is not None and self.max_ms < self.min_ms:
            raise ConfigError("distribution max_ms must be >= min_ms")
        if self.kind == "constant" and self.value_ms < 0:
            raise ConfigError(f"constant value_ms must be >= 0, got {self.value_ms}")
        if self.kind == "uniform" and not 0 <= self.low_ms <= self.high_ms:
            raise ConfigError("uniform needs 0 <= low_ms <= high_ms")
        if self.kind in ("normal", "lognormal") and (self.mean_ms <= 0 or self.std_ms < 0):
            raise ConfigError(f"{self.kind} needs mean_ms > 0 and std_ms >= 0")
        if self.kind == "empirical" and self.empirical is None:
            raise ConfigError("kind 'empirical' needs an 'empirical' source")
        if self.kind != "empirical" and self.empirical is not None:
            raise ConfigError(f"'empirical' source given for kind {self.kind!r}")


@dataclass
class HardwareProfileConfig:
    """Service time of the Brain on one class of hardware, per Brain profile (model + sampler settings).

    The service time is what one inference occupies the Brain: model plus pre- and post-processing.
    """

    description: str = ""
    service: dict[str, DistributionConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.service:
            raise ConfigError("hardware profile needs at least one 'service' entry (per Brain profile)")


@dataclass
class ReplayConfig:
    """Where the logged runs live that the replay mode plays back: ``<results_dir>/<profile>/<run>/``."""

    results_dir: str = "results"


@dataclass
class TimingConfig:
    """How latency arises in the simulation-time (virtual time) mode.

    ``model``: a request sent at virtual time t delivers its plan at t + L with
    L = client overhead + uplink (scenario: delay, jitter, stalls, bandwidth) + transport overhead
    + Brain queue (FIFO) + service time (hardware profile) + downlink. ``replay``: L is the end-to-end
    latency logged for the same request of the same episode in an earlier real-time run, and the control
    ticks start exactly as late as they did there.
    """

    description: str = ""
    latency: Literal["model", "replay"] = "model"
    # model: hardware profile that provides the service time for the suite's Brain profile
    hardware: str | None = None
    # observation released (tick start) -> request leaves the agent (building and encoding the request)
    client_overhead: DistributionConfig = field(default_factory=DistributionConfig)
    # per round trip: serialization, kernel and gRPC overhead outside the impaired link
    transport_overhead: DistributionConfig = field(default_factory=DistributionConfig)
    # per message the link actually holds back: how late the forwarder delivers (timer granularity)
    delivery_lateness: DistributionConfig = field(default_factory=DistributionConfig)
    # start of a control tick after its deadline (constant 0 = ideal clock)
    tick_lateness: DistributionConfig = field(default_factory=DistributionConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    # Independent latency draws for the same episode seeds (replications of a whole experiment).
    timing_seed: int = 0
    # Brain model: parallel inference servers and the order in which waiting requests are served.
    brain_servers: int = 1
    scheduler: Literal["fifo"] = "fifo"

    def __post_init__(self) -> None:
        if self.latency == "model" and not self.hardware:
            raise ConfigError("timing with latency 'model' needs a hardware profile")
        if self.brain_servers < 1:
            raise ConfigError(f"brain_servers must be >= 1, got {self.brain_servers}")


# --------------------------------------------------------------------------------------------- suite
@dataclass
class RunSpec:
    """One measured condition inside a suite.

    ``strategy`` (phase 2) names an entry of ``strategies``; ``None`` keeps the phase-1 behaviour and the
    phase-1 metadata format. ``timing`` (phase 3a) names an entry of ``timings`` and runs the condition in
    virtual time (``python -m kortyn.run_virtual``); ``replay_run`` is the logged run a replay timing plays
    back (default: the scenario name). ``label`` defaults to the scenario name, or
    ``[<timing>__][<strategy>__]<scenario or replay_run>``.
    """

    scenario: str
    label: str | None = None
    strategy: str | None = None
    timing: str | None = None
    replay_run: str | None = None
    # Per-run agent overrides (phase 3a, virtual time; give the run an explicit label):
    sync_actions_per_chunk: int | None = None  # executed actions per plan in sync mode
    image_transport: ImageTransportConfig | None = None  # camera images on the wire

    def __post_init__(self) -> None:
        if self.sync_actions_per_chunk is not None and self.sync_actions_per_chunk < 1:
            raise ConfigError(f"run sync_actions_per_chunk must be >= 1, got {self.sync_actions_per_chunk}")

    @property
    def run_label(self) -> str:
        if self.label:
            return self.label
        base = self.replay_run or self.scenario
        if self.strategy is not None:
            base = f"{self.strategy}__{base}"
        return base if self.timing is None else f"{self.timing}__{base}"


@dataclass
class SuiteConfig:
    """All runs measured with one Brain profile. Runs are interleaved seed by seed."""

    profile: str
    runs: list[RunSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        labels = [run.run_label for run in self.runs]
        if len(labels) != len(set(labels)):
            raise ConfigError(f"suite {self.profile!r} has duplicate run labels: {labels}")


@dataclass
class ExperimentConfig:
    results_dir: str = "results"
    reports_dir: str = "reports"
    seed_start: int = 1000
    episodes: int = 50
    max_attempts_per_episode: int = 3
    brain_startup_timeout_s: float = 1800.0
    proxy_startup_timeout_s: float = 30.0
    # Linux: pin the Brain to cores 0..n-2 and agent + proxy to core n-1 (ignored where unsupported).
    pin_cpus: bool = True
    # Virtual-time runner (phase 3a): worker processes (0 = one per CPU core / torch_threads) and torch
    # intra-op threads per worker. Results do not depend on the worker count; they do depend (in the last
    # float bits of the model output) on the thread count, so keep it fixed within one comparison.
    workers: int = 0
    torch_threads: int = 1
    suites: list[SuiteConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ConfigError(f"experiment.episodes must be >= 1, got {self.episodes}")
        if self.max_attempts_per_episode < 1:
            raise ConfigError("experiment.max_attempts_per_episode must be >= 1")
        if self.workers < 0:
            raise ConfigError(f"experiment.workers must be >= 0, got {self.workers}")
        if self.torch_threads < 1:
            raise ConfigError(f"experiment.torch_threads must be >= 1, got {self.torch_threads}")


@dataclass
class KortynConfig:
    """Root of ``base.yaml``."""

    brain: BrainConfig = field(default_factory=BrainConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    profiles: dict[str, BrainProfile] = field(default_factory=dict)
    # Phase 2: named freshness strategies. "baseline" (phase 1) exists implicitly unless defined here.
    strategies: dict[str, StrategyConfig] = field(default_factory=dict)
    # Phase 3a (virtual time): service-time distributions per hardware class, and named timing models.
    hardware_profiles: dict[str, HardwareProfileConfig] = field(default_factory=dict)
    timings: dict[str, TimingConfig] = field(default_factory=dict)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self) -> None:
        for name, timing in self.timings.items():
            if timing.hardware is not None and timing.hardware not in self.hardware_profiles:
                raise ConfigError(
                    f"timing {name!r} references unknown hardware profile {timing.hardware!r}; "
                    f"known: {sorted(self.hardware_profiles)}"
                )
        for suite in self.experiment.suites:
            if suite.profile not in self.profiles:
                raise ConfigError(
                    f"suite references unknown profile {suite.profile!r}; known: {sorted(self.profiles)}"
                )
            for run in suite.runs:
                if run.strategy is not None:
                    self.strategy(run.strategy)
                if run.timing is not None:
                    timing = self.timing(run.timing)
                    hardware = timing.hardware
                    if hardware is not None and suite.profile not in self.hardware_profiles[hardware].service:
                        raise ConfigError(
                            f"hardware profile {hardware!r} has no service time for Brain profile "
                            f"{suite.profile!r}"
                        )
                elif run.replay_run is not None:
                    raise ConfigError(f"run {run.run_label!r}: replay_run needs a replay timing")
                if run.sync_actions_per_chunk is not None and not (
                    1 <= run.sync_actions_per_chunk <= self.agent.actions_per_chunk
                ):
                    raise ConfigError(
                        f"run {run.run_label!r}: sync_actions_per_chunk must be in "
                        "[1, agent.actions_per_chunk]"
                    )
        for name, strategy in self.strategies.items():
            threshold = strategy.request.refill_threshold
            if threshold is not None and threshold >= self.agent.actions_per_chunk:
                raise ConfigError(
                    f"strategy {name!r}: request.refill_threshold must be < agent.actions_per_chunk"
                )
        ports = {self.brain.port, self.proxy.port, self.proxy.control_port}
        if len(ports) != 3:
            raise ConfigError("brain.port, proxy.port and proxy.control_port must all differ")

    def profile(self, name: str) -> BrainProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ConfigError(f"unknown profile {name!r}; known: {sorted(self.profiles)}") from exc

    def strategy(self, name: str | None) -> StrategyConfig:
        """Strategy by name; ``None`` and an undefined ``"baseline"`` give the phase-1 behaviour."""
        if name is None:
            return StrategyConfig()
        if name in self.strategies:
            return self.strategies[name]
        if name == BASELINE_STRATEGY:
            return StrategyConfig(description="Phase 1 (Baseline)")
        known = sorted({BASELINE_STRATEGY, *self.strategies})
        raise ConfigError(f"unknown strategy {name!r}; known: {known}")

    def timing(self, name: str) -> TimingConfig:
        try:
            return self.timings[name]
        except KeyError as exc:
            raise ConfigError(f"unknown timing {name!r}; known: {sorted(self.timings)}") from exc


# --------------------------------------------------------------------------------------------- loading
def _check_port(name: str, port: int) -> None:
    if not 1 <= port <= 65535:
        raise ConfigError(f"{name} must be in [1, 65535], got {port}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def _root_config_error(exc: BaseException) -> ConfigError | None:
    """Find a ConfigError raised by a ``__post_init__`` that draccus wrapped into its own error."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ConfigError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def decode[T](cls: type[T], data: dict[str, Any], source: str) -> T:
    """Strictly decode ``data`` into dataclass ``cls`` (unknown keys and wrong types are errors)."""
    try:
        return draccus.decode(cls, data)  # type: ignore[no-any-return]
    except Exception as exc:  # draccus raises its own DecodingError hierarchy
        root = _root_config_error(exc)
        raise ConfigError(f"{source}: {root if root is not None else exc}") from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Mappings are merged key by key; everything else (lists, scalars) in ``override`` replaces ``base``."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config_tree(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read ``path`` and, if it says ``extends: <file>``, merge it over that file (relative to ``path``)."""
    resolved = path.resolve()
    if resolved in chain:
        cycle = " -> ".join(str(p) for p in (*chain, resolved))
        raise ConfigError(f"circular 'extends': {cycle}")
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str) or not parent:
        raise ConfigError(f"{path}: 'extends' must be a file name, got {parent!r}")
    return _deep_merge(_read_config_tree(path.parent / parent, (*chain, resolved)), data)


def load_config(path: Path | str = DEFAULT_BASE_CONFIG) -> KortynConfig:
    path = Path(path)
    return decode(KortynConfig, _read_config_tree(path), str(path))


def resolve_scenario_path(name_or_path: str) -> Path:
    """Accept a scenario name (``schlecht``) or a path to a YAML file."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"}:
        return candidate
    return EXPERIMENTS_DIR / f"{name_or_path}.yaml"


def load_scenario(name_or_path: str) -> ScenarioConfig:
    path = resolve_scenario_path(name_or_path)
    scenario = decode(ScenarioConfig, _read_yaml(path), str(path))
    if scenario.name != path.stem:
        raise ConfigError(f"{path}: name {scenario.name!r} must match the file name {path.stem!r}")
    return scenario


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.yaml") if p.stem != "base")
