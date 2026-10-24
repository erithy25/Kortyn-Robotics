from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Tensor(_message.Message):
    __slots__ = ("dtype", "shape", "data", "encoding", "encoded_shape")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    ENCODED_SHAPE_FIELD_NUMBER: _ClassVar[int]
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    encoding: str
    encoded_shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ..., encoding: _Optional[str] = ..., encoded_shape: _Optional[_Iterable[int]] = ...) -> None: ...

class GetInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FeatureSpec(_message.Message):
    __slots__ = ("key", "kind", "shape")
    KEY_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    key: str
    kind: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, key: _Optional[str] = ..., kind: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class BrainInfo(_message.Message):
    __slots__ = ("policy_type", "checkpoint", "device", "n_obs_steps", "max_actions", "action_dim", "input_features", "sampler", "lerobot_version", "torch_version", "warmup_inference_ms", "torch_threads", "supports_guidance")
    POLICY_TYPE_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    N_OBS_STEPS_FIELD_NUMBER: _ClassVar[int]
    MAX_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    ACTION_DIM_FIELD_NUMBER: _ClassVar[int]
    INPUT_FEATURES_FIELD_NUMBER: _ClassVar[int]
    SAMPLER_FIELD_NUMBER: _ClassVar[int]
    LEROBOT_VERSION_FIELD_NUMBER: _ClassVar[int]
    TORCH_VERSION_FIELD_NUMBER: _ClassVar[int]
    WARMUP_INFERENCE_MS_FIELD_NUMBER: _ClassVar[int]
    TORCH_THREADS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_GUIDANCE_FIELD_NUMBER: _ClassVar[int]
    policy_type: str
    checkpoint: str
    device: str
    n_obs_steps: int
    max_actions: int
    action_dim: int
    input_features: _containers.RepeatedCompositeFieldContainer[FeatureSpec]
    sampler: str
    lerobot_version: str
    torch_version: str
    warmup_inference_ms: float
    torch_threads: int
    supports_guidance: bool
    def __init__(self, policy_type: _Optional[str] = ..., checkpoint: _Optional[str] = ..., device: _Optional[str] = ..., n_obs_steps: _Optional[int] = ..., max_actions: _Optional[int] = ..., action_dim: _Optional[int] = ..., input_features: _Optional[_Iterable[_Union[FeatureSpec, _Mapping]]] = ..., sampler: _Optional[str] = ..., lerobot_version: _Optional[str] = ..., torch_version: _Optional[str] = ..., warmup_inference_ms: _Optional[float] = ..., torch_threads: _Optional[int] = ..., supports_guidance: _Optional[bool] = ...) -> None: ...

class InferRequest(_message.Message):
    __slots__ = ("client_id", "episode_id", "request_id", "obs_timestep", "seed", "max_actions", "observation", "guide_start_timestep", "guide_actions", "guide_weights", "task")
    class ObservationEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tensor
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OBS_TIMESTEP_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    MAX_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    GUIDE_START_TIMESTEP_FIELD_NUMBER: _ClassVar[int]
    GUIDE_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    GUIDE_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    episode_id: int
    request_id: int
    obs_timestep: int
    seed: int
    max_actions: int
    observation: _containers.MessageMap[str, Tensor]
    guide_start_timestep: int
    guide_actions: Tensor
    guide_weights: _containers.RepeatedScalarFieldContainer[float]
    task: str
    def __init__(self, client_id: _Optional[str] = ..., episode_id: _Optional[int] = ..., request_id: _Optional[int] = ..., obs_timestep: _Optional[int] = ..., seed: _Optional[int] = ..., max_actions: _Optional[int] = ..., observation: _Optional[_Mapping[str, Tensor]] = ..., guide_start_timestep: _Optional[int] = ..., guide_actions: _Optional[_Union[Tensor, _Mapping]] = ..., guide_weights: _Optional[_Iterable[float]] = ..., task: _Optional[str] = ...) -> None: ...

class InferResponse(_message.Message):
    __slots__ = ("request_id", "start_timestep", "actions", "server_queue_ms", "server_inference_ms", "server_total_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    START_TIMESTEP_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    SERVER_QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    SERVER_INFERENCE_MS_FIELD_NUMBER: _ClassVar[int]
    SERVER_TOTAL_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    start_timestep: int
    actions: Tensor
    server_queue_ms: float
    server_inference_ms: float
    server_total_ms: float
    def __init__(self, request_id: _Optional[int] = ..., start_timestep: _Optional[int] = ..., actions: _Optional[_Union[Tensor, _Mapping]] = ..., server_queue_ms: _Optional[float] = ..., server_inference_ms: _Optional[float] = ..., server_total_ms: _Optional[float] = ...) -> None: ...

class GetStatsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BrainStats(_message.Message):
    __slots__ = ("requests", "inferences", "abandoned", "busy_ms", "queue_wait_ms", "uptime_ms")
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    INFERENCES_FIELD_NUMBER: _ClassVar[int]
    ABANDONED_FIELD_NUMBER: _ClassVar[int]
    BUSY_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_WAIT_MS_FIELD_NUMBER: _ClassVar[int]
    UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
    requests: int
    inferences: int
    abandoned: int
    busy_ms: float
    queue_wait_ms: float
    uptime_ms: float
    def __init__(self, requests: _Optional[int] = ..., inferences: _Optional[int] = ..., abandoned: _Optional[int] = ..., busy_ms: _Optional[float] = ..., queue_wait_ms: _Optional[float] = ..., uptime_ms: _Optional[float] = ...) -> None: ...
