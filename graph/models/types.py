from enum import Enum


class Status(str, Enum):
    keep = "keep"
    discard = "discard"
    crash = "crash"


class Category(str, Enum):
    normalization = "normalization"
    attention = "attention"
    optimizer = "optimizer"
    schedule = "schedule"
    architecture = "architecture"
    hyperparameter = "hyperparameter"
    training_loop = "training_loop"
    regularization = "regularization"


class EdgeType(str, Enum):
    tried = "TRIED"
    improved_from = "IMPROVED_FROM"
    failed_from = "FAILED_FROM"
    motivated_by = "MOTIVATED_BY"
    invalidates = "INVALIDATES"
    correlated_with = "CORRELATED_WITH"
    led_to = "LED_TO"
    ran_by = "RAN_BY"
    ran_on = "RAN_ON"
    beat_global = "BEAT_GLOBAL"
    beat_own = "BEAT_OWN"


class GPUType(str, Enum):
    h200 = "H200"
    a100 = "A100"
    a10g = "A10G"
    l4 = "L4"
    t4 = "T4"
