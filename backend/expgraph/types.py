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
