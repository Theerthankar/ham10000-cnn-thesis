"""Label order and the experiment registry.

Twelve runs in three groups:

  E1-E6    the original 2x3 factorial (MobileNetV2, ResNet-50) at 224px.
           These are the pre-registered, confirmatory experiments.
  E7-E9    EfficientNet-B3 at 224px. Same input size as everything above, so
           any difference here is attributable to the architecture alone.
  E10-E12  EfficientNet-B3 at 300px, its native pretraining resolution.
           Paired against E7-E9 to separate "B3 is better/worse" from
           "B3 was handed a bigger input".

E7-E12 are an ablation added after the original six were registered. They are
kept in their own statistical family for exactly that reason (see
scripts/statistical_tests.py).
"""

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
DX_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224

EXPERIMENTS = {
    # --- confirmatory 2x3 factorial, 224px ---
    "E1":  {"architecture": "mobilenet_v2",    "strategy": "augmentation", "image_size": 224},
    "E2":  {"architecture": "resnet50",        "strategy": "augmentation", "image_size": 224},
    "E3":  {"architecture": "mobilenet_v2",    "strategy": "weighted_ce",  "image_size": 224},
    "E4":  {"architecture": "resnet50",        "strategy": "weighted_ce",  "image_size": 224},
    "E5":  {"architecture": "mobilenet_v2",    "strategy": "oversampling", "image_size": 224},
    "E6":  {"architecture": "resnet50",        "strategy": "oversampling", "image_size": 224},
    # --- EfficientNet-B3 ablation, matched 224px ---
    "E7":  {"architecture": "efficientnet_b3", "strategy": "augmentation", "image_size": 224},
    "E8":  {"architecture": "efficientnet_b3", "strategy": "weighted_ce",  "image_size": 224},
    "E9":  {"architecture": "efficientnet_b3", "strategy": "oversampling", "image_size": 224},
    # --- EfficientNet-B3 ablation, native 300px ---
    "E10": {"architecture": "efficientnet_b3", "strategy": "augmentation", "image_size": 300},
    "E11": {"architecture": "efficientnet_b3", "strategy": "weighted_ce",  "image_size": 300},
    "E12": {"architecture": "efficientnet_b3", "strategy": "oversampling", "image_size": 300},
}

CONFIRMATORY_EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5", "E6"]
ABLATION_EXPERIMENTS = ["E7", "E8", "E9", "E10", "E11", "E12"]
ALL_EXPERIMENTS = CONFIRMATORY_EXPERIMENTS + ABLATION_EXPERIMENTS

ARCHITECTURE_LABELS = {
    "mobilenet_v2": "MobileNetV2",
    "resnet50": "ResNet-50",
    "efficientnet_b3": "EfficientNet-B3",
}

STRATEGY_LABELS = {
    "augmentation": "Augmentation only",
    "weighted_ce": "Weighted CE",
    "oversampling": "Random oversampling",
}


def experiment_label(exp_id: str) -> str:
    """Short human-readable name, e.g. 'E10 EfficientNet-B3 @300 / Weighted CE'."""
    exp = EXPERIMENTS[exp_id]
    arch = ARCHITECTURE_LABELS[exp["architecture"]]
    strategy = STRATEGY_LABELS[exp["strategy"]]
    return f"{exp_id} {arch} @{exp['image_size']} / {strategy}"
