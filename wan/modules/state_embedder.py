"""StateEmbedder: VLM JSON → learnable state tokens for DiT cross-attention.

Converts structured physical-state JSON (characters + scene) into a fixed-size
sequence of embedding tokens [B, num_tokens, dim] that are concatenated to the
T5 text context before entering the DiT cross-attention layers.

Design:
  - Per-character: continuous (x, y, scale) + discrete (pose, facing, action,
    cloth_upper, cloth_lower, holding) → 240-dim
  - Scene: discrete (location, lighting, time_of_day) → 80-dim
  - Concat [240×3 + 80] = 800 → MLP → [num_tokens × dim]
  - Learnable slot embeddings distinguish character/scene tokens
"""

import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# ============================================================
# Vocabulary definitions (index 0 = "unknown" fallback)
# ============================================================

POSE_VOCAB = [
    "unknown", "standing", "sitting", "walking", "running", "lying",
    "crouching", "kneeling", "jumping", "leaning", "bending",
]  # 11

FACING_VOCAB = [
    "unknown", "left", "right", "forward", "backward",
    "forward_left", "forward_right", "backward_left", "backward_right",
]  # 9

ACTION_VOCAB = [
    "unknown",
    # Basic movement
    "walking", "running", "jumping", "climbing", "crawling",
    "dancing", "swimming", "falling", "sliding", "rolling",
    # Upper body
    "waving", "pointing", "reaching", "grabbing", "throwing",
    "catching", "pushing", "pulling", "lifting", "carrying",
    "holding", "dropping", "clapping", "hugging", "shaking_hands",
    # Interactions
    "eating", "drinking", "reading", "writing", "typing",
    "cooking", "cleaning", "painting", "playing_instrument",
    "taking_photo", "using_phone", "driving", "riding",
    # Combat / sport
    "kicking", "punching", "blocking", "dodging",
    "swinging", "shooting", "aiming",
    # Stationary
    "sitting_down", "standing_up", "turning", "bowing",
    "nodding", "looking_around",
    # Placeholder
    "none", "other",
]  # 50

COLOR_VOCAB = [
    "unknown", "black", "white", "red", "blue", "green", "yellow",
    "orange", "purple", "pink", "brown", "gray", "beige", "navy",
    "none",
]  # 15

HOLDING_VOCAB = [
    "unknown",
    # Common handheld
    "bag", "backpack", "suitcase", "umbrella", "cup", "bottle",
    "phone", "camera", "book", "pen", "key", "wallet",
    # Tools / weapons
    "knife", "sword", "gun", "stick", "hammer", "flashlight",
    "tool", "brush", "racket",
    # Food
    "food", "plate", "bowl",
    # Musical / sports
    "ball", "guitar", "microphone",
    # Misc
    "box", "rope", "flag", "flower", "hat", "glasses",
    "newspaper", "laptop", "tablet",
    # Placeholder
    "none", "other",
]  # 39

LOCATION_VOCAB = [
    "unknown",
    # Indoor
    "living_room", "bedroom", "kitchen", "bathroom", "office",
    "classroom", "hallway", "elevator", "staircase", "restaurant",
    "cafe", "bar", "shop", "hospital", "gym",
    # Outdoor
    "street", "park", "garden", "forest", "beach", "mountain",
    "bridge", "parking_lot", "rooftop", "stadium",
    # Transport
    "car_interior", "bus_interior", "train_interior",
    # Placeholder
    "other",
]  # 30

LIGHTING_VOCAB = [
    "unknown", "warm", "cool", "bright", "dim", "natural",
    "dramatic", "neon", "candlelight", "overcast",
]  # 10

TIME_OF_DAY_VOCAB = [
    "unknown", "day", "night", "dawn", "dusk", "afternoon",
]  # 6

# Lookup dicts for fast encoding
_VOCAB_TABLES = {
    "pose": {v: i for i, v in enumerate(POSE_VOCAB)},
    "facing": {v: i for i, v in enumerate(FACING_VOCAB)},
    "action": {v: i for i, v in enumerate(ACTION_VOCAB)},
    "color": {v: i for i, v in enumerate(COLOR_VOCAB)},
    "holding": {v: i for i, v in enumerate(HOLDING_VOCAB)},
    "location": {v: i for i, v in enumerate(LOCATION_VOCAB)},
    "lighting": {v: i for i, v in enumerate(LIGHTING_VOCAB)},
    "time_of_day": {v: i for i, v in enumerate(TIME_OF_DAY_VOCAB)},
}

MAX_CHARACTERS = 3
NUM_STATE_TOKENS = 8  # output token count


def _lookup(vocab_name: str, value: str) -> int:
    """Look up a value in a vocabulary, returning 0 (unknown) if not found."""
    table = _VOCAB_TABLES[vocab_name]
    if value is None:
        return 0
    return table.get(str(value).lower().strip(), 0)


def json_to_tensors(
    state_json: dict,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Convert a physical-state JSON dict to tensors for StateEmbedder.

    Expected JSON structure:
    {
      "characters": [
        {"x": 0.3, "y": 0.5, "scale": 0.8,
         "pose": "standing", "facing": "left", "action": "walking",
         "cloth_upper": "blue", "cloth_lower": "black", "holding": "bag"},
        ...  (up to 3)
      ],
      "scene": {
        "location": "street", "lighting": "natural", "time_of_day": "day"
      }
    }

    Returns dict with keys:
      - char_continuous: [MAX_CHARACTERS, 3]  (x, y, scale)
      - char_pose:       [MAX_CHARACTERS]     (int indices)
      - char_facing:     [MAX_CHARACTERS]
      - char_action:     [MAX_CHARACTERS]
      - char_cloth_upper:[MAX_CHARACTERS]
      - char_cloth_lower:[MAX_CHARACTERS]
      - char_holding:    [MAX_CHARACTERS]
      - scene_location:  [1]
      - scene_lighting:  [1]
      - scene_time:      [1]
      - char_mask:        [MAX_CHARACTERS]  (1.0 if character present, else 0.0)
    """
    characters = state_json.get("characters", [])[:MAX_CHARACTERS]
    scene = state_json.get("scene", {})

    # Character tensors
    char_cont = torch.zeros(MAX_CHARACTERS, 3, device=device)
    char_pose = torch.zeros(MAX_CHARACTERS, dtype=torch.long, device=device)
    char_facing = torch.zeros(MAX_CHARACTERS, dtype=torch.long, device=device)
    char_action = torch.zeros(MAX_CHARACTERS, dtype=torch.long, device=device)
    char_cloth_upper = torch.zeros(MAX_CHARACTERS, dtype=torch.long, device=device)
    char_cloth_lower = torch.zeros(MAX_CHARACTERS, dtype=torch.long, device=device)
    char_holding = torch.zeros(MAX_CHARACTERS, dtype=torch.long, device=device)
    char_mask = torch.zeros(MAX_CHARACTERS, device=device)

    for i, ch in enumerate(characters):
        char_cont[i, 0] = float(ch.get("x", 0.5))
        char_cont[i, 1] = float(ch.get("y", 0.5))
        char_cont[i, 2] = float(ch.get("scale", 0.5))
        char_pose[i] = _lookup("pose", ch.get("pose"))
        char_facing[i] = _lookup("facing", ch.get("facing"))
        char_action[i] = _lookup("action", ch.get("action"))
        char_cloth_upper[i] = _lookup("color", ch.get("cloth_upper"))
        char_cloth_lower[i] = _lookup("color", ch.get("cloth_lower"))
        char_holding[i] = _lookup("holding", ch.get("holding"))
        char_mask[i] = 1.0

    # Scene tensors
    scene_location = torch.tensor([_lookup("location", scene.get("location"))],
                                  dtype=torch.long, device=device)
    scene_lighting = torch.tensor([_lookup("lighting", scene.get("lighting"))],
                                  dtype=torch.long, device=device)
    scene_time = torch.tensor([_lookup("time_of_day", scene.get("time_of_day"))],
                              dtype=torch.long, device=device)

    return {
        "char_continuous": char_cont,
        "char_pose": char_pose,
        "char_facing": char_facing,
        "char_action": char_action,
        "char_cloth_upper": char_cloth_upper,
        "char_cloth_lower": char_cloth_lower,
        "char_holding": char_holding,
        "char_mask": char_mask,
        "scene_location": scene_location,
        "scene_lighting": scene_lighting,
        "scene_time": scene_time,
    }


def batch_json_to_tensors(
    state_jsons: List[dict],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Batch multiple state JSONs into batched tensors."""
    singles = [json_to_tensors(sj, device=device) for sj in state_jsons]
    return {k: torch.stack([s[k] for s in singles]) for k in singles[0]}


class StateEmbedder(nn.Module):
    """Encodes structured physical-state JSON into DiT-compatible tokens.

    Architecture:
      Per-character (max 3):
        [x, y, scale] → Linear(3, 64)          → [64]
        pose           → Embedding(11, 32)      → [32]
        facing         → Embedding(9, 16)       → [16]
        action         → Embedding(50, 48)      → [48]
        cloth_upper    → Embedding(15, 24)      → [24]
        cloth_lower    → Embedding(15, 24)      → [24]
        holding        → Embedding(39, 32)      → [32]
        concat → [240] per character, masked by char_mask

      Scene:
        location    → Embedding(30, 48) → [48]
        lighting    → Embedding(10, 16) → [16]
        time_of_day → Embedding(6, 16)  → [16]
        concat → [80]

      All concat: [240 × 3 + 80] = [800]
      MLP: Linear(800, 1024) → GELU → Linear(1024, num_tokens × dim)
      reshape → [num_tokens, dim]

      + Learnable slot embeddings (4 slots: char0, char1, char2, scene)
        distributed across num_tokens output tokens.
    """

    CHAR_DIM = 240  # per-character feature dim
    SCENE_DIM = 80
    HIDDEN_DIM = 1024

    def __init__(self, dim: int = 3072, num_tokens: int = NUM_STATE_TOKENS,
                 state_dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.state_dropout = state_dropout

        # --- Per-character encoders ---
        self.char_continuous_proj = nn.Linear(3, 64)
        self.pose_emb = nn.Embedding(len(POSE_VOCAB), 32)
        self.facing_emb = nn.Embedding(len(FACING_VOCAB), 16)
        self.action_emb = nn.Embedding(len(ACTION_VOCAB), 48)
        self.cloth_upper_emb = nn.Embedding(len(COLOR_VOCAB), 24)
        self.cloth_lower_emb = nn.Embedding(len(COLOR_VOCAB), 24)
        self.holding_emb = nn.Embedding(len(HOLDING_VOCAB), 32)

        # --- Scene encoders ---
        self.location_emb = nn.Embedding(len(LOCATION_VOCAB), 48)
        self.lighting_emb = nn.Embedding(len(LIGHTING_VOCAB), 16)
        self.time_emb = nn.Embedding(len(TIME_OF_DAY_VOCAB), 16)

        # --- MLP projection ---
        total_in = self.CHAR_DIM * MAX_CHARACTERS + self.SCENE_DIM  # 800
        self.mlp = nn.Sequential(
            nn.Linear(total_in, self.HIDDEN_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.HIDDEN_DIM, num_tokens * dim),
        )

        # --- Slot embeddings (distinguish char0/1/2 vs scene tokens) ---
        # Tokens 0-1: char0, 2-3: char1, 4-5: char2, 6-7: scene
        self.slot_embeddings = nn.Embedding(4, dim)  # 4 slots

        # Build slot assignment: which slot each token belongs to
        tokens_per_char = 2
        tokens_scene = num_tokens - tokens_per_char * MAX_CHARACTERS  # 2
        slot_ids = []
        for c in range(MAX_CHARACTERS):
            slot_ids.extend([c] * tokens_per_char)
        slot_ids.extend([3] * tokens_scene)  # slot 3 = scene
        self.register_buffer("slot_ids", torch.tensor(slot_ids, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training with frozen DiT."""
        for emb in [self.pose_emb, self.facing_emb, self.action_emb,
                     self.cloth_upper_emb, self.cloth_lower_emb, self.holding_emb,
                     self.location_emb, self.lighting_emb, self.time_emb,
                     self.slot_embeddings]:
            nn.init.normal_(emb.weight, std=0.02)

        nn.init.xavier_uniform_(self.char_continuous_proj.weight)
        nn.init.zeros_(self.char_continuous_proj.bias)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Zero-init the final projection bias for near-zero initial output
        final_linear = self.mlp[-1]
        nn.init.zeros_(final_linear.bias)
        # Also scale down final linear weight for stability
        with torch.no_grad():
            final_linear.weight.mul_(0.1)

    def _encode_characters(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode per-character features. Returns [B, MAX_CHARACTERS, CHAR_DIM]."""
        cont = self.char_continuous_proj(tensors["char_continuous"])     # [B, 3, 64]
        pose = self.pose_emb(tensors["char_pose"])                       # [B, 3, 32]
        facing = self.facing_emb(tensors["char_facing"])                 # [B, 3, 16]
        action = self.action_emb(tensors["char_action"])                 # [B, 3, 48]
        cloth_u = self.cloth_upper_emb(tensors["char_cloth_upper"])      # [B, 3, 24]
        cloth_l = self.cloth_lower_emb(tensors["char_cloth_lower"])      # [B, 3, 24]
        holding = self.holding_emb(tensors["char_holding"])              # [B, 3, 32]

        char_feats = torch.cat([cont, pose, facing, action,
                                cloth_u, cloth_l, holding], dim=-1)  # [B, 3, 240]

        # Mask out absent characters
        mask = tensors["char_mask"].unsqueeze(-1)  # [B, 3, 1]
        return char_feats * mask

    def _encode_scene(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode scene features. Returns [B, SCENE_DIM]."""
        loc = self.location_emb(tensors["scene_location"])     # [B, 1, 48]
        light = self.lighting_emb(tensors["scene_lighting"])   # [B, 1, 16]
        tod = self.time_emb(tensors["scene_time"])             # [B, 1, 16]
        return torch.cat([loc, light, tod], dim=-1).squeeze(1)  # [B, 80]

    def forward(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            tensors: Output of json_to_tensors() or batch_json_to_tensors(),
                     with batch dimension.

        Returns:
            [B, num_tokens, dim] state embedding tokens.
        """
        B = tensors["char_continuous"].shape[0]

        char_feats = self._encode_characters(tensors)  # [B, 3, 240]
        scene_feats = self._encode_scene(tensors)      # [B, 80]

        # Flatten characters and concat with scene
        combined = torch.cat([
            char_feats.reshape(B, -1),  # [B, 720]
            scene_feats,                # [B, 80]
        ], dim=-1)  # [B, 800]

        # State dropout: zero out entire state vector during training
        if self.training and self.state_dropout > 0:
            drop_mask = torch.rand(B, 1, device=combined.device) > self.state_dropout
            combined = combined * drop_mask.float()

        # MLP projection + reshape
        tokens = self.mlp(combined)  # [B, num_tokens * dim]
        tokens = tokens.reshape(B, self.num_tokens, self.dim)  # [B, 8, 3072]

        # Add slot embeddings
        slots = self.slot_embeddings(self.slot_ids)  # [num_tokens, dim]
        tokens = tokens + slots.unsqueeze(0)

        return tokens
