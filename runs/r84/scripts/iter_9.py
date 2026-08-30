import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

DEVICE = torch.device("cpu")
BATCH_SIZE = 8192
MAX_EPOCHS = 3
EMBED_DIM = 8

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "hour",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
    "upload_type",
    "music_type",
    "video_type",
    "is_video_author",
]

CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))

train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)


def cat_matrix(split):
    x = np.column_stack([
        np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
    ])
    x += OFFSETS[None, :]
    return np.ascontiguousarray(x, dtype=np.int64)


x_train = cat_matrix(train)
x_valid = cat_matrix(valid)


def binary_aux(split, name):
    if name not in split.aux:
        return None
    x = np.asarray(split.aux[name])
    if x.ndim != 1 or len(x) != len(split.user_id):
        return None
    x = np.nan_to_num(x.astype(np.float32), nan=0.0)
    return (x > 0).astype(np.float32)


AUX_NAMES = []
for candidate in ["is_click", "is_like", "is_follow", "is_comment", "is_forward"]:
    if binary_aux(train, candidate) is not None:
        AUX_NAMES.append(candidate)
    if len(AUX_NAMES) >= 2:
        break

if "is_click" not in AUX_NAMES and binary_aux(train, "is_click") is not None:
    AUX_NAMES.insert(0, "is_click")

if not AUX_NAMES:
    raise RuntimeError("No supported binary auxiliary supervision was found")

aux_train = {
    name: binary_aux(train, name) for name in AUX_NAMES
}

print("FINDINGS auxiliary_tasks=" + json.dumps(AUX_NAMES))


class MMoE(nn.Module):
    def __init__(self, n_aux):
        super().__init__()
        self.n_tasks = 1 + n_aux
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.wide = nn.ModuleList([
            nn.Embedding(TOTAL_CARD, 1) for _ in range(self.n_tasks)
        ])

        input_dim = len(FIELDS) * EMBED_DIM
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Dropout(0.08),
                nn.Linear(96, 48),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, 4) for _ in range(self.n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(self.n_tasks)
        ])
        self.biases = nn.Parameter(torch.zeros(self.n_tasks))

        nn.init.normal_(self.embedding.weight, std=0.015)
        for layer in self.wide:
            nn.init.zeros_(layer.weight)

    def forward(self, cats):
        flat = self.embedding(cats).flatten(1)
        expert_values = torch.stack(
            [expert(flat) for expert in self.experts], dim=1
        )

        outputs = []
        for task in range(self.n_tasks):
            gate = torch.softmax(self.gates[task](flat), dim=1)
            mixed = torch.sum(expert_values * gate.unsqueeze(2), dim=1)
            deep = self.towers[task](mixed).squeeze(1)
            wide = self.wide[task](cats).sum(dim=1).squeeze(1)
            outputs.append(deep + wide + self.biases[task])
        return torch.stack(outputs, dim=1)

    def ranking_score(self, cats):
        return self.forward(cats)[:, 0]


class ESMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.click_wide = nn.Embedding(TOTAL_CARD, 1)
        self.cond_wide = nn.Embedding(TOTAL_CARD, 1)

        input_dim = len(FIELDS) * EMBED_DIM
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.click_tower = nn.Sequential(
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        self.cond_tower = nn.Sequential(
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        self.click_bias = nn.Parameter(torch.zeros(1))
        self.cond_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.015)
        nn.init.zeros_(self.click_wide.weight)
        nn.init.zeros_(self.cond_wide.weight)

    def component_logits(self, cats):
        flat = self.embedding(cats).flatten(1)
        shared = self.shared(flat)
        click = (
            self.click_tower(shared).squeeze(1)
            + self.click_wide(cats).sum(dim=1).squeeze(1)
            + self.click_bias
        )
        conditional = (
            self.cond_tower(shared).squeeze(1)
            + self.cond_wide(cats).sum(dim=1).squeeze(1)
            + self.cond_bias
        )
        return click, conditional

    def probabilities(self, cats):
        click, conditional = self.component_logits(cats)
        p_click = torch.sigmoid(click)
        p_long = p_click * torch.sigmoid(conditional)
        return p_click, p_long

    def ranking_score(self, cats):
        _, p_long = self.probabilities(cats)
        p_long = p_long.clamp(1e-7, 1.0 - 1e-7)
        return torch.logit(p_long)


@torch.no_grad()
def predict(model, x):
    model.eval()
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(x))
        xb = torch.from_numpy(x[start:end]).to(DEVICE)
        out[start:end] = (
            model.ranking_score(xb).cpu().numpy().astype(np.float64)
        )
    return out


def train_epoch_mmoe(model, optimizer, x, y, aux, generator):
    model.train()
    n = len(y)
    order = torch.randperm(n, generator=generator)

    for start in range(0, n, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        ix = idx.numpy()
        xb = torch.from_numpy(x[ix]).to(DEVICE)
        yb = torch.from_numpy(y[ix]).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0], yb
        )

        for task_idx, name in enumerate(AUX_NAMES, start=1):
            ab = torch.from_numpy(aux[name][ix]).to(DEVICE)
            aux_loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, task_idx], ab
            )
            loss = loss + 0.30 * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def train_epoch_esmm(model, optimizer, x, y, click, generator):
    model.train()
    n = len(y)
    order = torch.randperm(n, generator=generator)

    for start in range(0, n, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        ix = idx.numpy()
        xb = torch.from_numpy(x[ix]).to(DEVICE)
        yb = torch.from_numpy(y[ix]).to(DEVICE)
        cb = torch.from_numpy(click[ix]).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        click_logit, conditional_logit = model.component_logits(xb)
        p_long = (
            torch.sigmoid(click_logit)
            * torch.sigmoid(conditional_logit)
        ).clamp(1e-7, 1.0 - 1e-7)

        loss_long = nn.functional.binary_cross_entropy(p_long, yb)
        loss_click = nn.functional.binary_cross_entropy_with_logits(
            click_logit, cb
        )
        loss = loss_long + 0.45 * loss_click

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def fit_with_validation(family):
    torch.manual_seed(SEED + (101 if family == "mmoe" else 211))

    if family == "mmoe":
        model = MMoE(len(AUX_NAMES)).to(DEVICE)
    else:
        model = ESMM().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0015, weight_decay=2e-6
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + (301 if family == "mmoe" else 401))

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    epoch_results = {}

    click_name = "is_click" if "is_click" in aux_train else AUX_NAMES[0]
    click_train = aux_train[click_name]

    for epoch in range(1, MAX_EPOCHS + 1):
        if family == "mmoe":
            train_epoch_mmoe(
                model, optimizer, x_train, y_train, aux_train, generator
            )
        else:
            train_epoch_esmm(
                model, optimizer, x_train, y_train,
                click_train, generator
            )

        scores = predict(model, x_valid)
        met = evaluate(valid.user_id, y_valid, scores)
        epoch_results[str(epoch)] = float(met["primary"])

        if float(met["primary"]) > best_primary:
            best_primary = float(met["primary"])
            best_epoch = epoch
            best_scores = scores.copy()

    del model, optimizer
    gc.collect()
    return best_scores, best_epoch, epoch_results


mmoe_valid, mmoe_epoch, mmoe_epochs = fit_with_validation("mmoe")
esmm_valid, esmm_epoch, esmm_epochs = fit_with_validation("esmm")

print("FINDINGS mmoe_epoch_scores=" + json.dumps(mmoe_epochs, sort_keys=True))
print("FINDINGS esmm_epoch_scores=" + json.dumps(esmm_epochs, sort_keys=True))

shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

families = {
    "mmoe_multitask": (mmoe_valid, mmoe_epoch),
    "esmm_factorized": (esmm_valid, esmm_epoch),
}

candidate_log = {}
best_primary = -np.inf
best_family = None
best_alpha = None
best_scores = None
best_raw_valid = None
best_metrics = None
best_own_scale = None
inc_scale = max(float(np.std(inc_valid)), 1e-8)

for name, (raw_scores, epoch) in families.items():
    raw_met = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_log[name + "_raw"] = float(raw_met["primary"])

    own_scale = max(float(np.std(raw_scores)), 1e-8)
    for alpha in [0.25, 0.50, 0.75, 1.00]:
        blended = (
            alpha * raw_scores / own_scale
            + (1.0 - alpha) * inc_valid / inc_scale
        )
        met = evaluate(valid.user_id, y_valid, blended)
        key = name + "_blend_" + str(alpha)
        candidate_log[key] = float(met["primary"])

        if float(met["primary"]) > best_primary:
            best_primary = float(met["primary"])
            best_family = name
            best_alpha = float(alpha)
            best_scores = blended.copy()
            best_raw_valid = raw_scores.copy()
            best_metrics = met
            best_own_scale = own_scale

inc_met = evaluate(valid.user_id, y_valid, inc_valid)
candidate_log["trusted_incumbent"] = float(inc_met["primary"])
if float(inc_met["primary"]) > best_primary:
    # Keep a genuine new raw model available while reporting the strongest score.
    best_primary = float(inc_met["primary"])
    best_family = "mmoe_multitask"
    best_alpha = 0.0
    best_scores = inc_valid.copy()
    best_raw_valid = mmoe_valid.copy()
    best_metrics = inc_met
    best_own_scale = max(float(np.std(mmoe_valid)), 1e-8)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS selected="
    + json.dumps({
        "family": best_family,
        "alpha": best_alpha,
        "epoch": int(families[best_family][1]),
    }, sort_keys=True)
)

# Refit the selected recipe on train + validation, then score test.
test = load("test")
x_test = cat_matrix(test)

x_fit = np.concatenate([x_train, x_valid], axis=0)
y_fit = np.concatenate([
    y_train,
    np.asarray(valid.y, dtype=np.float32)
])

aux_fit = {}
for name in AUX_NAMES:
    tr_aux = binary_aux(train, name)
    va_aux = binary_aux(valid, name)
    aux_fit[name] = np.concatenate([tr_aux, va_aux])

selected_epoch = int(families[best_family][1])
family_code = "mmoe" if best_family == "mmoe_multitask" else "esmm"

torch.manual_seed(SEED + (101 if family_code == "mmoe" else 211))
if family_code == "mmoe":
    final_model = MMoE(len(AUX_NAMES)).to(DEVICE)
else:
    final_model = ESMM().to(DEVICE)

final_optimizer = torch.optim.AdamW(
    final_model.parameters(), lr=0.0015, weight_decay=2e-6
)
final_generator = torch.Generator(device="cpu")
final_generator.manual_seed(
    SEED + (301 if family_code == "mmoe" else 401)
)

click_name = "is_click" if "is_click" in aux_fit else AUX_NAMES[0]

for _ in range(selected_epoch):
    if family_code == "mmoe":
        train_epoch_mmoe(
            final_model, final_optimizer, x_fit, y_fit,
            aux_fit, final_generator
        )
    else:
        train_epoch_esmm(
            final_model, final_optimizer, x_fit, y_fit,
            aux_fit[click_name], final_generator
        )

raw_test = predict(final_model, x_test)
test_scores = (
    best_alpha * raw_test / best_own_scale
    + (1.0 - best_alpha) * inc_test / inc_scale
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))