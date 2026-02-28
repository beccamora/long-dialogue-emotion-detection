"""
Long Dialogue Emotion Detection with Commonsense Graph Guidance
Reproduction & Component Analysis
"""
import torch
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache
from collections import Counter



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sanity_check(batch):
    print("Batch keys:", batch.keys())
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)
    print("utterances type:", type(batch["utterances"]))

set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def dialog_collate_fn(batch):
    # One dialogue per batch (as in the paper)
    assert len(batch) == 1
    item = batch[0]
    return {
        "input_ids": item["input_ids"],        # (U, T)
        "attention_mask": item["attention_mask"],
        "labels": item["labels"],              # (U,)
        "utterances": item["utterances"]
    }

"""### 1. Load DailyDialog"""

class DailyDialogDataset(Dataset):
    def __init__(self, split, tokenizer, max_len=128):
        self.data = load_dataset("roskoN/dailydialog", split=split)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        utterances = self.data[idx]["utterances"]
        emotions = self.data[idx]["emotions"]

        enc = self.tokenizer(
            utterances,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(emotions, dtype=torch.long),
            "utterances": utterances
        }

    def __len__(self):
        return len(self.data)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_ds = DailyDialogDataset("train", tokenizer, max_len=64)
val_ds   = DailyDialogDataset("validation", tokenizer, max_len=64)
test_ds  = DailyDialogDataset("test", tokenizer, max_len=64)

train_ds.data = train_ds.data.select(range(4000))
val_ds.data   = val_ds.data.select(range(500))
test_ds.data  = test_ds.data.select(range(500))

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=dialog_collate_fn)
val_loader   = DataLoader(val_ds, batch_size=1, collate_fn=dialog_collate_fn)
test_loader  = DataLoader(test_ds, batch_size=1, collate_fn=dialog_collate_fn)

print("Train dialogs:", len(train_ds))
print("Val dialogs:", len(val_ds))
print("Test dialogs:", len(test_ds))

batch = next(iter(train_loader))
sanity_check(batch)

def compute_class_weights(dataset, num_classes):
    counter = Counter()
    for i in range(len(dataset)):
        labels = dataset[i]["labels"].tolist()
        counter.update(labels)

    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = counter[c]

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # normalize
    return weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_weights = compute_class_weights(train_ds, num_classes=7).to(device)
print("Class weights:", class_weights)

"""### 2. Tokenizer + ATOMIC loader (cached)"""

atomic_ds = load_dataset("atomic", split="train")

def extract_atomic_reactions(dataset):
    reactions = []

    for ex in dataset:
        for r in ex["xReact"]:
            if r and r != "none":
                reactions.append(r.lower().strip())

        for r in ex["oReact"]:
            if r and r != "none":
                reactions.append(r.lower().strip())

    return list(set(reactions))

atomic_texts = extract_atomic_reactions(atomic_ds)
print("Atomic reactions:", len(atomic_texts))

# Sanity Check
atomic_texts[:20]

@lru_cache(maxsize=50000)
def retrieve_atomic_cpu(text, k=3):
    with torch.no_grad():
        emb = sbert.encode(text, convert_to_tensor=True)  # stays on CPU by default
        sims = F.cosine_similarity(atomic_embeds_cpu, emb.unsqueeze(0), dim=1)
        idx = sims.topk(k).indices
        vec = atomic_embeds_cpu[idx].mean(dim=0)  # CPU tensor
    return vec  # CPU

sbert = SentenceTransformer("all-MiniLM-L6-v2").to(device)

for p in sbert.parameters():
    p.requires_grad = False

atomic_embeds_cpu = sbert.encode(
    atomic_texts, convert_to_tensor=True, batch_size=64, device="cpu"
)

def precompute_atomic_for_hf_dataset(hf_dataset):
    cache = {}
    for ex in hf_dataset:
        for u in ex["utterances"]:
            if u not in cache:
                cache[u] = retrieve_atomic_cpu(u)
    return cache

print("Precomputing atomic_cache (train+val+test)...")
atomic_cache = {}
atomic_cache.update(precompute_atomic_for_hf_dataset(train_ds.data))
atomic_cache.update(precompute_atomic_for_hf_dataset(val_ds.data))
atomic_cache.update(precompute_atomic_for_hf_dataset(test_ds.data))
print("Cache size:", len(atomic_cache))

v = retrieve_atomic_cpu("I feel very sad today")
print(v.shape)

"""### 3. Graph Utils"""

def build_adjacency_topk(embs, k=4):
    U = embs.size(0)

    # Handle very short dialogues safely
    if U <= 1:
        return torch.eye(U, device=embs.device)

    k_eff = min(k, U - 1)

    A = torch.eye(U, device=embs.device)

    # temporal edges
    for i in range(U - 1):
        A[i, i+1] = 1
        A[i+1, i] = 1

    norm = F.normalize(embs, p=2, dim=1)
    sim = norm @ norm.T
    sim.fill_diagonal_(-1e9)

    idx = sim.topk(k_eff, dim=1).indices
    for i in range(U):
        A[i, idx[i]] = 1
        A[idx[i], i] = 1

    deg = A.sum(dim=1)
    D = torch.diag(torch.pow(deg + 1e-8, -0.5))
    return D @ A @ D

class SimpleGCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, A, X):
        return F.relu(A @ self.lin(X))

class TopicExtractor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, X):
        w = torch.softmax(self.attn(X).squeeze(-1), dim=0)
        return (w.unsqueeze(1) * X).sum(dim=0)

x = torch.randn(5, 768).to(device)
A = build_adjacency_topk(x, k=3)
print(A.shape)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_confusion(model, loader, title):
    model.eval()
    gold, preds = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["utterances"]

            logits = model(ids, mask, texts)
            preds.extend(logits.argmax(1).cpu().numpy())
            gold.extend(labels.cpu().numpy())

    cm = confusion_matrix(gold, preds, normalize="true")
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="magma")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def get_gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return None

"""### 4. Models: T1 / T2 / T3"""

class TextOnlyModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.cls = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, texts=None):
        out = self.encoder(input_ids, attention_mask)
        h = out.last_hidden_state[:, 0]
        return self.cls(h)

class TextAtomicModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.cls = nn.Linear(768, num_labels)
        self.atomic_proj = nn.Linear(384, 768)

    def forward(self, input_ids, attention_mask, texts):
        out = self.encoder(input_ids, attention_mask)
        h = out.last_hidden_state[:, 0]

        atomic = torch.stack([atomic_cache.get(t, retrieve_atomic_cpu(t)) for t in texts]).to(h.device)
        atomic = self.atomic_proj(atomic)

        h = F.layer_norm(h, h.shape[-1:])
        atomic = F.layer_norm(atomic, atomic.shape[-1:])

        h = h + atomic

        return self.cls(h)

class FullModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.gcn = SimpleGCNLayer(768)
        self.dropout = nn.Dropout(0.2)          # reduced dropout
        self.topic = TopicExtractor(768)
        self.atomic_proj = nn.Linear(384, 768)
        self.cls = nn.Linear(768 * 2, num_labels)
        self.emotion_proj = nn.Linear(768, 128, bias=False)

    def forward(self, input_ids, attention_mask, texts):
        # ===== 1. Text encoding =====
        out = self.encoder(input_ids, attention_mask)
        h_text = out.last_hidden_state[:, 0]     # (U, 768)

        # ===== 2. Graph on TEXT ONLY =====
        h_em = F.normalize(self.emotion_proj(h_text), dim=1)
        A = build_adjacency_topk(h_em, k=4)
        g = self.gcn(A, h_text)

        # ===== 3. Residual + dropout =====
        g = g + h_text
        g = self.dropout(g)

        # ===== 4. ATOMIC fusion AFTER graph =====
        atomic = torch.stack(
            [atomic_cache.get(t, retrieve_atomic_cpu(t)) for t in texts]
        ).to(h_text.device)
        atomic = self.atomic_proj(atomic)

        g = F.layer_norm(g, g.shape[-1:])
        atomic = F.layer_norm(atomic, atomic.shape[-1:])
        g = g + atomic

        # ===== 5. Topic aggregation (paper-style) =====
        topic = self.topic(g)                   # (768,)
        topic_rep = topic.unsqueeze(0).expand_as(g)

        # ===== 6. Classification =====
        fused = torch.cat([g, topic_rep], dim=1)  # (U, 1536)
        logits = self.cls(fused)

        return logits

batch = next(iter(train_loader))
ids = batch["input_ids"].to(device)
mask = batch["attention_mask"].to(device)
texts = batch["utterances"]

model = FullModel(7).to(device)
out = model(ids, mask, texts)
print(out.shape)

"""### 5. Training + Evaluation (end-to-end)"""

def train_epoch(model, loader, optimizer):
    model.train()
    loss_sum = 0
    start = time.time()

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        texts = batch["utterances"]

        logits = model(ids, mask, texts)
        loss = F.cross_entropy(logits, labels, weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    epoch_time = time.time() - start
    return loss_sum / len(loader), epoch_time

def evaluate(model, loader):
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["utterances"]

            logits = model(ids, mask, texts)

            preds.extend(logits.argmax(1).cpu().numpy())
            gold.extend(labels.cpu().numpy())

    acc = accuracy_score(gold, preds)
    f1_macro = f1_score(gold, preds, average="macro")
    f1_weighted = f1_score(gold, preds, average="weighted")

    return acc, f1_macro, f1_weighted, gold, preds

def main():
    """### 6. Run Experiments"""

    def configure_last_layer_finetune(model):
        # Freeze everything
        for p in model.encoder.parameters():
            p.requires_grad = False

        # Unfreeze last RoBERTa layer
        for name, p in model.encoder.named_parameters():
            if "layer.11" in name:
                p.requires_grad = True

        # Heads always train
        for name, p in model.named_parameters():
            if not name.startswith("encoder"):
                p.requires_grad = True


    def make_optimizer(model):
        encoder_params = []
        head_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("encoder"):
                encoder_params.append(p)
            else:
                head_params.append(p)

        return torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": 1e-5},
                {"params": head_params, "lr": 2e-4},
            ],
            weight_decay=0.01
        )

    results = {}
    param_counts = {}

    # =========================
    # T1: Text-only baseline
    # =========================

    model_t1 = TextOnlyModel(7).to(device)

    # Freeze everything first
    for p in model_t1.encoder.parameters():
        p.requires_grad = False

    # Unfreeze ONLY last RoBERTa layer
    for name, p in model_t1.encoder.named_parameters():
        if "layer.11" in name:
            p.requires_grad = True

    # Classifier head should always train
    for p in model_t1.cls.parameters():
        p.requires_grad = True

    # Two learning rates (CRITICAL)
    encoder_params = []
    head_params = []

    for name, p in model_t1.named_parameters():
        if p.requires_grad:
            if name.startswith("encoder"):
                encoder_params.append(p)
            else:
                head_params.append(p)

    opt_t1 = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": 1e-5},
            {"params": head_params, "lr": 2e-4},
        ],
        weight_decay=0.01
    )

    times = []

    for epoch in range(3):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        loss, t = train_epoch(model_t1, train_loader, opt_t1)
        mem = get_gpu_mem_mb()
        times.append(t)

        acc, f1, f1_weighted, _, _ = evaluate(model_t1, val_loader)
        print(f"[T1-lastFT] Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, f1={f1:.4f}")
        print(f"Peak GPU memory: {mem:.1f} MB")
        print("------")

    print("Avg epoch time:", sum(times)/len(times))

    acc, f1, f1_weighted, _, _ = evaluate(model_t1, test_loader)
    results["T1_Text"] = {"acc": acc, "f1": f1, "f1_weighted": f1_weighted}
    results["T1_Text"]["avg_epoch_time"] = sum(times)/len(times)

    plot_confusion(model_t1, val_loader, "Validation Confusion Matrix - T1")
    plot_confusion(model_t1, test_loader, "Test Confusion Matrix - T1")

    param_counts["T1"] = count_trainable_params(model_t1)

    del model_t1; opt_t1; torch.cuda.empty_cache()

    # =========================
    # T2: Text + ATOMIC
    # =========================

    model_t2 = TextAtomicModel(7).to(device)

    configure_last_layer_finetune(model_t2)
    opt_t2 = make_optimizer(model_t2)

    times = []

    for epoch in range(5):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        loss, t = train_epoch(model_t2, train_loader, opt_t2)
        mem = get_gpu_mem_mb()
        times.append(t)
        acc, f1, f1_weighted, _, _ = evaluate(model_t2, val_loader)
        print(f"[T2] Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, f1={f1:.4f}")
        print(f"Peak GPU memory: {mem:.1f} MB")
        print("------")

    print("Avg epoch time:", sum(times)/len(times))

    acc, f1, f1_weighted, _, _ = evaluate(model_t2, test_loader)
    results["T2_Text+Atomic"] = {"acc": acc, "f1": f1, "f1_weighted": f1_weighted}
    results["T2_Text+Atomic"]["avg_epoch_time"] = sum(times)/len(times)

    plot_confusion(model_t2, val_loader, "Validation Confusion Matrix - T2")
    plot_confusion(model_t2, test_loader, "Test Confusion Matrix - T2")

    param_counts["T2"] = count_trainable_params(model_t2)

    del model_t2; opt_t2; torch.cuda.empty_cache()

    # =========================
    # T3: Full model
    # =========================

    model_t3 = FullModel(7).to(device)

    configure_last_layer_finetune(model_t3)
    opt_t3 = make_optimizer(model_t3)
    times = []

    for epoch in range(7):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        loss, t = train_epoch(model_t3, train_loader, opt_t3)
        mem = get_gpu_mem_mb()
        times.append(t)
        acc, f1, f1_weighted, _, _ = evaluate(model_t3, val_loader)
        print(f"[T3] Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, f1={f1:.4f}")
        print(f"Peak GPU memory: {mem:.1f} MB")
        print("------")

    print("Avg epoch time:", sum(times)/len(times))

    acc, f1, f1_weighted, _, _ = evaluate(model_t3, test_loader)
    results["T3_Full"] = {"acc": acc, "f1": f1, "f1_weighted": f1_weighted}
    results["T3_Full"]["avg_epoch_time"] = sum(times)/len(times)

    plot_confusion(model_t3, val_loader, f"Validation Confusion Matrix - T3")
    plot_confusion(model_t3, test_loader, "Test Confusion Matrix - T3")

    param_counts["T3"] = count_trainable_params(model_t3)

    del model_t3; opt_t3; torch.cuda.empty_cache()

    results

    """### 7. Ablation"""

    class FullNoGraphNoTopic(nn.Module):
        def __init__(self, num_labels):
            super().__init__()
            self.encoder = RobertaModel.from_pretrained("roberta-base")
            self.atomic_proj = nn.Linear(384, 768)
            self.cls = nn.Linear(768 * 2, num_labels)

        def forward(self, input_ids, attention_mask, texts):
            out = self.encoder(input_ids, attention_mask)
            h = out.last_hidden_state[:, 0]   # text only

            atomic = torch.stack(
                [atomic_cache.get(t, retrieve_atomic_cpu(t)) for t in texts]
            ).to(h.device)
            atomic = self.atomic_proj(atomic)

            h = F.layer_norm(h, h.shape[-1:])
            atomic = F.layer_norm(atomic, atomic.shape[-1:])

            h = h + atomic

            # same shape as FullModel classifier input
            return self.cls(torch.cat([h, h], dim=1))

    # =========================
    # Ablation: T3 without Graph+Topic (FullNoGraphNoTopic)
    # =========================

    model_ab_no_topic = FullNoGraphNoTopic(7).to(device)

    configure_last_layer_finetune(model_ab_no_topic)
    opt_ab_no_topic = make_optimizer(model_ab_no_topic)

    times = []
    for epoch in range(7):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        loss, t = train_epoch(model_ab_no_topic, train_loader, opt_ab_no_topic)
        mem = get_gpu_mem_mb()
        times.append(t)
        acc, f1, f1_weighted, _, _ = evaluate(model_ab_no_topic, val_loader)
        print(f"[Abl-NoGraphNoTopic] Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, f1={f1:.4f}")
        print(f"Peak GPU memory: {mem:.1f} MB")
        print("------")

    print("Avg epoch time:", sum(times)/len(times))

    acc, f1, f1_weighted, gold_ab, preds_ab = evaluate(model_ab_no_topic, test_loader)
    results["Abl_NoGraphNoTopic"] = {"acc": acc, "f1": f1, "f1_weighted": f1_weighted}
    results["Abl_NoGraphNoTopic"]["avg_epoch_time"] = sum(times)/len(times)

    plot_confusion(model_ab_no_topic, val_loader, f"Validation Confusion Matrix - Abl_NoGraphNoTopic")
    plot_confusion(model_ab_no_topic, test_loader, "Test Confusion Matrix - Abl_NoGraphNoTopic")

    param_counts["Abl_NoGraphNoTopic"] = count_trainable_params(model_ab_no_topic)

    del model_ab_no_topic; opt_ab_no_topic; torch.cuda.empty_cache()

    class FullNoGraph(nn.Module):
        def __init__(self, num_labels):
            super().__init__()
            self.encoder = RobertaModel.from_pretrained("roberta-base")
            self.topic = TopicExtractor(768)
            self.atomic_proj = nn.Linear(384, 768)
            self.cls = nn.Linear(768 * 2, num_labels)

        def forward(self, input_ids, attention_mask, texts):
            out = self.encoder(input_ids, attention_mask)
            h = out.last_hidden_state[:, 0]

            atomic = torch.stack(
                [atomic_cache.get(t, retrieve_atomic_cpu(t)) for t in texts]
            ).to(h.device)
            atomic = self.atomic_proj(atomic)

            h = F.layer_norm(h, h.shape[-1:])
            atomic = F.layer_norm(atomic, atomic.shape[-1:])
            h = h + atomic

            topic = self.topic(h)
            topic_rep = topic.unsqueeze(0).expand_as(h)

            fused = torch.cat([h, topic_rep], dim=1)
            return self.cls(fused)

    # =========================
    # Ablation: T3 without Graph (FullNoGraph)
    # =========================

    model_ab_no_graph = FullNoGraph(7).to(device)

    configure_last_layer_finetune(model_ab_no_graph)
    opt_ab_no_graph = make_optimizer(model_ab_no_graph)


    times = []

    for epoch in range(7):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        loss, t = train_epoch(model_ab_no_graph, train_loader, opt_ab_no_graph)
        mem = get_gpu_mem_mb()
        times.append(t)
        acc, f1, f1_weighted, _, _ = evaluate(model_ab_no_graph, val_loader)
        print(f"[Abl-NoGraph] Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, f1={f1:.4f}")
        print(f"Peak GPU memory: {mem:.1f} MB")
        print("------")

    print("Avg epoch time:", sum(times)/len(times))

    acc, f1, f1_weighted, gold_ng, preds_ng = evaluate(model_ab_no_graph, test_loader)
    results["Abl_NoGraph"] = {"acc": acc, "f1": f1, "f1_weighted": f1_weighted}
    results["Abl_NoGraph"]["avg_epoch_time"] = sum(times)/len(times)

    plot_confusion(model_ab_no_graph, val_loader, f"Validation Confusion Matrix - Abl-NoGraph")
    plot_confusion(model_ab_no_graph, test_loader, "Test Confusion Matrix - Abl-NoGraph")

    param_counts["Abl_NoGraph"] = count_trainable_params(model_ab_no_graph)

    del model_ab_no_graph; torch.cuda.empty_cache()

    """### 9. Auxiliar Information"""

    results["T1_Text"]["params"] = param_counts["T1"]
    results["T2_Text+Atomic"]["params"] = param_counts["T2"]
    results["T3_Full"]["params"] = param_counts["T3"]
    results["Abl_NoGraph"]["params"] = param_counts["Abl_NoGraph"]
    results["Abl_NoGraphNoTopic"]["params"] = param_counts["Abl_NoGraphNoTopic"]

    print("T1 params:", param_counts["T1"])
    print("T2 params:", param_counts["T2"])
    print("T3 params:", param_counts["T3"])
    print("Abl_NoGraph:", param_counts["Abl_NoGraph"])
    print("Abl_NoGraphNoTopic:", param_counts["Abl_NoGraphNoTopic"])

    import json

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("params.json", "w") as f:
        json.dump(param_counts, f, indent=2)

        pass

if __name__ == "__main__":
    main()

