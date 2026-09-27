import gc
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError:
    raise ImportError("Please install PyTorch Geometric: pip install torch-geometric")

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("WARNING: thop not installed (pip install thop). "
          "FLOPs-based hardware feasibility checks will be skipped.")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from scipy.stats import norm as scipy_norm
    HAS_BO_DEPS = True
except ImportError:
    HAS_BO_DEPS = False
    print("WARNING: scikit-learn/scipy not installed. "
          "GNN hyperparameter Bayesian Optimization will fall back to a fixed default config.")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Global seeding utility ---
def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches. Call once per seed, before any
    randomness for that seed's run (data generation AND model init)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- (5) Multiple training/evaluation seeds, per reviewer request ---
EXPERIMENT_SEEDS: List[int] = [42, 43, 44]
BO_SEED = 42  # BO is run ONCE, independently of the experiment seeds above

# --- Set to True for a quick smoke test, False for the full run ---
QUICK_TEST = False

if QUICK_TEST:
    SNR_RANGE = np.array([-20, -15, -10, -5, 0, 5, 10])
    N_MC = 1000
    N_MC_CALIB = 1000
    SENSING_SAMPLES = 1024
    SEQUENCE_LENGTH = 64
    TRAIN_WINDOWS_PER_TRIAL = 4
    BATCH_SIZE = 64
    EPOCHS = 20
    N_NODES = 4
    BO_N_MC = 300
    BO_N_MC_CALIB = 200
    BO_EPOCHS = 8
    BO_N_INITIAL = 6
    BO_N_ITERS = 15
    EXPERIMENT_SEEDS = [42, 43]  # fewer seeds for the smoke test
else:
    SNR_RANGE = np.arange(-10, 12, 2)
    N_MC = 10000
    N_MC_CALIB = 2000
    SENSING_SAMPLES = 2048
    SEQUENCE_LENGTH = 100
    TRAIN_WINDOWS_PER_TRIAL = 4
    BATCH_SIZE = 128
    EPOCHS = 100
    N_NODES = 8
    BO_N_MC = 1500
    BO_N_MC_CALIB = 800
    BO_EPOCHS = 15
    BO_N_INITIAL = 8
    BO_N_ITERS = 20

SAMPLES_PER_SYMBOL = 4

# --- Fixed Parameters ---
# NOTE: PU_PRIOR is used as an i.i.d. Bernoulli(PU_PRIOR) draw per trial in
# generate_cooperative_sensing_dataset() -- a memoryless model, NOT a
# two-state Markov chain (corrected in the v5.0 patch and unchanged here).
PU_PRIOR = 0.30
TARGET_PD = 0.90
TARGET_PF = 0.01

# --- Rural propagation parameters ---
# (1) RENAMED: this is a free-space + breakpoint-distance (two-slope)
# log-distance path-loss model with log-normal shadowing and a simple
# vegetation-attenuation term -- NOT an implementation of the ITU-R P.1546
# empirical curves. Numerically unchanged from the prior version; only the
# name and documentation are corrected.
FREQ_MHZ = 600.0
TX_HEIGHT_M = 30.0
RX_HEIGHT_M = 1.5
D_MIN_KM = 2.0
D_MAX_KM = 10.0
VEG_MAX_DEPTH_M = 50.0
VEG_MAX_ATTEN_DB = 15.0
VEG_SPECIFIC_ATTEN_DB_PER_M = 0.5
SHADOWING_STD_DB = 8.0

# --- FLOPs / hardware feasibility ---
SENSING_INTERVAL_S = 0.1
HW_UTILIZATION_MARGIN = 0.5

# --- GNN hyperparameter Bayesian Optimization ---
BO_REPRESENTATIVE_SNRS = [-15.0, -5.0, 10.0]
BO_PF_PENALTY_LAMBDA = 5.0

# --- Training stability ---
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5

# ==============================================================================
# 2. RURAL PATH-LOSS MODEL (two-slope breakpoint model; see note above)
# ==============================================================================
def two_slope_breakpoint_path_loss_db(freq_mhz, distance_km, h1_m, h2_m):
    """Free-space path loss with a breakpoint-distance excess-loss term and
    simple antenna-height gain corrections. This is NOT ITU-R P.1546 -- see
    module docstring, item (1)."""
    fspl_db = 32.45 + 20 * np.log10(freq_mhz) + 20 * np.log10(distance_km)
    d_break_km = 4.12 * (np.sqrt(h1_m) + np.sqrt(h2_m))
    excess_loss_db = np.where(
        distance_km > d_break_km,
        30.0 * np.log10(np.maximum(distance_km, 1e-9) / d_break_km),
        0.0,
    )
    h1_gain_db = 20.0 * np.log10(max(h1_m, 1.0) / 10.0)
    h2_gain_db = 10.0 * np.log10(max(h2_m, 1.0) / 10.0)
    return fspl_db + excess_loss_db - h1_gain_db - h2_gain_db

def vegetation_attenuation_db(depth_m, max_atten_db=VEG_MAX_ATTEN_DB,
                               specific_atten_db_per_m=VEG_SPECIFIC_ATTEN_DB_PER_M):
    return max_atten_db * (1.0 - np.exp(-specific_atten_db_per_m * depth_m / max_atten_db))

def _compute_avg_channel_loss_db(n_samples=2000):
    d_km = np.random.uniform(D_MIN_KM, D_MAX_KM, size=n_samples)
    veg_depth_m = np.random.uniform(0, VEG_MAX_DEPTH_M, size=n_samples)
    pl = two_slope_breakpoint_path_loss_db(FREQ_MHZ, d_km, TX_HEIGHT_M, RX_HEIGHT_M)
    veg = vegetation_attenuation_db(veg_depth_m)
    return float(np.mean(pl + veg))

# ==============================================================================
# 3. SIGNAL GENERATION -- unchanged (numerically) from v5.0
# ==============================================================================
_QPSK_CONSTELLATION = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

def generate_received_signal_rural_batch(snr_db, n_samples, pu_states, sps=SAMPLES_PER_SYMBOL,
                                          avg_channel_loss_db=None):
    B = pu_states.shape[0]
    d_km = np.random.uniform(D_MIN_KM, D_MAX_KM, size=B)
    path_loss_db = two_slope_breakpoint_path_loss_db(FREQ_MHZ, d_km, TX_HEIGHT_M, RX_HEIGHT_M)
    shadowing_db = np.random.normal(0, SHADOWING_STD_DB, size=B)
    veg_depth_m = np.random.uniform(0, VEG_MAX_DEPTH_M, size=B)
    veg_attenuation_db = vegetation_attenuation_db(veg_depth_m)
    total_channel_loss_db = path_loss_db + shadowing_db + veg_attenuation_db

    effective_snr_db = snr_db - (total_channel_loss_db - avg_channel_loss_db)
    snr_linear = 10 ** (effective_snr_db / 10)

    noise = (np.random.randn(B, n_samples) + 1j * np.random.randn(B, n_samples)) / np.sqrt(2)
    h_real = np.random.normal(0, 1 / np.sqrt(2), size=B)
    h_imag = np.random.normal(0, 1 / np.sqrt(2), size=B)
    h = h_real + 1j * h_imag

    n_symbols = int(np.ceil(n_samples / sps)) + 1
    bits = np.random.randint(0, 4, size=(B, n_symbols))
    symbols = _QPSK_CONSTELLATION[bits]
    s = np.repeat(symbols, sps, axis=1)[:, :n_samples]

    active = pu_states.astype(bool)
    y = noise.copy()
    if active.any():
        y[active] = np.sqrt(snr_linear)[active, None] * h[active, None] * s[active] + noise[active]

    return y.real.astype(np.float32), y.imag.astype(np.float32)

# ==============================================================================
# 4. DATASET GENERATION -- unchanged from v5.0
# ==============================================================================
def generate_cooperative_sensing_dataset(snr_db, n_mc, sensing_samples, seq_length,
                                          avg_channel_loss_db, n_nodes=N_NODES, n_windows=1):
    max_start = max(1, sensing_samples - seq_length)
    n_windows = max(1, min(n_windows, max_start))

    pu_state_trial = (np.random.rand(n_mc) < PU_PRIOR).astype(np.float32)
    Y = np.repeat(pu_state_trial, n_windows).reshape(-1, 1)
    pu_states_expanded = np.repeat(pu_state_trial, n_windows * n_nodes)

    i_sig, q_sig = generate_received_signal_rural_batch(
        snr_db, sensing_samples, pu_states_expanded, SAMPLES_PER_SYMBOL,
        avg_channel_loss_db=avg_channel_loss_db)

    B = pu_states_expanded.shape[0]
    starts = (np.random.randint(0, max_start, size=B) if max_start > 1
              else np.zeros(B, dtype=int))
    win_idx = starts[:, None] + np.arange(seq_length)[None, :]
    i_win = np.take_along_axis(i_sig, win_idx, axis=1)
    q_win = np.take_along_axis(q_sig, win_idx, axis=1)

    X = np.stack([i_win, q_win], axis=-1)
    X = X.reshape(n_mc, n_windows, n_nodes, seq_length, 2)
    X = X.reshape(n_mc * n_windows, n_nodes, seq_length, 2).astype(np.float32)

    return torch.from_numpy(X), torch.from_numpy(Y.astype(np.float32))

# ==============================================================================
# 5. COOPERATIVE ENERGY DETECTION (BASELINE) -- (2) TRUE OR-FUSION
# ==============================================================================
def wilson_ci_95(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """(4) Wilson score interval. Does not degenerate at k=0 or k=n, unlike
    the Wald interval used previously."""
    if n <= 0:
        return (float('nan'), float('nan'))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))

def calibrate_cooperative_ed_threshold_orfusion(X_calib, Y_calib, target_pf):
    """(2) Per-node threshold calibration for TRUE OR-fusion: each node's
    threshold is calibrated independently so that, under H0, EACH node's own
    false-alarm rate is target_pf. (Because the cluster-level Pf under OR
    across 8 nodes is higher than any single node's Pf, this is the
    calibration convention consistent with the manuscript's stated per-node
    threshold lambda; the resulting CLUSTER Pf is measured empirically, not
    assumed.)"""
    node_energies = np.mean(X_calib ** 2, axis=(2, 3))  # (n_mc, n_nodes)
    h0_mask = (Y_calib.squeeze() == 0)
    h0_energies = node_energies[h0_mask]  # (n_h0, n_nodes)
    # One threshold per node, from that node's own H0 energy distribution.
    thresholds = np.percentile(h0_energies, (1 - target_pf) * 100, axis=0)  # (n_nodes,)
    return thresholds

def evaluate_cooperative_ed_orfusion(thresholds, X_eval, Y_eval, return_decisions: bool = False):
    """(2) TRUE OR-fusion: cluster declares H1 if ANY node's energy exceeds
    that node's own calibrated threshold."""
    node_energies = np.mean(X_eval ** 2, axis=(2, 3))          # (n_eval, n_nodes)
    node_decisions = (node_energies > thresholds[None, :])      # (n_eval, n_nodes)
    decisions = node_decisions.any(axis=1).astype(int)          # OR across nodes
    h0_mask = Y_eval.squeeze() == 0
    h1_mask = Y_eval.squeeze() == 1
    Pd = np.sum(decisions[h1_mask] == 1) / np.sum(h1_mask) if np.sum(h1_mask) > 0 else 0
    Pf = np.sum(decisions[h0_mask] == 1) / np.sum(h0_mask) if np.sum(h0_mask) > 0 else 0
    if return_decisions:
        return Pd, Pf, decisions
    return Pd, Pf

def n_h0_h1_trials(Y_eval) -> Tuple[int, int]:
    y = Y_eval.squeeze() if hasattr(Y_eval, 'squeeze') else np.asarray(Y_eval).squeeze()
    n_h0 = int(np.sum(np.asarray(y) == 0))
    n_h1 = int(np.sum(np.asarray(y) == 1))
    return n_h0, n_h1

# ==============================================================================
# 6. MODELS: main GAT sensor + FOUR ablation variants
# ==============================================================================
@dataclass
class GNNHyperparams:
    lr: float = 0.001
    weight_decay: float = 0.0
    gat_hidden: int = 64
    gat_heads: int = 4
    cnn_filters1: int = 32
    cnn_filters2: int = 32
    dropout: float = 0.3

def _make_local_cnn(hp: GNNHyperparams) -> nn.Sequential:
    """Per-node 2-layer 1D-CNN feature extractor, shared by all model variants."""
    return nn.Sequential(
        nn.Conv1d(2, hp.cnn_filters1, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.Conv1d(hp.cnn_filters1, hp.cnn_filters2, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )

class GNNSensor(torch.nn.Module):
    """Main sensor: CNN feature extractor + 2-layer GATConv (heads averaged)
    + global mean pool. Used both for GAT(BO) and, with default
    hyperparameters, for the GAT(no BO) ablation condition."""
    def __init__(self, seq_length, n_nodes=N_NODES, hp: Optional[GNNHyperparams] = None):
        super().__init__()
        hp = hp or GNNHyperparams()
        self.n_nodes = n_nodes
        self.local_cnn = _make_local_cnn(hp)
        self.gat1 = GATConv(hp.cnn_filters2, hp.gat_hidden, heads=hp.gat_heads, concat=False)
        self.gat2 = GATConv(hp.gat_hidden, hp.gat_hidden, heads=hp.gat_heads, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(hp.gat_hidden, 32), nn.ReLU(), nn.Dropout(hp.dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x_flat, edge_index, batch_vec):
        x = x_flat.permute(0, 2, 1)
        node_feats = self.local_cnn(x).squeeze(-1)
        x = F.elu(self.gat1(node_feats, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        graph_feats = global_mean_pool(x, batch_vec)
        return self.classifier(graph_feats)

class CNNDecisionFusionSensor(torch.nn.Module):
    """ABLATION (relabeled from 'CNN-only'): each node is classified
    independently from its own CNN embedding; per-node H1 probabilities are
    combined with a max() rule (decision-level fusion, the continuous analog
    of OR-fusion). NOTE: this condition still cooperates at the decision
    level across all 8 nodes -- it does NOT isolate "no cooperation" (see
    SingleNodeSensor below for that). It isolates the CNN feature
    extractor's contribution with no LEARNED (attention-based) cooperation,
    holding decision-level fusion constant relative to ED's own OR-fusion."""
    def __init__(self, seq_length, n_nodes=N_NODES, hp: Optional[GNNHyperparams] = None):
        super().__init__()
        hp = hp or GNNHyperparams()
        self.n_nodes = n_nodes
        self.local_cnn = _make_local_cnn(hp)
        self.node_classifier = nn.Sequential(
            nn.Linear(hp.cnn_filters2, 32), nn.ReLU(), nn.Dropout(hp.dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x_flat, edge_index, batch_vec):
        x = x_flat.permute(0, 2, 1)
        node_feats = self.local_cnn(x).squeeze(-1)
        node_probs = self.node_classifier(node_feats)
        node_probs = node_probs.view(-1, self.n_nodes)
        graph_prob, _ = node_probs.max(dim=1, keepdim=True)
        return graph_prob

class SingleNodeSensor(torch.nn.Module):
    """ABLATION (NEW): a genuine no-cooperation baseline. Uses ONLY node 0's
    CNN embedding; the other (n_nodes - 1) nodes' data is never read. There
    is no fusion of any kind, at either the feature or decision level. This
    is the condition that actually isolates "no cross-node cooperation,"
    which CNNDecisionFusionSensor does not (it fuses 8 nodes' decisions)."""
    def __init__(self, seq_length, n_nodes=N_NODES, hp: Optional[GNNHyperparams] = None):
        super().__init__()
        hp = hp or GNNHyperparams()
        self.n_nodes = n_nodes
        self.local_cnn = _make_local_cnn(hp)
        self.classifier = nn.Sequential(
            nn.Linear(hp.cnn_filters2, 32), nn.ReLU(), nn.Dropout(hp.dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x_flat, edge_index, batch_vec):
        # x_flat: (B * n_nodes, seq_len, 2), nodes ordered contiguously per graph.
        seq_len = x_flat.size(1)
        x = x_flat.view(-1, self.n_nodes, seq_len, 2)[:, 0, :, :]   # node 0 ONLY
        x = x.permute(0, 2, 1)
        node_feats = self.local_cnn(x).squeeze(-1)
        return self.classifier(node_feats)

class NonGraphFusionSensor(torch.nn.Module):
    """ABLATION: cooperative fusion WITHOUT graph attention. The same
    per-node CNN embeddings are mean-pooled across the cluster, then passed
    to a single shared classifier -- no GATConv, no attention."""
    def __init__(self, seq_length, n_nodes=N_NODES, hp: Optional[GNNHyperparams] = None):
        super().__init__()
        hp = hp or GNNHyperparams()
        self.n_nodes = n_nodes
        self.local_cnn = _make_local_cnn(hp)
        self.classifier = nn.Sequential(
            nn.Linear(hp.cnn_filters2, 32), nn.ReLU(), nn.Dropout(hp.dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x_flat, edge_index, batch_vec):
        x = x_flat.permute(0, 2, 1)
        node_feats = self.local_cnn(x).squeeze(-1)
        node_feats = node_feats.view(-1, self.n_nodes, node_feats.size(-1))
        pooled = node_feats.mean(dim=1)
        return self.classifier(pooled)

def create_base_edge_index(n_nodes):
    edges = [[u, v] for u in range(n_nodes) for v in range(n_nodes) if u != v]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

_BATCHED_GRAPH_CACHE: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_batched_graph_meta(n_nodes, B, base_edge_index):
    key = (n_nodes, B)
    if key not in _BATCHED_GRAPH_CACHE:
        edge_indices = [base_edge_index + (b * n_nodes) for b in range(B)]
        batch_edge_index = torch.cat(edge_indices, dim=1).to(DEVICE)
        batch_vec = torch.arange(B).repeat_interleave(n_nodes).to(DEVICE)
        _BATCHED_GRAPH_CACHE[key] = (batch_edge_index, batch_vec)
    return _BATCHED_GRAPH_CACHE[key]

def _build_batched_graph(batch_X, n_nodes, base_edge_index):
    B = batch_X.size(0)
    seq_len = batch_X.size(2)
    batch_X_flat = batch_X.reshape(B * n_nodes, seq_len, 2).to(DEVICE)
    batch_edge_index, batch_vec = _get_batched_graph_meta(n_nodes, B, base_edge_index)
    return batch_X_flat, batch_edge_index, batch_vec

def train_sensor_model(model_builder, hp: Optional[GNNHyperparams] = None,
                        X_train: Optional[torch.Tensor] = None,
                        Y_train: Optional[torch.Tensor] = None,
                        epochs: Optional[int] = None, verbose: bool = True):
    """(3b) Generalized training loop. As of this version, X_train/Y_train
    are ALWAYS supplied by the caller (generated once per seed/SNR and
    shared across every learned condition), rather than generated
    internally per-condition as in earlier versions -- this removes
    training-data variation as a confound between architectures."""
    hp = hp or GNNHyperparams()
    epochs = EPOCHS if epochs is None else epochs
    assert X_train is not None and Y_train is not None, \
        "X_train/Y_train must be provided (shared training data design)."

    model = model_builder(SEQUENCE_LENGTH, N_NODES, hp).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)

    base_edge_index = create_base_edge_index(N_NODES)
    model.train()
    prev_losses = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        perm = torch.randperm(X_train.size(0))
        X_shuffled, Y_shuffled = X_train[perm], Y_train[perm]
        epoch_loss = 0

        for i in range(0, len(X_shuffled), BATCH_SIZE):
            batch_X = X_shuffled[i:i + BATCH_SIZE]
            batch_Y = Y_shuffled[i:i + BATCH_SIZE].to(DEVICE)
            batch_X_flat, batch_edge_index, batch_vec = _build_batched_graph(
                batch_X, N_NODES, base_edge_index)

            optimizer.zero_grad()
            preds = model(batch_X_flat, batch_edge_index, batch_vec)
            loss = criterion(preds, batch_Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(X_shuffled)
        scheduler.step(avg_epoch_loss)

        if verbose:
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        if epoch >= (EARLY_STOPPING_PATIENCE - 1):
            prev_losses.append(avg_epoch_loss)
            if len(prev_losses) > EARLY_STOPPING_PATIENCE:
                prev_losses.pop(0)
            if len(prev_losses) == EARLY_STOPPING_PATIENCE:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        if verbose:
                            print(f"      [INFO] Early stopping at epoch {epoch+1}")
                        break

    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    return model

def calibrate_sensor_threshold(model, X_calib, Y_calib, target_pf):
    model.eval()
    base_edge_index = create_base_edge_index(N_NODES)
    X_flat, batch_edge_index, batch_vec = _build_batched_graph(X_calib, N_NODES, base_edge_index)
    with torch.no_grad():
        preds = model(X_flat, batch_edge_index, batch_vec).cpu().squeeze()
    h0_preds = preds[Y_calib.squeeze() == 0]
    sorted_h0, _ = torch.sort(h0_preds)
    idx = int(len(sorted_h0) * (1.0 - target_pf))
    return sorted_h0[min(idx, len(sorted_h0) - 1)].item()

def evaluate_sensor_model(model, threshold, X_eval, Y_eval, return_decisions: bool = False):
    model.eval()
    base_edge_index = create_base_edge_index(N_NODES)
    X_flat, batch_edge_index, batch_vec = _build_batched_graph(X_eval, N_NODES, base_edge_index)
    with torch.no_grad():
        preds = model(X_flat, batch_edge_index, batch_vec).cpu().squeeze()
    h0_mask = Y_eval.squeeze() == 0
    h1_mask = Y_eval.squeeze() == 1
    decisions = (preds > threshold).float()
    Pd = (decisions[h1_mask] == 1).sum().item() / h1_mask.sum().item() if h1_mask.sum() > 0 else 0
    Pf = (decisions[h0_mask] == 1).sum().item() / h0_mask.sum().item() if h0_mask.sum() > 0 else 0
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    if return_decisions:
        return Pd, Pf, decisions.numpy().astype(int)
    return Pd, Pf

# ==============================================================================
# 7. FLOPS PROFILING -- unchanged from v5.0
# ==============================================================================
def profile_gnn_sensor_flops(model, seq_length, n_nodes=N_NODES) -> Optional[float]:
    if not HAS_THOP:
        return None
    model_cpu = model.to('cpu')
    model_cpu.eval()
    dummy_x = torch.randn(n_nodes, seq_length, 2)
    edge_index = create_base_edge_index(n_nodes)
    batch_vec = torch.zeros(n_nodes, dtype=torch.long)
    macs, _params = thop_profile(model_cpu, inputs=(dummy_x, edge_index, batch_vec), verbose=False)
    model.to(DEVICE)
    return float(macs) * 2.0

def required_flops_per_sec(flops_per_inference: float, sensing_interval_s: float) -> float:
    return flops_per_inference / sensing_interval_s

@dataclass
class HardwareComputeSpec:
    name: str
    peak_flops_per_sec: float
    precision_note: str
    source_note: str

# (6) Kept as a FLOPs-only SCREENING comparison, explicitly across mixed
# precisions -- see the "no apparent FLOP-budget barrier" framing in the
# manuscript. Not a substitute for latency/memory/energy measurement.
HARDWARE_TIERS_FLOPS: Dict[str, HardwareComputeSpec] = {
    'Tier_1_Jetson_Nano': HardwareComputeSpec(
        'Jetson Nano', 472e9, 'FP16',
        'NVIDIA Jetson Nano module spec: 472 GFLOPS (FP16), 128-core Maxwell GPU.'),
    'Tier_2_Jetson_Orin_NX': HardwareComputeSpec(
        'Jetson Orin NX 16GB', 3.76e12, 'FP16',
        'NVIDIA DS-10712-001 datasheet: 3.76 TFLOPS FP16, GPU only, standard clock.'),
    'Tier_3_RTX_A2000': HardwareComputeSpec(
        'RTX A2000', 8.0e12, 'FP32',
        'NVIDIA RTX A2000 product spec: ~8.0 TFLOPS peak FP32.'),
}

def check_hardware_feasibility(flops_per_inference: float, sensing_interval_s: float,
                                margin: float = HW_UTILIZATION_MARGIN,
                                tiers: Dict[str, HardwareComputeSpec] = HARDWARE_TIERS_FLOPS
                                ) -> Dict[str, dict]:
    required = required_flops_per_sec(flops_per_inference, sensing_interval_s)
    results = {}
    print(f"\n[Hardware screening -- FLOPs only, mixed precisions, lower-bound numerator] "
          f"Required: {required/1e6:.2f} MFLOPs/sec")
    for tier_id, spec in tiers.items():
        budget = spec.peak_flops_per_sec * margin
        feasible = required <= budget
        results[tier_id] = {
            'name': spec.name, 'precision': spec.precision_note,
            'peak_flops_per_sec': spec.peak_flops_per_sec,
            'derated_budget_flops_per_sec': budget,
            'required_flops_per_sec': required,
            'no_flop_budget_barrier': feasible,
        }
        print(f"  {spec.name:<22} [{spec.precision_note}] peak={spec.peak_flops_per_sec/1e9:9.2f} GFLOPS/s  "
              f"derated({margin:.0%})={budget/1e9:9.2f} GFLOPS/s  "
              f"{'no apparent FLOP-budget barrier' if feasible else 'FLOP budget insufficient'}")
    print("  NOTE: lower-bound FLOPs numerator compared against PEAK vendor figures at "
          "MIXED numerical precisions (FP16 for both Jetsons, FP32 for the RTX A2000). "
          "This supports only the claim that there is no apparent FLOP-budget barrier; "
          "it is not a verified statement of edge-hardware feasibility, which requires "
          "direct latency, memory, and energy measurement on target hardware.")
    return results

# ==============================================================================
# 8. BAYESIAN OPTIMIZATION OF GNN HYPERPARAMETERS -- unchanged from v5.0,
#    run ONCE (independent of the experiment seeds in Section 9)
# ==============================================================================
HPO_SEARCH_SPACE = [
    {'name': 'lr',            'type': 'log_uniform', 'low': 1e-4, 'high': 5e-3},
    {'name': 'weight_decay',  'type': 'log_uniform', 'low': 1e-6, 'high': 1e-3},
    {'name': 'gat_hidden',    'type': 'categorical',  'values': [32, 64, 128]},
    {'name': 'gat_heads',     'type': 'categorical',  'values': [2, 4, 8]},
    {'name': 'cnn_filters1',  'type': 'categorical',  'values': [16, 32, 64]},
    {'name': 'cnn_filters2',  'type': 'categorical',  'values': [32, 64, 128]},
    {'name': 'dropout',       'type': 'uniform',      'low': 0.1, 'high': 0.5},
]

def _encode_params(params: dict) -> np.ndarray:
    vec = []
    for spec in HPO_SEARCH_SPACE:
        val = params[spec['name']]
        if spec['type'] == 'log_uniform':
            lo, hi = np.log(spec['low']), np.log(spec['high'])
            vec.append((np.log(val) - lo) / (hi - lo))
        elif spec['type'] == 'uniform':
            lo, hi = spec['low'], spec['high']
            vec.append((val - lo) / (hi - lo))
        else:
            values = spec['values']
            idx = values.index(val)
            vec.append(idx / (len(values) - 1) if len(values) > 1 else 0.0)
    return np.array(vec, dtype=np.float64)

def _decode_params(vec: np.ndarray) -> dict:
    params = {}
    for x, spec in zip(vec, HPO_SEARCH_SPACE):
        x = float(np.clip(x, 0.0, 1.0))
        if spec['type'] == 'log_uniform':
            lo, hi = np.log(spec['low']), np.log(spec['high'])
            params[spec['name']] = float(np.exp(lo + x * (hi - lo)))
        elif spec['type'] == 'uniform':
            lo, hi = spec['low'], spec['high']
            params[spec['name']] = float(lo + x * (hi - lo))
        else:
            values = spec['values']
            idx = int(round(x * (len(values) - 1)))
            params[spec['name']] = values[idx]
    return params

def evaluate_gnn_hp_objective(hp: GNNHyperparams, avg_channel_loss_db: float) -> float:
    scores = []
    for snr in BO_REPRESENTATIVE_SNRS:
        X_train, Y_train = generate_cooperative_sensing_dataset(
            snr, BO_N_MC, SENSING_SAMPLES, SEQUENCE_LENGTH, avg_channel_loss_db,
            n_windows=TRAIN_WINDOWS_PER_TRIAL)
        model = train_sensor_model(lambda sl, n, h: GNNSensor(sl, n, h), hp=hp,
                                    X_train=X_train, Y_train=Y_train,
                                    epochs=BO_EPOCHS, verbose=False)
        X_calib, Y_calib = generate_cooperative_sensing_dataset(
            snr, BO_N_MC_CALIB, SENSING_SAMPLES, SEQUENCE_LENGTH, avg_channel_loss_db)
        X_eval, Y_eval = generate_cooperative_sensing_dataset(
            snr, BO_N_MC, SENSING_SAMPLES, SEQUENCE_LENGTH, avg_channel_loss_db)
        thr = calibrate_sensor_threshold(model, X_calib, Y_calib, TARGET_PF)
        pd, pf = evaluate_sensor_model(model, thr, X_eval, Y_eval)
        scores.append(pd - BO_PF_PENALTY_LAMBDA * max(0.0, pf - TARGET_PF))
        del model, X_train, Y_train, X_calib, Y_calib, X_eval, Y_eval
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    return float(np.mean(scores))

def _expected_improvement(X_candidates: np.ndarray, gp: "GaussianProcessRegressor",
                           f_best: float, xi: float = 0.01) -> np.ndarray:
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * scipy_norm.cdf(Z) + sigma * scipy_norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei

def bayesian_optimize_gnn_hyperparams(avg_channel_loss_db: float, n_initial: int = BO_N_INITIAL,
                                       n_iters: int = BO_N_ITERS, seed: int = BO_SEED):
    if not HAS_BO_DEPS:
        print("[BO] scikit-learn/scipy unavailable -- using default GNNHyperparams().")
        return GNNHyperparams(), None, []

    rng = np.random.RandomState(seed)
    X_obs: List[np.ndarray] = []
    y_obs: List[float] = []
    history: List[Tuple[dict, float]] = []

    print(f"\n[BO] Starting GNN hyperparameter search (run once, seed={seed}, "
          f"independent of the {len(EXPERIMENT_SEEDS)} experiment seeds): "
          f"{n_initial} random + {max(0, n_iters - n_initial)} GP-guided trials")

    for i in range(n_initial):
        x = rng.uniform(0, 1, size=len(HPO_SEARCH_SPACE))
        params = _decode_params(x)
        score = evaluate_gnn_hp_objective(GNNHyperparams(**params), avg_channel_loss_db)
        X_obs.append(x); y_obs.append(score); history.append((params, score))
        print(f"  [BO random {i+1}/{n_initial}] score={score:.4f} params={params}")

    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
        noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                   random_state=seed)

    for i in range(max(0, n_iters - n_initial)):
        gp.fit(np.array(X_obs), np.array(y_obs))
        f_best = max(y_obs)
        candidates = rng.uniform(0, 1, size=(2000, len(HPO_SEARCH_SPACE)))
        ei = _expected_improvement(candidates, gp, f_best)
        x_next = candidates[int(np.argmax(ei))]
        params = _decode_params(x_next)
        score = evaluate_gnn_hp_objective(GNNHyperparams(**params), avg_channel_loss_db)
        X_obs.append(x_next); y_obs.append(score); history.append((params, score))
        print(f"  [BO guided {i+1}/{n_iters-n_initial}] score={score:.4f} "
              f"(best so far={max(y_obs):.4f}) params={params}")

    best_idx = int(np.argmax(y_obs))
    best_params, best_score = history[best_idx]
    best_random_score = max(s for _, s in history[:n_initial])
    print(f"[BO] Done. Best score={best_score:.4f} (best random-init score={best_random_score:.4f}), "
          f"params={best_params}")
    return GNNHyperparams(**best_params), best_score, history

def plot_bo_convergence(history: List[Tuple[dict, float]], filename: str = 'gnn_bo_convergence.png'):
    if not history:
        return
    scores = [s for _, s in history]
    running_best = np.maximum.accumulate(scores)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(scores) + 1), scores, 'o', color='tab:blue', alpha=0.5, label='Trial score')
    ax.plot(range(1, len(scores) + 1), running_best, '-', color='tab:red', linewidth=2,
             label='Best score so far')
    ax.axvline(BO_N_INITIAL + 0.5, color='k', linestyle='--', alpha=0.4,
               label='Random \u2192 GP-guided')
    ax.set_title('GNN Hyperparameter Bayesian Optimization Convergence')
    ax.set_xlabel('Trial'); ax.set_ylabel(f'Objective (Pd \u2212 {BO_PF_PENALTY_LAMBDA}\u00b7max(0, Pf\u2212target))')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"[BO] Convergence plot saved to '{filename}'")
    plt.show()

# ==============================================================================
# 9. MAIN DRIVER -- multi-seed, shared training data, matched hyperparameters
# ==============================================================================
# Learned conditions and the (fixed) hyperparameters each uses. Only GAT_BO
# gets the Bayesian-optimized configuration; every other learned condition
# uses the SAME untuned default, per item (3c) above.
LEARNED_MODEL_BUILDERS = {
    'GAT_BO':            lambda sl, n, h: GNNSensor(sl, n, h),
    'GAT_noBO':          lambda sl, n, h: GNNSensor(sl, n, h),
    'CNN_DecisionFusion': lambda sl, n, h: CNNDecisionFusionSensor(sl, n, h),
    'SingleNode':        lambda sl, n, h: SingleNodeSensor(sl, n, h),
    'NonGraphFusion':    lambda sl, n, h: NonGraphFusionSensor(sl, n, h),
}
ALL_CONDITIONS = ['ED'] + list(LEARNED_MODEL_BUILDERS.keys())

def run_sensing_paper_experiment():
    # --- BO runs ONCE, before the seed loop, independent of experiment seeds ---
    set_global_seed(BO_SEED)
    avg_channel_loss_db = _compute_avg_channel_loss_db()
    print(f"[Path-loss model] Avg channel loss: {avg_channel_loss_db:.2f} dB "
          f"(two-slope breakpoint model; d=[{D_MIN_KM},{D_MAX_KM}] km, veg=[0,{VEG_MAX_DEPTH_M}] m)")

    print("\n[HPO] Bayesian-optimizing GAT sensor hyperparameters (run once)...")
    best_gnn_hp, best_bo_score, bo_history = bayesian_optimize_gnn_hyperparams(avg_channel_loss_db)
    plot_bo_convergence(bo_history)
    default_hp = GNNHyperparams()  # shared, untuned, matched hyperparameters (item 3c)

    print(f"\nStarting multi-seed sensing + ablation experiment | "
          f"Seeds: {EXPERIMENT_SEEDS} | MC Trials: {N_MC} | SNR points: {len(SNR_RANGE)}")
    print("=" * 70)

    # results[seed][condition]['pd'|'pf'] -> list over SNR_RANGE
    all_seed_results: Dict[int, Dict[str, Dict[str, list]]] = {}
    paired_decisions: Dict[str, np.ndarray] = {}  # "seed_{s}_snr_{snr}_{cond}" -> array
    last_gnn_model = None

    for seed in EXPERIMENT_SEEDS:
        print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
        set_global_seed(seed)  # (3d) governs THIS seed's data generation + model init

        results = {c: {'pd': [], 'pf': [], 'pd_ci': [], 'pf_ci': []} for c in ALL_CONDITIONS}

        for snr in SNR_RANGE:
            print(f"  SNR: {snr} dB")
            # --- shared calib/eval sets: identical across ED + all 5 learned conditions ---
            X_calib, Y_calib = generate_cooperative_sensing_dataset(
                snr, N_MC_CALIB, SENSING_SAMPLES, SEQUENCE_LENGTH, avg_channel_loss_db)
            X_eval, Y_eval = generate_cooperative_sensing_dataset(
                snr, N_MC, SENSING_SAMPLES, SEQUENCE_LENGTH, avg_channel_loss_db)
            # --- (3b) shared TRAINING set: identical across all 5 learned conditions ---
            X_train, Y_train = generate_cooperative_sensing_dataset(
                snr, N_MC, SENSING_SAMPLES, SEQUENCE_LENGTH, avg_channel_loss_db,
                n_windows=TRAIN_WINDOWS_PER_TRIAL)

            n_h0, n_h1 = n_h0_h1_trials(Y_eval.numpy())
            paired_decisions[f"seed_{seed}_snr_{snr}_Y"] = Y_eval.numpy().squeeze().astype(int)

            # --- ED baseline: TRUE OR-fusion (item 2) ---
            thr_ed = calibrate_cooperative_ed_threshold_orfusion(X_calib.numpy(), Y_calib.numpy(), TARGET_PF)
            pd, pf, dec = evaluate_cooperative_ed_orfusion(thr_ed, X_eval.numpy(), Y_eval.numpy(), return_decisions=True)
            results['ED']['pd'].append(pd); results['ED']['pf'].append(pf)
            results['ED']['pd_ci'].append(wilson_ci_95(int(round(pd * n_h1)), n_h1))
            results['ED']['pf_ci'].append(wilson_ci_95(int(round(pf * n_h0)), n_h0))
            paired_decisions[f"seed_{seed}_snr_{snr}_ED"] = dec
            print(f"    ED (OR-fusion)      -> Pd: {pd:.4f}  Pf: {pf:.4f}")

            # --- 5 learned conditions, shared training data, matched hyperparameters ---
            for cond, builder in LEARNED_MODEL_BUILDERS.items():
                hp = best_gnn_hp if cond == 'GAT_BO' else default_hp
                model = train_sensor_model(builder, hp=hp, X_train=X_train, Y_train=Y_train, verbose=False)
                thr = calibrate_sensor_threshold(model, X_calib, Y_calib, TARGET_PF)
                pd, pf, dec = evaluate_sensor_model(model, thr, X_eval, Y_eval, return_decisions=True)
                results[cond]['pd'].append(pd); results[cond]['pf'].append(pf)
                results[cond]['pd_ci'].append(wilson_ci_95(int(round(pd * n_h1)), n_h1))
                results[cond]['pf_ci'].append(wilson_ci_95(int(round(pf * n_h0)), n_h0))
                paired_decisions[f"seed_{seed}_snr_{snr}_{cond}"] = dec
                print(f"    {cond:<18} -> Pd: {pd:.4f}  Pf: {pf:.4f}")
                if cond == 'GAT_BO':
                    last_gnn_model = model

            del X_calib, Y_calib, X_eval, Y_eval, X_train, Y_train
            gc.collect()

        all_seed_results[seed] = results

    # --- Save paired decisions (all seeds, all SNRs, all conditions) ---
    np.savez_compressed('paired_decisions_multiseed.npz', **paired_decisions)
    print("\n[Reproducibility] Per-seed paired per-trial decisions saved to "
          "'paired_decisions_multiseed.npz' -- use for exact per-seed or seed-pooled "
          "McNemar's/binomial tests.")

    # --- FLOPs profiling + hardware screening (architecture/hp-dependent only, not seed-dependent) ---
    gnn_flops_per_inference = None
    if last_gnn_model is not None:
        gnn_flops_per_inference = profile_gnn_sensor_flops(last_gnn_model, SEQUENCE_LENGTH, N_NODES)
        if gnn_flops_per_inference is not None:
            print(f"\n[FLOPs] GAT sensor (BO-tuned: {best_gnn_hp}): "
                  f"~{gnn_flops_per_inference/1e6:.2f} MFLOPs/inference (lower bound)")
            check_hardware_feasibility(gnn_flops_per_inference, SENSING_INTERVAL_S)

    # --- Export per-seed + mean-across-seed CSV ---
    import csv
    with open('ml_crn_multiseed_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['seed', 'snr_db']
        for cond in ALL_CONDITIONS:
            header += [f'{cond}_pd', f'{cond}_pd_ci_lo', f'{cond}_pd_ci_hi',
                       f'{cond}_pf', f'{cond}_pf_ci_lo', f'{cond}_pf_ci_hi']
        writer.writerow(header)
        for seed in EXPERIMENT_SEEDS:
            res = all_seed_results[seed]
            for i, snr in enumerate(SNR_RANGE):
                row = [seed, snr]
                for cond in ALL_CONDITIONS:
                    row += [res[cond]['pd'][i], res[cond]['pd_ci'][i][0], res[cond]['pd_ci'][i][1],
                            res[cond]['pf'][i], res[cond]['pf_ci'][i][0], res[cond]['pf_ci'][i][1]]
                writer.writerow(row)
    print("[Export] Per-seed results (all conditions, all SNRs, all seeds) -> "
          "'ml_crn_multiseed_results.csv'")

    # Mean +/- std across seeds, for a quick summary table
    with open('ml_crn_multiseed_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['snr_db']
        for cond in ALL_CONDITIONS:
            header += [f'{cond}_pd_mean', f'{cond}_pd_std', f'{cond}_pf_mean', f'{cond}_pf_std']
        writer.writerow(header)
        for i, snr in enumerate(SNR_RANGE):
            row = [snr]
            for cond in ALL_CONDITIONS:
                pds = np.array([all_seed_results[s][cond]['pd'][i] for s in EXPERIMENT_SEEDS])
                pfs = np.array([all_seed_results[s][cond]['pf'][i] for s in EXPERIMENT_SEEDS])
                row += [pds.mean(), pds.std(ddof=1), pfs.mean(), pfs.std(ddof=1)]
            writer.writerow(row)
    print("[Export] Cross-seed mean +/- std summary -> 'ml_crn_multiseed_summary.csv'")

    print("\n" + "=" * 70)
    print("Multi-seed sensing + ablation experiment complete.")
    print("=" * 70)
    return all_seed_results, paired_decisions, gnn_flops_per_inference, best_gnn_hp

if __name__ == "__main__":
    run_sensing_paper_experiment()
