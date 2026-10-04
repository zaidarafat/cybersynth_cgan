"""
Cyber-Synth: leakage-free evaluation of label-conditioned GAN augmentation for
rare-attack detection (NSL-KDD, CIC-IDS-2017, CSE-CIC-IDS-2018).

What one run does (per dataset and seed):
  1. load + clean the data once; split train/test (stratified) FIRST;
  2. fit all preprocessing on the training partition only;
  3. for every augmentation method, generate synthetic records for the minority
     classes, add them to the training partition only, train classifiers, and
     evaluate on the real test partition;
  4. write one JSON file per method with per-class metrics, fidelity, diversity,
     label consistency, memorisation and timings.

Augmentation methods (paper Section 3.6):
  none         no augmentation (real data only)
  ros          random oversampling
  smote        SMOTE, k = 5
  gan_pseudo   A0  unconditional GAN, labels assigned by a real-trained RF
  cgan_gonly   A1  label given to the generator only
  cgan_emp     A2  label to G and D, labels sampled at their empirical frequency
  cgan         A3  label to G and D, class-balanced label sampling (main model)
  wgan_gp      A4  A3 with the Wasserstein-GP objective
  acgan        A5  A3 with an auxiliary-classifier discriminator
  cgan_ordinal A6  A3 with ordinal (instead of one-hot) categorical encoding
  cgan_nolog   A6b A3 without the log transform of heavy-tailed features
  cgan_nozero  A6c A3 without zero/non-zero switches for zero-inflated features
  wgan_gp_nozero  A4 without zero/non-zero switches
  cgan_shallow A7  A3 with one hidden layer per network
  cgan_deep    A7  A3 with six hidden layers per network
  ctgan        CTGAN reference implementation (pip install ctgan)

Summarise results with:  python summarize_results.py results/cic17
"""
import argparse, copy, glob, json, os, platform, random, sys, time

# =====================================================================
#  SETTINGS FOR THE VS CODE "RUN" BUTTON
#  Edit these lines, save, and press Run. (Command-line arguments, if
#  given, override them.)
# =====================================================================
DATASET = "nsl"            # "nsl", "cic17" or "cic18"
DATA_PATHS = {
    "nsl":   r"E:\Zaid PHD\old work\Datasets\KDDTrain+.txt",       # the header-less NSL-KDD training file
    "cic17": r"E:\Zaid PHD\old work\Datasets\CIC-IDS-2017\*.csv",          # folder with the 8 CIC-IDS-2017 CSV files
    "cic18": r"E:\Zaid PHD\Cherepovets\old work\Datasets\CSE-CIC-IDS-2018\*.csv",      # folder with the 10 CSE-CIC-IDS-2018 CSV files
}
QUICK_TEST = False         # True: one short run (~10-20 min) to check that everything works
                           # False: the full experiment
SEEDS = range(10)          # NSL-KDD: all 10 seeds. CIC: range(1, 10) = seeds 1-9 (main); range(1, 6) = seeds 1-5 (ablation)
METHOD_SET = "all"         # "main" (baselines + CGAN + WGAN-GP), "ablation" (A1-A7 variants) or "all"
METHOD_SETS = {
    "main":     ["none", "ros", "smote", "ctgan", "gan_pseudo", "cgan", "wgan_gp"],
    "ablation": ["cgan_gonly", "cgan_emp", "acgan", "cgan_ordinal", "cgan_nolog", "cgan_nozero",
                 "wgan_gp_nozero", "cgan_shallow", "cgan_deep"],
}
METHOD_SETS["all"] = METHOD_SETS["main"] + METHOD_SETS["ablation"]
# Extra classifiers (XGBoost, LightGBM, DNN) are run only for these methods,
# to keep the total run time reasonable; the random forest runs for all methods.
EXTRA_CLF_METHODS = ["none", "smote", "gan_pseudo", "ctgan", "wgan_gp", "cgan"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# =====================================================================

# Per-dataset defaults (used by the Run button; documented in the paper).
DATASET_DEFAULTS = {
    "nsl":   ["--nslkdd", "--minority-classes", "Probe,R2L,U2R", "--target-ratio", "1.0"],
    "cic17": ["--label-col", "Label", "--group-families", "--exclude-classes", "Heartbleed",
              "--minority-classes", "Brute Force,Web Attack,Bot,Infiltration",
              "--max-train-per-class", "300000", "--target-ratio", "0.1"],
    "cic18": ["--label-col", "Label", "--group-families",
              "--minority-classes", "Brute Force,Bot,Infiltration,Web Attack",
              "--benign-frac", "0.15",
              "--max-train-per-class", "300000", "--target-ratio", "0.1"],
}

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

# joblib/loky probes the physical core count with `wmic`, which recent Windows
# builds no longer ship, and prints a warning plus traceback when it fails.
# Pre-filling its cache with the logical count (what it falls back to anyway)
# skips the probe.
try:
    from joblib.externals.loky.backend import context as _loky_context
    if _loky_context.physical_cores_cache is None:
        _loky_context.physical_cores_cache = os.cpu_count() or 1
except ImportError:
    pass

ID_COLS = ["Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
           "Src Port", "Source Port", "Timestamp"]
VERSION = "v3"             # v3: + zero/non-zero switches; v2: ports, rounding, coverage selection
LOAD_SEED = 12345          # fixed: every seed sees the same loaded dataset


# ----------------------------------------------------------------- utilities
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


NSL_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"]
NSL_MAP = {
    "normal": "Normal",
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
                          "mailbomb", "processtable", "udpstorm"]},
    **{a: "Probe" for a in ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]},
    **{a: "R2L" for a in ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
                          "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
                          "snmpguess", "xlock", "xsnoop", "worm"]},
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel",
                          "ps", "sqlattack", "xterm"]},
}


def cic_family(label: str) -> str:
    """Map fine-grained CIC-IDS-2017 / CSE-CIC-IDS-2018 labels to attack families."""
    l = str(label).lower()
    if "benign" in l:
        return "Benign"
    if "web" in l or "xss" in l or "sql" in l:
        return "Web Attack"
    if l.startswith("ddos"):
        return "DDoS"
    if l.startswith("dos"):
        return "DoS"
    if "portscan" in l:
        return "PortScan"
    if "patator" in l or "bruteforce" in l or "brute force" in l:
        return "Brute Force"
    if "bot" in l:
        return "Bot"
    if "infilt" in l:
        return "Infiltration"
    if "heartbleed" in l:
        return "Heartbleed"
    return str(label)


# ---------------------------------------------------------------- data loading
def _clean_chunk(df, label_col, benign_frac, rng):
    df.columns = df.columns.str.strip()
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not found; columns: {list(df.columns)[:8]}...")
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    lab = df[label_col].astype(str).str.strip()
    # CIC-IDS-2017 web-attack labels carry a broken dash (a stored U+FFFD
    # replacement char, or a stray cp1252 en dash); normalise to " - ".
    lab = lab.str.replace(r"\s*(?:\ufffd|ï¿½|–|\x96)\s*", " - ", regex=True)
    df[label_col] = lab
    df = df[lab != label_col]                                  # repeated header rows
    if benign_frac < 1.0:                                      # fixed-seed benign subsample
        is_b = df[label_col].str.lower().str.contains("benign")
        keep = ~is_b.to_numpy() | (rng.random(len(df)) < benign_frac)
        df = df[keep]
    df = df.copy()
    feat = df.columns.drop(label_col)
    for c in feat:
        if not pd.api.types.is_numeric_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.isna().mean() < 0.5:                       # numeric stored as text
                df[c] = conv
    num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
    df[num] = df[num].astype(np.float32)
    return df


def _read_nslkdd(path):
    """Reads an NSL-KDD file in any of its common variants: with or without a
    header row, with or without the difficulty column, labels with a trailing
    '.' (KDD'99 style), or labels already grouped into the five categories."""
    raw = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    first = str(raw.iloc[0, 0]).strip()
    try:
        float(first)
    except ValueError:                       # first row is a header (e.g. "duration")
        raw = raw.iloc[1:].reset_index(drop=True)
    ncol = raw.shape[1]
    if ncol == 43:
        raw.columns = NSL_COLS
        raw = raw.drop(columns=["difficulty"])
    elif ncol == 42:
        raw.columns = NSL_COLS[:-1]
    else:
        raise ValueError(f"{os.path.basename(path)}: expected 42 or 43 columns (41 features + "
                         f"label [+ difficulty]), found {ncol}")
    lab = raw["label"].astype(str).str.strip().str.rstrip(".").str.lower()
    grouped = {"normal": "Normal", "dos": "DoS", "probe": "Probe", "r2l": "R2L", "u2r": "U2R"}
    mapping = {**{k.lower(): v for k, v in NSL_MAP.items()}, **grouped}
    unknown = sorted(set(lab) - set(mapping))
    if unknown:
        raise ValueError(f"unmapped NSL-KDD labels: {unknown}")
    raw["label"] = lab.map(mapping)
    return raw


def load_and_clean(pattern, label_col, nslkdd=False, benign_frac=1.0):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match {pattern!r}")
    rng = np.random.default_rng(LOAD_SEED)
    if nslkdd:   # KDDTrain+: 41 features + label (+ difficulty); header row optional
        df = pd.concat([_read_nslkdd(f) for f in files], ignore_index=True)
        df = _clean_chunk(df, "label", 1.0, rng)
    else:        # CIC files are read in chunks so CSE-CIC-IDS-2018 fits in memory
        parts = []
        for f in files:
            log(f"reading {os.path.basename(f)}")
            for chunk in pd.read_csv(f, low_memory=False, encoding="utf-8",
                                     encoding_errors="replace", chunksize=500_000):
                parts.append(_clean_chunk(chunk, label_col, benign_frac, rng))
        df = pd.concat(parts, ignore_index=True)
        del parts
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().drop_duplicates()
    return df.reset_index(drop=True)


# -------------------------------------------------------------- preprocessing
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def signed_exp(x):
    return np.sign(x) * np.expm1(np.abs(x))


def is_port_col(name: str) -> bool:
    n = name.lower().replace("_", " ")
    return "port" in n and "portscan" not in n


class Preprocessor:
    """Maps a raw feature frame to [-1, 1] numerical features followed by one-hot
    categorical blocks (or ordinal codes scaled with the numerical features).
    Fitted on the training partition only; invertible, so synthetic records from
    any generator can be mapped back to raw features and into a common space.

    Generator space only (port_as_category=True):
      * port columns become categories: the 50 most frequent training ports plus
        the 5 most frequent ports of every class; all others map to "other",
        which is inverted by drawing from the training ports in that bucket;
      * zero_flags=True: every numerical feature that is exactly 0 in at least 2% of
        training records gets a zero/non-zero switch (a two-way categorical head in
        the generator); records generated with the switch at "zero" get exactly 0;
    Every space: numerical columns that hold only whole numbers in the training
    data are rounded when synthetic records are mapped back (inverse_transform)."""

    def __init__(self, log_transform=True, categorical="onehot", port_as_category=False,
                 top_ports=50, top_ports_per_class=5, zero_flags=False, zero_min_frac=0.02):
        self.log_transform, self.categorical = log_transform, categorical
        self.zero_flags, self.zero_min_frac = zero_flags, zero_min_frac
        self.port_as_category = port_as_category
        self.top_ports, self.top_ports_per_class = top_ports, top_ports_per_class

    # -- port handling (generator space) --
    def _ports_to_cat(self, X):
        if not self.ports:
            return X
        X = X.copy()
        for c in self.ports:
            v = X[c].to_numpy()
            X[c] = np.where(np.isin(v, self.port_keep[c]),
                            "p" + pd.Series(v.astype(np.int64)).astype(str).to_numpy(), "other")
        return X

    # -- zero-inflated features (generator space) --
    def _add_zero_flags(self, X):
        if not self.zero_cols:
            return X
        X = X.copy()
        for c in self.zero_cols:
            X[c + "__is_zero"] = np.where(X[c].to_numpy() == 0, "zero", "nonzero")
        return X

    def fit(self, X: pd.DataFrame, y=None, seed=0):
        self.rng = np.random.default_rng(seed)
        self.ports = [c for c in X.columns if self.port_as_category and is_port_col(c)
                      and pd.api.types.is_numeric_dtype(X[c])]
        self.port_keep, self.port_other = {}, {}
        for c in self.ports:
            v = X[c].to_numpy().astype(np.int64)
            keep = set(pd.Series(v).value_counts().index[:self.top_ports])
            if y is not None:
                for k in np.unique(y):
                    keep |= set(pd.Series(v[y == k]).value_counts().index[:self.top_ports_per_class])
            self.port_keep[c] = np.array(sorted(keep))
            other = v[~np.isin(v, self.port_keep[c])]
            self.port_other[c] = other if len(other) else np.array([0])
        X = self._ports_to_cat(X)
        self.zero_cols = []
        if self.zero_flags:                                    # features with a spike at exactly 0
            for c in X.columns:
                if pd.api.types.is_numeric_dtype(X[c]):
                    f0 = float((X[c].to_numpy() == 0).mean())
                    if self.zero_min_frac <= f0 < 1.0:
                        self.zero_cols.append(c)
        X = self._add_zero_flags(X)
        self.cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        num = [c for c in X.columns if c not in self.cat]
        std = X[num].astype(np.float64).std().to_numpy()
        self.num = [c for c, s in zip(num, std) if s > 0]       # drop constant columns
        self.dropped_constant = [c for c, s in zip(num, std) if not s > 0]
        self.const_values = {c: X[c].iloc[0] for c in self.dropped_constant}
        vals = X[self.num].to_numpy(dtype=np.float64)
        self.int_cols = np.all(np.isclose(vals, np.round(vals)), axis=0)
        block = self._num_block(X)
        self.ord = self.ohe = None
        if self.cat:
            if self.categorical == "ordinal":
                self.ord = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1).fit(X[self.cat])
                block = np.hstack([block, self.ord.transform(X[self.cat])])
            else:
                self.ohe = OneHotEncoder(handle_unknown="ignore",
                                         sparse_output=False).fit(X[self.cat])
        self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(block)
        self.n_num = block.shape[1]
        self.cat_sizes = [len(c) for c in self.ohe.categories_] if self.ohe is not None else []
        self.dim = self.n_num + sum(self.cat_sizes)
        names = list(self.num) + (list(self.cat) if self.ord is not None else [])
        if self.ohe is not None:
            for c, cats in zip(self.cat, self.ohe.categories_):
                names += [f"{c}={v}" for v in cats]
        self.feature_names = names
        return self

    def _num_block(self, X):
        v = X[self.num].to_numpy(dtype=np.float64)
        return signed_log(v) if self.log_transform else v

    def transform(self, X):
        X = self._add_zero_flags(self._ports_to_cat(X))
        block = self._num_block(X)
        if self.ord is not None:
            block = np.hstack([block, self.ord.transform(X[self.cat])])
        parts = [np.clip(self.scaler.transform(block), -1, 1)]
        if self.ohe is not None:
            parts.append(self.ohe.transform(X[self.cat]))
        return np.hstack(parts).astype(np.float32)

    def round_integers(self, df):
        """Round columns that are whole-numbered in the training data."""
        cols = [c for c, is_int in zip(self.num, self.int_cols) if is_int and c in df.columns]
        if cols:
            df[cols] = np.round(df[cols].to_numpy(dtype=np.float64)).astype(np.float32)
        return df

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        block = self.scaler.inverse_transform(np.clip(Z[:, :self.n_num], -1, 1))
        n = len(self.num)
        vals = signed_exp(block[:, :n]) if self.log_transform else block[:, :n]
        out = pd.DataFrame(vals.astype(np.float32), columns=self.num)
        out = self.round_integers(out)
        for c, v in self.const_values.items():
            out[c] = v
        if self.ord is not None:
            for j, c in enumerate(self.cat):
                cats = self.ord.categories_[j]
                code = np.clip(np.rint(block[:, n + j]), 0, len(cats) - 1).astype(int)
                out[c] = cats[code]
        elif self.ohe is not None:
            off = self.n_num
            for c, cats in zip(self.cat, self.ohe.categories_):
                out[c] = cats[Z[:, off:off + len(cats)].argmax(1)]
                off += len(cats)
        for c in self.zero_cols:                               # the switch decides exact zeros
            flag = out.pop(c + "__is_zero").astype(str).to_numpy()
            v = out[c].to_numpy().copy()
            v[flag == "zero"] = 0
            out[c] = v
        return self.ports_from_cat(out)

    def ports_from_cat(self, out):
        """Port categories back to port numbers ("other" -> a training port from that bucket)."""
        for c in self.ports:
            lab = out[c].astype(str).reset_index(drop=True)
            is_other = ~lab.str.startswith("p").to_numpy()
            port = np.zeros(len(lab), dtype=np.float32)
            port[~is_other] = lab[~is_other].str[1:].astype(np.float32).to_numpy()
            port[is_other] = self.rng.choice(self.port_other[c], is_other.sum())
            out[c] = port
        return out


# --------------------------------------------------------------------- models
def mlp(inp, hidden, out, act):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


ARCH = {  # hidden widths of generator / discriminator
    "base":    ([128, 256, 512], [512, 256, 128]),
    "shallow": ([256], [256]),
    "deep":    ([128, 256, 512, 512, 512, 512], [512, 512, 512, 256, 128, 128]),
}


class Generator(nn.Module):
    """Tanh outputs for numerical features; Gumbel-softmax heads for one-hot
    categorical blocks (hard one-hot when sampling)."""

    def __init__(self, noise_dim, cond_dim, n_num, cat_sizes, hidden):
        super().__init__()
        self.n_num, self.cat_sizes = n_num, list(cat_sizes)
        self.net = mlp(noise_dim + cond_dim, hidden, n_num + sum(cat_sizes), nn.ReLU)

    def forward(self, z, y=None, hard=False, tau=0.2):
        h = self.net(z if y is None else torch.cat([z, y], 1))
        parts, off = [torch.tanh(h[:, :self.n_num])], self.n_num
        for k in self.cat_sizes:
            parts.append(F.gumbel_softmax(h[:, off:off + k], tau=tau, hard=hard))
            off += k
        return torch.cat(parts, 1)


class Critic(nn.Module):
    """Discriminator / critic returning logits (the sigmoid is inside the loss)."""

    def __init__(self, in_dim, cond_dim, hidden, out=1):
        super().__init__()
        self.cond = cond_dim > 0
        self.net = mlp(in_dim + cond_dim, hidden, out, lambda: nn.LeakyReLU(0.2))

    def forward(self, x, y=None):
        return self.net(torch.cat([x, y], 1) if self.cond else x)


# -------------------------------------------------------------------- metrics
def mmd2_torch(a, b):
    """Unbiased MMD^2, Gaussian kernel, median-heuristic bandwidth."""
    z = torch.cat([a, b])
    d2 = torch.cdist(z, z).pow(2)
    iu = torch.triu_indices(len(z), len(z), 1, device=z.device)
    pos = d2[iu[0], iu[1]]
    pos = pos[pos > 0]
    sigma2 = pos.median() if len(pos) else torch.tensor(1.0, device=z.device)
    K = torch.exp(-d2 / (2 * sigma2))
    m, n = len(a), len(b)
    kxx = (K[:m, :m].sum() - K[:m, :m].diagonal().sum()) / (m * (m - 1))
    kyy = (K[m:, m:].sum() - K[m:, m:].diagonal().sum()) / (n * (n - 1))
    return float(kxx + kyy - 2 * K[:m, m:].mean())


def mmd2(a, b, max_n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(max_n, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(max_n, len(b)), replace=False)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return mmd2_torch(torch.tensor(a, dtype=torch.float64, device=dev),
                      torch.tensor(b, dtype=torch.float64, device=dev))


def safe_corr(x):
    """Correlation matrix; constant columns get correlation 0 instead of NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(np.corrcoef(x, rowvar=False))


def density_coverage(real, fake, k=5, max_n=5000, seed=0):
    """Naeem et al. (2020): fidelity (density) and diversity (coverage)."""
    rng = np.random.default_rng(seed)
    real = real[rng.choice(len(real), min(max_n, len(real)), replace=False)]
    fake = fake[rng.choice(len(fake), min(max_n, len(fake)), replace=False)]
    k = min(k, len(real) - 1)
    if k < 1 or len(fake) == 0:
        return float("nan"), float("nan")
    radii = NearestNeighbors(n_neighbors=k + 1).fit(real).kneighbors(real)[0][:, -1]
    inside = np.zeros((len(fake), len(real)), dtype=bool)
    for i in range(0, len(fake), 500):                         # chunked distances
        d = np.sqrt(((fake[i:i + 500, None, :] - real[None, :, :]) ** 2).sum(-1))
        inside[i:i + 500] = d < radii[None, :]
    density = inside.sum() / (k * len(fake))
    coverage = inside.any(0).mean()
    return float(density), float(coverage)


def fidelity(real, synth, seed, names=None):
    d = real.shape[1]
    ks = [ks_2samp(real[:, j], synth[:, j]) for j in range(d)]
    order = np.argsort([-k.statistic for k in ks])[:8]
    worst = [{"feature": names[j] if names else j, "ks": float(ks[j].statistic),
              "real_median": float(np.median(real[:, j])), "synth_median": float(np.median(synth[:, j]))}
             for j in order]
    dens, cov = density_coverage(real, synth, seed=seed)
    dcr = NearestNeighbors(n_neighbors=1).fit(real).kneighbors(synth)[0][:, 0]
    return {
        "mmd2_rbf": mmd2(real, synth, seed=seed),
        "wasserstein_mean": float(np.mean([wasserstein_distance(real[:, j], synth[:, j])
                                           for j in range(d)])),
        "corr_dist": float(np.linalg.norm(safe_corr(real) - safe_corr(synth))
                           / real.shape[1]),     # Frobenius norm / number of features
        "ks_stat_median": float(np.median([k.statistic for k in ks])),
        "ks_reject_frac_bonferroni": float(np.mean([k.pvalue < 0.05 / d for k in ks])),
        "density": dens, "coverage": cov,
        "dcr_median": float(np.median(dcr)), "dcr_exact_copies": int((dcr < 1e-6).sum()),
        "most_different_features": worst,
    }


# ------------------------------------------------------------ GAN training
GAN_SPECS = {
    "gan_pseudo":   dict(kind="gan", cond="none"),
    "cgan_gonly":   dict(kind="gan", cond="g", pi="uniform"),
    "cgan_emp":     dict(kind="gan", cond="both", pi="empirical"),
    "cgan":         dict(kind="gan", cond="both", pi="uniform"),
    "wgan_gp":      dict(kind="wgan", cond="both", pi="uniform"),
    "acgan":        dict(kind="acgan", cond="g", pi="uniform"),
    "cgan_ordinal": dict(kind="gan", cond="both", pi="uniform", categorical="ordinal"),
    "cgan_nolog":   dict(kind="gan", cond="both", pi="uniform", log=False),
    "cgan_nozero":  dict(kind="gan", cond="both", pi="uniform", zero=False),
    "wgan_gp_nozero": dict(kind="wgan", cond="both", pi="uniform", zero=False),
    "cgan_shallow": dict(kind="gan", cond="both", pi="uniform", arch="shallow"),
    "cgan_deep":    dict(kind="gan", cond="both", pi="uniform", arch="deep"),
}


class BatchSampler:
    def __init__(self, y, K, pi, rng):
        self.rng, self.K = rng, K
        self.order = np.argsort(y, kind="stable")
        self.sizes = np.bincount(y, minlength=K)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.present = np.flatnonzero(self.sizes)
        self.pi, self.n = pi, len(y)

    def __call__(self, B):
        if self.pi == "empirical":                # rows uniformly = labels at empirical rate
            return self.rng.integers(0, self.n, B)
        cls = self.rng.choice(self.present, B)    # class-balanced
        off = (self.rng.random(B) * self.sizes[cls]).astype(np.int64)
        return self.order[self.starts[cls] + off]


def train_gan(spec, X, y, K, n_num, cat_sizes, minority, cfg, device, rng):
    """Returns the generator checkpoint with the best selection score (mean per-class
    coverage minus MMD^2) on training data; no test data is used for model selection."""
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device, dtype=torch.long)
    eye = torch.eye(K, device=device)
    hid_g, hid_d = ARCH[spec.get("arch", "base")]
    cond_g = K if spec["cond"] in ("g", "both") else 0
    cond_d = K if spec["cond"] == "both" else 0
    kind = spec["kind"]
    G = Generator(cfg.noise_dim, cond_g, n_num, cat_sizes, hid_g).to(device)
    D = Critic(X.shape[1], cond_d, hid_d, out=(1 + K) if kind == "acgan" else 1).to(device)
    if kind == "wgan":
        oG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
        oD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
        n_critic = 5
    else:
        oG = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
        oD = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
        n_critic = 1
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    sampler = BatchSampler(y, K, spec.get("pi", "empirical") if cond_g else "empirical", rng)
    B = cfg.batch
    ones, zeros = torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)

    # fixed real reference sets for checkpoint selection (training data only)
    ref = {}
    for c in (minority if cond_g else [None]):
        pool = np.flatnonzero(y == c) if c is not None else np.arange(len(y))
        pick = rng.choice(pool, min(1000, len(pool)), replace=False)
        ref[c] = Xt[torch.as_tensor(pick, device=device)].double()

    # k-NN radii of the reference sets, for coverage (Naeem et al., 2020)
    radii = {}
    for c, r in ref.items():
        k = min(5, len(r) - 1)
        radii[c] = torch.cdist(r, r).topk(k + 1, largest=False).values[:, -1] if k >= 1 else None

    def score():
        """Selection score on training data: mean over classes of coverage - MMD^2.
        Coverage rewards synthetic records that land where real records are."""
        G.eval()
        with torch.no_grad():
            mm, cv = [], []
            for c, r in ref.items():
                if len(r) < 2:
                    continue
                z = torch.randn(1000, cfg.noise_dim, device=device)
                yy = eye[torch.full((1000,), c, device=device)] if c is not None else None
                f = G(z, yy, hard=True).double()
                mm.append(mmd2_torch(r, f))
                cv.append(float((torch.cdist(r, f) < radii[c][:, None]).any(1).double().mean()))
        G.train()
        if not mm:
            return float("nan"), float("nan"), float("nan")
        return float(np.mean(cv) - np.mean(mm)), float(np.mean(mm)), float(np.mean(cv))

    history, best, best_state, best_iter = [], -np.inf, None, cfg.iters
    train_time, t = 0.0, time.time()
    for it in range(1, cfg.iters + 1):
        for _ in range(n_critic):
            idx = torch.as_tensor(sampler(B), device=device)
            xr, cr = Xt[idx], yt[idx]
            yr = eye[cr] if cond_g else None
            z = torch.randn(B, cfg.noise_dim, device=device)
            xf = G(z, yr).detach()
            yd = yr if cond_d else None
            if kind == "wgan":
                a = torch.rand(B, 1, device=device)
                xh = (a * xr + (1 - a) * xf).requires_grad_(True)
                grad = torch.autograd.grad(D(xh, yd).sum(), xh, create_graph=True)[0]
                gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
                lossD = D(xf, yd).mean() - D(xr, yd).mean() + 10 * gp
            elif kind == "acgan":
                o_r, o_f = D(xr), D(xf)
                lossD = (bce(o_r[:, :1], ones) + bce(o_f[:, :1], zeros)
                         + ce(o_r[:, 1:], cr) + ce(o_f[:, 1:], cr))
            else:
                lossD = bce(D(xr, yd), ones) + bce(D(xf, yd), zeros)
            oD.zero_grad(set_to_none=True); lossD.backward(); oD.step()
        # generator step (non-saturating loss)
        z = torch.randn(B, cfg.noise_dim, device=device)
        xf = G(z, yr)
        if kind == "wgan":
            lossG = -D(xf, yr).mean()
        elif kind == "acgan":
            o_f = D(xf)
            lossG = bce(o_f[:, :1], ones) + ce(o_f[:, 1:], cr)
        else:
            lossG = bce(D(xf, yr if cond_d else None), ones)
        oG.zero_grad(set_to_none=True); lossG.backward(); oG.step()
        if it % cfg.eval_every == 0 or it == cfg.iters:
            train_time += time.time() - t
            s, m2, cov = score()
            history.append({"iter": it, "loss_D": lossD.item(), "loss_G": lossG.item(),
                            "select_score": s, "select_mmd2": m2, "select_coverage": cov})
            if s > best:
                best, best_state, best_iter = s, copy.deepcopy(G.state_dict()), it
            t = time.time()
    if best_state is not None:
        G.load_state_dict(best_state)
    G.eval()
    return G, history, train_time, best_iter


@torch.no_grad()
def gan_sample(G, c, n, K, noise_dim, device, chunk=100_000):
    out = []
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        z = torch.randn(m, noise_dim, device=device)
        y = None
        if c is not None:
            y = torch.zeros(m, K, device=device); y[:, c] = 1
        out.append(G(z, y, hard=True).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, G.n_num + sum(G.cat_sizes)), np.float32)


# ----------------------------------------------------------------- baselines
def smote(P, n_new, rng, k=5):
    if n_new <= 0:
        return np.zeros((0, P.shape[1]), np.float32)
    if len(P) < 2:
        return np.repeat(P, n_new, 0)
    k = min(k, len(P) - 1)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P)[1][:, 1:]
    i = rng.integers(0, len(P), n_new)
    j = nbr[i, rng.integers(0, k, n_new)]
    lam = rng.random((n_new, 1)).astype(np.float32)
    return (P[i] + lam * (P[j] - P[i])).astype(np.float32)


def run_ctgan(Xraw_tr, ytr_names, minority_names, needs, cfg, device, seed, port_pp):
    """CTGAN reference implementation, trained on all minority records plus a
    random sample of the other classes (CTGAN does not scale to millions of rows)."""
    from ctgan import CTGAN
    rng = np.random.default_rng(seed)
    is_min = np.isin(ytr_names, minority_names)
    others = np.flatnonzero(~is_min)
    keep = np.concatenate([np.flatnonzero(is_min),
                           rng.choice(others, min(len(others), cfg.ctgan_rows), replace=False)])
    df = port_pp._ports_to_cat(Xraw_tr.iloc[keep].reset_index(drop=True))  # same port buckets as the GANs
    df["__label__"] = np.asarray(ytr_names)[keep]
    discrete = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    torch.manual_seed(seed)
    model = CTGAN(epochs=cfg.ctgan_epochs, enable_gpu=(device.type == "cuda"))
    t = time.time()
    model.fit(df, discrete_columns=discrete)
    train_time = time.time() - t
    out = {}
    for c, n in needs.items():
        if n > 0:
            s = model.sample(n, condition_column="__label__", condition_value=c)
            s = port_pp.ports_from_cat(s.drop(columns="__label__").reset_index(drop=True))
            out[c] = port_pp.round_integers(s)
    return out, train_time


# ---------------------------------------------------------------- classifiers
def make_rf(seed):
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)


def fit_predict(name, Xtr, ytr, Xte, K, seed, device):
    t = time.time()
    if name == "rf":
        m = make_rf(seed).fit(Xtr, ytr); p = m.predict(Xte)
    elif name == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                          tree_method="hist", device="cpu",   # CPU: consistent across machines
                          random_state=seed, n_jobs=-1)
        m.fit(Xtr, ytr); p = m.predict(Xte)
    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                           min_child_samples=20, min_child_weight=1.0, reg_lambda=1.0,
                           random_state=seed, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr); p = m.predict(Xte)
    elif name == "dnn":
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(32, K)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xt = torch.tensor(Xtr, device=device)
        yt = torch.tensor(ytr, device=device, dtype=torch.long)
        g = torch.Generator().manual_seed(seed)
        steps_per_epoch = max(1, -(-len(Xt) // 1024))
        epochs = int(min(200, max(10, -(-3000 // steps_per_epoch))))   # >= 3000 updates
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), generator=g).to(device)
            for i in range(0, len(Xt), 1024):
                b = perm[i:i + 1024]
                loss = F.cross_entropy(net(Xt[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            Xe = torch.tensor(Xte, device=device)
            p = torch.cat([net(Xe[i:i + 65536]).argmax(1)
                           for i in range(0, len(Xe), 65536)]).cpu().numpy()
    else:
        raise ValueError(name)
    secs = time.time() - t
    # sanity check: a classifier that cannot fit its own training data has failed
    rng = np.random.default_rng(seed)
    chk = rng.choice(len(Xtr), min(len(Xtr), 100_000), replace=False)
    if name == "dnn":
        with torch.no_grad():
            pc = net(torch.tensor(Xtr[chk], device=device)).argmax(1).cpu().numpy()
    else:
        pc = m.predict(Xtr[chk])
    train_f1 = float(f1_score(ytr[chk], np.asarray(pc).astype(int), average="macro", zero_division=0))
    return np.asarray(p).astype(int), secs, train_f1


def metrics(y, p, classes):
    K = len(classes)
    return {"accuracy": float((y == p).mean()),
            "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(K)), zero_division=0)),
            "mcc": float(matthews_corrcoef(y, p)),
            "per_class": classification_report(y, p, labels=list(range(K)), target_names=classes,
                                               output_dict=True, zero_division=0)}


def available_classifiers(names):
    ok = []
    for n in names:
        try:
            if n == "xgb":
                import xgboost  # noqa: F401
            if n == "lgbm":
                import lightgbm  # noqa: F401
            ok.append(n)
        except ImportError:
            log(f"classifier {n} skipped (pip install {'xgboost' if n == 'xgb' else 'lightgbm'})")
    return ok


# ---------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob of CSV files")
    ap.add_argument("--out-dir", required=True, help="one JSON per method and seed goes here")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--nslkdd", action="store_true", help="header-less NSL-KDD file")
    ap.add_argument("--group-families", action="store_true",
                    help="merge fine-grained CIC labels into attack families")
    ap.add_argument("--exclude-classes", default="", help="classes whose records are removed")
    ap.add_argument("--benign-frac", type=float, default=1.0,
                    help="fraction of benign records kept at load time (fixed seed)")
    ap.add_argument("--minority-classes", default="", help="comma-separated classes to augment")
    ap.add_argument("--minority-frac", type=float, default=0.05,
                    help="if --minority-classes is empty: augment classes below this share")
    ap.add_argument("--max-train-per-class", type=int, default=0,
                    help="random undersampling cap per class in the training partition (0 = off)")
    ap.add_argument("--target-ratio", type=float, default=1.0,
                    help="augmented class size as a fraction of the largest training class")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--methods", default="none,smote,cgan")
    ap.add_argument("--classifiers", default="rf,xgb,lgbm,dnn")
    ap.add_argument("--extra-clf-methods", default=",".join(EXTRA_CLF_METHODS))
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--noise-dim", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--ctgan-epochs", type=int, default=100)
    ap.add_argument("--ctgan-rows", type=int, default=50000)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def _is_done(path):
    """A result counts as done if it exists and recorded no error (failed runs are redone)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            r = json.load(f)
    except Exception:
        return False
    return "error" not in r and not any("error" in e for e in r.get("eval", {}).values())


def _save(res, out_dir, method, seed):
    path = os.path.join(out_dir, method, f"seed{seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"saved {path}")


def run(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    log(f"device: {device}{' (' + gpu + ')' if gpu else ''}")
    label_col = "label" if a.nslkdd else a.label_col
    df = load_and_clean(a.data, label_col, a.nslkdd, a.benign_frac)
    if a.group_families:
        df[label_col] = df[label_col].map(cic_family)
    if a.exclude_classes:
        drop = [c.strip() for c in a.exclude_classes.split(",")]
        df = df[~df[label_col].isin(drop)].reset_index(drop=True)
    vc = df[label_col].value_counts()
    df = df[df[label_col].isin(vc[vc >= 2].index)].reset_index(drop=True)
    log(f"{len(df):,} records after cleaning:\n{df[label_col].value_counts().to_string()}")
    Xall, yall = df.drop(columns=[label_col]), df[label_col].astype(str).to_numpy()
    del df
    classes = sorted(np.unique(yall).tolist())
    K = len(classes)
    yall_i = np.searchsorted(classes, yall)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    clfs = available_classifiers([c.strip() for c in a.classifiers.split(",") if c.strip()])
    extra_for = set(m.strip() for m in a.extra_clf_methods.split(","))
    wanted = [m.strip() for m in a.minority_classes.split(",") if m.strip()]
    missing = [m for m in wanted if m not in classes]
    if missing:
        raise ValueError(f"minority classes {missing} not in data; classes are {classes}")

    for seed in [int(s) for s in a.seeds.split(",")]:
        todo = [m for m in methods if a.overwrite or not
                _is_done(os.path.join(a.out_dir, m, f"seed{seed}.json"))]
        if not todo:
            log(f"seed {seed}: all methods done, skipping"); continue
        log(f"===== seed {seed}: {', '.join(todo)} =====")
        set_seed(seed)
        rng = np.random.default_rng(seed)
        tr_idx, te_idx = train_test_split(np.arange(len(yall_i)), test_size=a.test_size,
                                          stratify=yall_i, random_state=seed)
        full_counts = np.bincount(yall_i[tr_idx], minlength=K)
        if a.max_train_per_class > 0:                        # training partition only
            keep = []
            for c in range(K):
                ids = tr_idx[yall_i[tr_idx] == c]
                if len(ids) > a.max_train_per_class:
                    ids = rng.choice(ids, a.max_train_per_class, replace=False)
                keep.append(ids)
            tr_idx = np.sort(np.concatenate(keep))
        Xraw_tr = Xall.iloc[tr_idx].reset_index(drop=True)
        Xraw_te = Xall.iloc[te_idx].reset_index(drop=True)
        ytr, yte = yall_i[tr_idx], yall_i[te_idx]
        pp = Preprocessor(True, "onehot").fit(Xraw_tr)       # common space (classifiers, metrics)
        Xtr, Xte = pp.transform(Xraw_tr), pp.transform(Xraw_te)
        counts = np.bincount(ytr, minlength=K)
        minority = ([classes.index(m) for m in wanted] if wanted else
                    [c for c in range(K) if counts[c] / counts.sum() < a.minority_frac])
        target = int(a.target_ratio * counts.max())
        needs = {c: max(target - counts[c], 0) for c in minority}
        log(f"train {len(ytr):,} / test {len(yte):,}; minority {[classes[c] for c in minority]}; "
            f"target {target:,} per minority class")

        t = time.time()
        base = make_rf(seed).fit(Xtr, ytr)                   # real-only RF (label consistency)
        base_seconds = time.time() - t
        base_pred = base.predict(Xte)
        chk = np.random.default_rng(seed).choice(len(Xtr), min(len(Xtr), 100_000), replace=False)
        base_train_f1 = float(f1_score(ytr[chk], base.predict(Xtr[chk]), average="macro", zero_division=0))

        for method in todo:
            log(f"--- {method} (seed {seed}) ---")
            set_seed(seed)
            mrng = np.random.default_rng(seed + 1000)
            res = {"method": method, "seed": seed, "version": VERSION, "config": vars(a), "device": str(device),
                   "gpu": gpu, "torch": torch.__version__, "python": platform.python_version(),
                   "classes": classes, "minority": [classes[c] for c in minority],
                   "train_counts_before_cap": full_counts.tolist(), "train_counts": counts.tolist(),
                   "test_counts": np.bincount(yte, minlength=K).tolist(),
                   "features": {"numerical": len(pp.num), "categorical": pp.cat,
                                "dropped_constant": pp.dropped_constant, "dim": pp.dim},
                   "target_per_class": target, "timing": {}, "fidelity": {}, "eval": {}}
            syn, syn_tstr = {}, {}
            try:
                if method == "none":
                    pass
                elif method == "ros":
                    for c in minority:
                        P = Xtr[ytr == c]
                        syn[c] = P[mrng.integers(0, len(P), needs[c])]
                elif method == "smote":
                    t = time.time()
                    for c in minority:
                        syn[c] = smote(Xtr[ytr == c], needs[c], mrng)
                    res["timing"]["generate_seconds"] = time.time() - t
                elif method == "ctgan":
                    port_pp = Preprocessor(True, "onehot", port_as_category=True).fit(Xraw_tr, ytr, seed)
                    raw, tt = run_ctgan(Xraw_tr, np.array(classes)[ytr],
                                        [classes[c] for c in minority],
                                        {classes[c]: int(needs[c] + counts[c]) for c in minority},
                                        a, device, seed, port_pp)
                    res["timing"]["generator_train_seconds"] = tt
                    for c in minority:
                        if classes[c] in raw:
                            Z = pp.transform(raw[classes[c]])
                            syn[c], syn_tstr[c] = Z[:needs[c]], Z[needs[c]:]
                elif method in GAN_SPECS:
                    spec = GAN_SPECS[method]
                    if spec.get("categorical") == "ordinal" and not pp.cat:
                        log(f"{method}: the data have no categorical features; not applicable")
                        res["not_applicable"] = "no categorical features"
                        _save(res, a.out_dir, method, seed)
                        continue
                    gpp = Preprocessor(spec.get("log", True), spec.get("categorical", "onehot"),
                                       port_as_category=True,
                                       zero_flags=spec.get("zero", True)).fit(Xraw_tr, ytr, seed)
                    Xg = gpp.transform(Xraw_tr)
                    res["generator_space"] = {"dim": gpp.dim, "port_columns": gpp.ports,
                                              "port_categories": {c: len(v) + 1 for c, v in gpp.port_keep.items()},
                                              "integer_columns": int(gpp.int_cols.sum()),
                                              "zero_flag_columns": len(gpp.zero_cols)}
                    G, hist, tt, best_iter = train_gan(spec, Xg, ytr, K, gpp.n_num, gpp.cat_sizes,
                                                       minority, a, device, mrng)
                    res["timing"]["generator_train_seconds"] = tt
                    res["loss_history"], res["selected_iter"] = hist, best_iter
                    to_common = lambda Z: pp.transform(gpp.inverse_transform(Z))
                    t = time.time()
                    if spec["cond"] == "none":               # A0: unconditional + pseudo-labels
                        want = {c: int(needs[c] + counts[c]) for c in minority}
                        got = {c: [] for c in minority}
                        drawn, cap = 0, max(2_000_000, 20 * sum(want.values()))
                        while drawn < cap and any(sum(len(x) for x in got[c]) < want[c]
                                                  for c in minority):
                            Z = to_common(gan_sample(G, None, 200_000, K, a.noise_dim, device))
                            lab = base.predict(Z); drawn += len(Z)
                            for c in minority:
                                got[c].append(Z[lab == c])
                        found = {}
                        for c in minority:
                            Zc = np.vstack(got[c])
                            found[classes[c]] = int(len(Zc))
                            syn[c], syn_tstr[c] = Zc[:needs[c]], Zc[needs[c]:needs[c] + counts[c]]
                        res["pseudo_label"] = {"generated": drawn, "found_per_class": found}
                    else:
                        for c in minority:
                            Z = to_common(gan_sample(G, c, int(needs[c] + counts[c]), K,
                                                     a.noise_dim, device))
                            syn[c], syn_tstr[c] = Z[:needs[c]], Z[needs[c]:]
                    res["timing"]["generate_seconds"] = time.time() - t
                else:
                    raise ValueError(f"unknown method {method}")

                # fidelity and label consistency (synthetic vs real training records)
                if method != "none":
                    lc, where = {}, {}
                    for c in minority:
                        S = syn.get(c)
                        if S is None or len(S) == 0:
                            continue
                        sub = S[mrng.choice(len(S), min(len(S), 5000), replace=False)]
                        res["fidelity"][classes[c]] = fidelity(Xtr[ytr == c], sub, seed, pp.feature_names)
                        pred = base.predict(sub)
                        lc[classes[c]] = float((pred == c).mean())
                        where[classes[c]] = {classes[k]: round(float((pred == k).mean()), 4)
                                             for k in np.unique(pred)}
                    res["label_consistency"] = lc
                    res["synthetic_classified_as"] = where

                # classification: real (+ synthetic) training data, real test data
                parts = [(syn[c], c) for c in minority if c in syn and len(syn[c])]
                Xa = np.vstack([Xtr] + [s for s, _ in parts]) if parts else Xtr
                ya = np.concatenate([ytr] + [np.full(len(s), c) for s, c in parts]) if parts else ytr
                for clf in clfs:
                    if clf != "rf" and method not in extra_for:
                        continue
                    try:
                        if method == "none" and clf == "rf":
                            p, secs, train_f1 = base_pred, base_seconds, base_train_f1
                        else:
                            p, secs, train_f1 = fit_predict(clf, Xa, ya, Xte, K, seed, device)
                    except Exception as e:                   # one classifier failing must not
                        log(f"!! {clf} failed: {e!r}")       # discard the method's other results
                        res["eval"][clf] = {"error": repr(e)}
                        continue
                    res["eval"][clf] = metrics(yte, p, classes)
                    res["eval"][clf]["train_macro_f1"] = train_f1
                    if train_f1 < 0.5:
                        res["eval"][clf]["degenerate"] = True
                        log(f"!! {clf} failed to fit its training data (train macro-F1 {train_f1:.3f}); "
                            f"excluded from the summary tables")
                    res["timing"][f"{clf}_seconds"] = secs
                    pc = res["eval"][clf]["per_class"]
                    log(f"{clf}: macro-F1 {res['eval'][clf]['macro_f1']:.4f} | " +
                        ", ".join(f"{classes[c]} R={pc[classes[c]]['recall']:.3f}" for c in minority))
                # TSTR: minority classes replaced by the same number of synthetic records
                if syn_tstr and all(c in syn_tstr and len(syn_tstr[c]) for c in minority):
                    keep = ~np.isin(ytr, minority)
                    Xs = np.vstack([Xtr[keep]] + [syn_tstr[c] for c in minority])
                    ys = np.concatenate([ytr[keep]] + [np.full(len(syn_tstr[c]), c) for c in minority])
                    p, _, _ = fit_predict("rf", Xs, ys, Xte, K, seed, device)
                    res["eval"]["rf_tstr"] = metrics(yte, p, classes)
            except Exception as e:                           # record the failure and go on
                import traceback
                log(f"!! {method} failed: {e!r}")
                res["error"] = traceback.format_exc()
            _save(res, a.out_dir, method, seed)
    log("finished")


def run_from_settings():
    pattern = DATA_PATHS[DATASET]
    if not glob.glob(pattern):
        sys.exit(f"\nNo data found at {pattern}\n"
                 f"Open this file, find DATA_PATHS near the top, and set the path for "
                 f"'{DATASET}' to where your files are.")
    if QUICK_TEST:
        out = os.path.join(RESULTS_DIR, DATASET + "_quicktest")
        argv = ["--data", pattern, "--out-dir", out, "--seeds", "0", "--iters", "1000",
                "--eval-every", "250", "--methods", "none,smote,gan_pseudo,cgan,wgan_gp",
                "--classifiers", "rf", "--overwrite"]
    else:
        out = os.path.join(RESULTS_DIR, DATASET + "_" + VERSION)   # never mixes with older runs
        argv = ["--data", pattern, "--out-dir", out, "--seeds", ",".join(map(str, SEEDS)),
                "--methods", ",".join(METHOD_SETS[METHOD_SET])]
    run(parse_args(argv + DATASET_DEFAULTS[DATASET]))
    print(f"\nDone. Results are in {out}\n"
          f"Summarise them with:  python summarize_results.py \"{out}\"")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(parse_args())
    else:
        run_from_settings()
