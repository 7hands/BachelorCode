"""Alle Netze und ein Bauplan dafuer - benutzt von Training und Live-System.

Beide bauen ueber baue() aus denselben Angaben, die auch im Checkpoint stehen.
Damit kann die Architektur live nicht von der trainierten abweichen.

  arch: voll12 | pos4 | tips6pos | GCN      direkte Klassifikation
        beat | beat_gcn                      Stufe 1 der indirekten Modelle
        clf                                  Stufe 2 der indirekten Modelle

TCN-Block (Bai et al. 2018): zwei kausale, dilatierte Faltungen mit Kernel 3,
Weight Norm, ReLU, Dropout 0.3 und Residualverbindung. Vier Bloecke mit den
Dilatationen 1, 2, 4, 8 ergeben ein rezeptives Feld von 61 Frames (2,4 s).

  python modelle.py      # Selbsttest: Parameterzahlen der trainierten Modelle
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import merkmale as MM

KANAELE = {"voll12": 12, "pos4": 4, "tips6pos": 12}
GRAPH = ("GCN", "beat_gcn")
DILS = (1, 2, 4, 8)

# Handskelett von MediaPipe: 21 Knoten, 21 Kanten
BONES = [(0, 1), (1, 2), (2, 3), (3, 4),            # Daumen
         (0, 5), (5, 6), (6, 7), (7, 8),            # Zeigefinger
         (5, 9), (9, 10), (10, 11), (11, 12),       # Mittelfinger
         (9, 13), (13, 14), (14, 15), (15, 16),     # Ringfinger
         (13, 17), (17, 18), (18, 19), (19, 20),    # kleiner Finger
         (0, 17)]                                   # Handflaeche schliessen


# --- TCN ----------------------------------------------------------------------
class Block(nn.Module):
    """Kausaler TCN-Block: (B,C,T) -> (B,C',T)."""

    def __init__(self, cin, cout, k=3, d=1, drop=0.3, convs=2, wn=True):
        super().__init__()
        self.pad = (k - 1) * d
        mk = lambda a, b: weight_norm(nn.Conv1d(a, b, k, dilation=d)) if wn \
            else nn.Conv1d(a, b, k, dilation=d)
        self.c1 = mk(cin, cout)
        self.c2 = mk(cout, cout) if convs == 2 else None
        self.drop = nn.Dropout(drop)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        y = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
        if self.c2 is not None:
            y = self.drop(F.relu(self.c2(F.pad(y, (self.pad, 0)))))
        return y + (x if self.res is None else self.res(x))


def _tcn(cin, dils, convs, wn, ch=32):
    c, bl = cin, []
    for d in dils:
        bl.append(Block(c, ch, 3, d, 0.3, convs, wn)); c = ch
    return nn.Sequential(*bl)


class TCNKopf(nn.Module):
    """Direkte Klassifikation: TCN, Mittel ueber die Zeit, lineare Schicht."""

    def __init__(self, cin, nclass, dils, convs, wn, ch=32):
        super().__init__()
        self.net = _tcn(cin, dils, convs, wn, ch)
        self.head = nn.Linear(ch, nclass)

    def forward(self, x):
        return self.head(self.net(x).mean(dim=2))


class BeatX(nn.Module):
    """Stufe 1: Phase (cos, sin) je Frame, 1x1-Faltung + tanh."""

    def __init__(self, cin, dils, convs, wn, ch=32):
        super().__init__()
        self.net = _tcn(cin, dils, convs, wn, ch)
        self.head = nn.Conv1d(ch, 2, 1)

    def forward(self, x):
        return torch.tanh(self.head(self.net(x)))


class ClfX(TCNKopf):
    """Stufe 2: aus den vier Phasenkurven die Taktart (wie TCNKopf, 5 Klassen)."""

    def __init__(self, cin, dils, convs, wn, ch=32, nclass=5):
        super().__init__(cin, nclass, dils, convs, wn, ch)


# --- GCN ----------------------------------------------------------------------
def adjacency():
    """D^-1/2 (A+I) D^-1/2 - symmetrisch normiert."""
    A = np.eye(21)
    for a, b in BONES:
        A[a, b] = A[b, a] = 1
    d = A.sum(1) ** -0.5
    return torch.tensor((A * d).T * d, dtype=torch.float32)


class GraphConv(nn.Module):
    """(B,C,T,V) -> (B,C',T,V). 1x1-Faltung mit denselben Gewichten fuer alle
    Knoten, danach Mischung jedes Knotens mit seinen Nachbarn ueber A."""

    def __init__(self, cin, cout, A):
        super().__init__()
        self.register_buffer("A", A)
        self.lin = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return torch.einsum("bctv,vw->bctw", self.lin(x), self.A)


class STBlockX(nn.Module):
    """ST-GCN-Block: Graph-Faltung, dann die Zeitfaltungen des TCN-Blocks je Knoten."""

    def __init__(self, cin, cout, A, k=3, d=1, drop=0.3, convs=2, wn=True):
        super().__init__()
        self.g = GraphConv(cin, cout, A)
        self.pad = (k - 1) * d
        mk = lambda: weight_norm(nn.Conv2d(cout, cout, (k, 1), dilation=(d, 1))) \
            if wn else nn.Conv2d(cout, cout, (k, 1), dilation=(d, 1))
        self.t1 = mk()
        self.t2 = mk() if convs == 2 else None
        self.drop = nn.Dropout(drop)
        self.res = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        y = F.relu(self.g(x))
        y = self.drop(F.relu(self.t1(F.pad(y, (0, 0, self.pad, 0)))))
        if self.t2 is not None:
            y = self.drop(F.relu(self.t2(F.pad(y, (0, 0, self.pad, 0)))))
        return y + (x if self.res is None else self.res(x))


class GCNX(nn.Module):
    """nclass gesetzt: direkte Klassifikation. nclass=None: Stufe 1 (Phase je Frame)."""

    def __init__(self, nclass, ch=32, dils=DILS, cin=2, convs=2, wn=True):
        super().__init__()
        A = adjacency()
        c, bl = cin, []
        for d in dils:
            bl.append(STBlockX(c, ch, A, d=d, convs=convs, wn=wn)); c = ch
        self.net = nn.Sequential(*bl)
        self.nclass = nclass
        self.head = nn.Linear(ch, nclass) if nclass else nn.Conv1d(ch, 2, 1)

    def forward(self, x):
        h = self.net(x).mean(dim=3)                      # Mittel ueber die Knoten
        return self.head(h.mean(dim=2)) if self.nclass else torch.tanh(self.head(h))


# --- Bauplan ------------------------------------------------------------------
def baue(arch, dils=DILS, convs=2, wn=True, nclass=5, cin=None):
    """-> Modul. cin nur fuer Stufe 1/2 der indirekten Modelle (4 Kanaele)."""
    if arch == "GCN":
        return GCNX(nclass, dils=dils, convs=convs, wn=wn)
    if arch == "beat_gcn":
        return GCNX(None, dils=dils, convs=convs, wn=wn)
    if arch == "beat":
        return BeatX(cin or 4, dils, convs, wn)
    if arch == "clf":
        return ClfX(cin or 4, dils, convs, wn, nclass=nclass)
    return TCNKopf(KANAELE[arch], nclass, dils, convs, wn)


def eingang(arch, variante=None):
    """-> Funktion (rec, s, e) -> Eingabefenster (C, W) bzw. (2, W, 21)."""
    if arch in GRAPH or variante == "graph":
        return MM.graph_win
    var = variante or arch
    return lambda r, s, e: MM.feats(r, s, e, var).T


def demo():
    """Parameterzahlen gegen die trainierten Modelle der Arbeit."""
    n = lambda m: sum(p.numel() for p in m.parameters())
    soll = {"voll12": 23749, "pos4": 22725, "tips6pos": 23749, "GCN": 28613}
    for arch, zahl in soll.items():
        assert n(baue(arch)) == zahl, f"{arch}: {n(baue(arch))} statt {zahl}"
    b, c, g = n(baue("beat")), n(baue("clf")), n(baue("beat_gcn"))
    assert 2 * b + c == 67977, f"TCN indirekt: {2 * b + c} statt 67977"
    assert 2 * g + c == 79753, f"GCN indirekt: {2 * g + c} statt 79753"
    print("modelle OK: Parameterzahlen stimmen mit den trainierten Modellen ueberein")


if __name__ == "__main__":
    demo()
