"""Training und Auswertung JE AUFNAHME - fuer den Vergleich der Fenstergroessen.

Je Aufnahme entscheidet die Mehrheit aller Fenster; gemessen wird der Anteil
richtig erkannter Aufnahmen. Fuer die Wahl der Fenstergroesse interessiert
nur die Rangfolge. Die Auswertung je Takt macht takt_training.py, das die
Bausteine hier (Splits, Training, Kaskade) mitbenutzt.

Eine Zeile je (Split, Modell, Fenster, Seed) in der CSV - Rohwerte, kein
Mittelwert. Schon gerechnete Kombinationen werden uebersprungen.

  python training.py --selfcheck          # nur zeigen, was gerechnet wuerde
  python training.py
  python training.py --modelle pos4,GCN --fenster 50,75 --seeds 3

Jede Einstellung unten ist auch als Flag verfuegbar (gleicher Name mit --).
"""

# ═══════════════════════════ EINSTELLUNGEN ═══════════════════════════════════

MODELLE = ["voll12", "pos4", "tips6pos", "GCN", "TCN->TCN", "GCN->TCN"]

FENSTER = [10, 25, 50, 75, 100]            # Frames, fuer die direkten Modelle
KASKADE_W1 = [10, 25, 50]                  # Stufe 1 der indirekten Modelle
KASKADE_W2 = [25, 50, 75, 100, 125, 150, 175]   # Stufe 2

SPLITS = None                          # None = alle: ohne <Person> je Person + 80/20

SEEDS = 10
EPOCHEN = 10

BLOCK_CONVS = 2                        # Faltungen je TCN-Block
BLOCK_WEIGHTNORM = True

GEMITTELTE_BAHNEN = True               # kuenstliche Aufnahmen aus dem Mittel
NUR_KAMERA_1 = False                   # True halbiert die Daten
BLOCK2_IN_TEST = True                  # Teil 3b zaehlt im Split ohne 3 (BLOCK2_ZU)
                                       # zum ungesehenen Test

CSV = "fenster_vergleich.csv"

# ═════════════════════════════════════════════════════════════════════════════

import csv
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

import daten as DS
import merkmale as MM
import modelle as MD


def _f(name, vorgabe, typ=str):
    """Kommandozeilen-Flag --name ueberschreibt die Einstellung."""
    flag = "--" + name.lower()
    if flag not in sys.argv:
        return vorgabe
    roh = sys.argv[sys.argv.index(flag) + 1]
    if isinstance(vorgabe, list) or vorgabe is None:
        return [typ(x.strip()) for x in roh.split(",")]
    if isinstance(vorgabe, bool):
        return roh.lower() in ("1", "true", "ja", "yes")
    return type(vorgabe)(roh)


MODELLE = _f("modelle", MODELLE)
FENSTER = _f("fenster", FENSTER, int)
KASKADE_W1 = _f("kaskade_w1", KASKADE_W1, int)
KASKADE_W2 = _f("kaskade_w2", KASKADE_W2, int)
SEEDS = _f("seeds", SEEDS)
EPOCHEN = _f("epochen", EPOCHEN)
BLOCK_CONVS = _f("block_convs", BLOCK_CONVS)
BLOCK_WEIGHTNORM = _f("block_weightnorm", BLOCK_WEIGHTNORM)
GEMITTELTE_BAHNEN = _f("gemittelte_bahnen", GEMITTELTE_BAHNEN)
NUR_KAMERA_1 = _f("nur_kamera_1", NUR_KAMERA_1)
BLOCK2_IN_TEST = _f("block2_in_test", BLOCK2_IN_TEST)
CSV = _f("csv", CSV)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KLASSEN = [1, 2, 3, 4, 6]
KASK = ("TCN->TCN", "GCN->TCN")        # indirekte Modelle: Stufe 1 + Stufe 2
HOP1 = 4                               # Kurvenraster der Stufe 1 in Frames
# Reihenfolge der gehaltenen Personen = Reihenfolge der Zufallsziehungen, mit
# der die gespeicherten Modelle trainiert wurden (nicht aendern).
LOPO = tuple(DS.PERSONEN[i] for i in (1, 2, 0))
ALLE = "alle Personen (80/20)"
SPLITS = _f("splits", SPLITS) or [f"ohne {p}" for p in LOPO] + [ALLE]


def splits(recs):
    """-> {Name: (train, {Testname: recs})}

    Je Person ein Split ohne diese Person (ungesehen), dazu ein Split ueber
    alle. Von den trainierenden Personen werden je (Person, Taktart) 20 % der
    Aufnahmen zurueckgehalten ("bekannt"). Beide Kameras einer Aufnahme
    bleiben zusammen. Die gemittelten Bahnen entstehen JE SPLIT nur aus den
    Trainingsaufnahmen - sonst flossen die zurueckgehaltenen 20 % ueber die
    Mittelbahn ihrer Person ins Training.

    Teil 3b zaehlt zur Person BLOCK2_ZU (3), wird aber gesondert behandelt,
    weil ein Teil der Aufnahmen von einer vierten Person stammt:
      * Split ohne 3: nicht im Training; mit BLOCK2_IN_TEST Teil des
        ungesehenen Tests von 3.
      * alle anderen Splits: nur im Training, nie im Testset "bekannt".
    """
    rng = np.random.default_rng(0)
    aus = {}
    for weg in list(LOPO) + [None]:
        ohne_b2 = weg == DS.BLOCK2_ZU
        b2 = [r for r in recs if r["block2"]]
        drin = [r for r in recs if r["person"] != weg and not r["block2"]]
        held = [r for r in recs if r["person"] == weg and not r["block2"]] if weg else []
        tr, te = [], []
        gr = {}
        for r in drin:
            gr.setdefault((r["person"], r["meter"]), []).append(r)
        for _, rs in sorted(gr.items(), key=lambda kv: (DS._rang(kv[0][0]), kv[0][1])):
            namen = sorted({x["name"].split("#")[0] for x in rs})
            n = max(1, int(round(len(namen) * 0.2))) if len(namen) > 2 else 0
            raus = set(rng.choice(namen, size=n, replace=False)) if n else set()
            for x in rs:
                (te if x["name"].split("#")[0] in raus else tr).append(x)
        if not ohne_b2:                                  # Teil 3b nur ins Training
            tr += b2
        sy = DS.gemittelt(tr) if GEMITTELTE_BAHNEN else []
        tests = {"bekannt (20%)": te}
        if ohne_b2 and BLOCK2_IN_TEST:
            tests[f"ungesehen: {weg}"] = held + b2
        elif held:
            tests[f"ungesehen: {weg}"] = held
        name = f"ohne {weg}" if weg else ALLE
        aus[name] = (tr + sy, tests)

        # Kontrollen - lieber hier abbrechen als eine Zahl zu viel berichten
        trn = {x["name"] for x in tr + sy}
        quellen = {q for x in sy for q in x.get("quellen", [])}
        for tn, tv in tests.items():
            tn_namen = {x["name"] for x in tv}
            assert not (trn & tn_namen), f"LECK {name} / {tn}"
            assert not (quellen & tn_namen), f"LECK ueber die Mittelbahnen: {name} / {tn}"
            b2_erlaubt = ohne_b2 and BLOCK2_IN_TEST and tn.startswith("ungesehen")
            assert b2_erlaubt or not any(x["block2"] for x in tv), \
                f"Teil 3b im Test: {name} / {tn}"
        if weg:
            assert not any(x["person"] == weg for x in tr + sy), f"{weg} im Training von {name}"
        assert ohne_b2 != any(x["block2"] for x in tr + sy), \
            f"Regel fuer Teil 3b verletzt in {name}"
    return aus


def dils(win):
    return MD.DILS


def netz(mn, win):
    """Ein direktes Modell mit den eingestellten Blockoptionen."""
    return MD.baue(mn, dils(win), BLOCK_CONVS, BLOCK_WEIGHTNORM, len(KLASSEN))


def bauer(mn):
    return (MM.graph_win if mn == "GCN"
            else (lambda r, s, e, v=mn: MM.feats(r, s, e, v).T))


def stapel(recs, mn, win):
    """Trainingsfenster mit Schrittweite win/4 (mindestens 4 Frames)."""
    build = bauer(mn)
    X, Y = [], []
    for r in recs:
        for s in range(0, len(r["raw"]) - win + 1, max(4, win // 4)):
            X.append(build(r, s, s + win)); Y.append(KLASSEN.index(int(r["meter"])))
    return (torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE),
            torch.tensor(Y).to(DEVICE))


def fit(m, X, Y):
    """Adam, Lernrate 2e-3, Batch 256, Kreuzentropie."""
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    lf = torch.nn.CrossEntropyLoss()
    for _ in range(EPOCHEN):
        m.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            j = perm[i:i + 256]
            loss = lf(m(X[j]), Y[j])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


# --- Indirekte Modelle ----------------------------------------------------------
def stufe1(tr, key, seed, w1, graph):
    """Stufe 1 auf 'ictus' oder 'down': Phase (cos, sin) je Frame, maskierter MSE."""
    torch.manual_seed(seed); np.random.seed(seed)
    build = MM.graph_win if graph else (lambda r, s, e: MM.feats(r, s, e, "pos4").T)
    X, Y, M = [], [], []
    for r in tr:
        tgt, msk = MM.phase_target(r[key], len(r["raw"]))
        for s in range(0, len(r["raw"]) - w1 + 1, max(4, w1 // 4)):
            X.append(build(r, s, s + w1))
            Y.append(tgt[s:s + w1]); M.append(msk[s:s + w1])
    X = torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE)
    Y = torch.tensor(np.array(Y), dtype=torch.float32).transpose(1, 2).to(DEVICE)
    M = torch.tensor(np.array(M), dtype=torch.float32).to(DEVICE)
    m = MD.baue("beat_gcn" if graph else "beat", dils(w1), BLOCK_CONVS,
                BLOCK_WEIGHTNORM, cin=4).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(EPOCHEN):
        m.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            j = perm[i:i + 256]; mk = M[j][:, None, :]
            loss = (((m(X[j]) - Y[j]) ** 2) * mk).sum() / mk.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
    del X, Y, M
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return m.eval()


def kurve(mi, md, rec, w1, graph):
    """Vierkanalige Phasenkurve einer ganzen Aufnahme.

    Stufe 1 laeuft auf Fenstern von w1 Frames, deren Enden HOP1 Frames
    auseinanderliegen; von jedem Fenster werden nur die letzten HOP1 Werte
    uebernommen, weil nur diese den vollen Kontext hinter sich haben."""
    n = len(rec["raw"])
    ends = list(range(w1, n + 1, HOP1))
    if len(ends) < 2:
        return None
    build = MM.graph_win if graph else (lambda r, s, e: MM.feats(r, s, e, "pos4").T)
    c = np.zeros((n, 4), np.float32)
    xs = torch.tensor(np.stack([build(rec, e - w1, e) for e in ends]),
                      dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        oi = torch.cat([mi(xs[i:i + 256]) for i in range(0, len(xs), 256)]).cpu().numpy()
        od = torch.cat([md(xs[i:i + 256]) for i in range(0, len(xs), 256)]).cpu().numpy()
    for k, e in enumerate(ends):
        lo = max(e - HOP1, w1 - HOP1); j = lo - (e - w1)
        c[lo:e, :2] = oi[k][:, j:].T; c[lo:e, 2:] = od[k][:, j:].T
    return c


def stufe2_daten(tr, kurven, w1, win):
    """Trainingsfenster der Stufe 2 aus den Kurven der Trainingsaufnahmen."""
    X, Y = [], []
    for r in tr:
        c = kurven.get(r["name"])
        if c is None or len(c) < w1 + win:
            continue
        for s in range(w1, len(c) - win + 1, max(4, win // 4)):
            X.append(c[s:s + win].T); Y.append(KLASSEN.index(int(r["meter"])))
    if not X:
        return None, None
    return (torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE),
            torch.tensor(Y).to(DEVICE))


# --- Auswertung -----------------------------------------------------------------
def vorhersagen(mod, mn, rec, win, w1=0, kurve_rec=None):
    """-> (Klasse je Fenster, Fensterenden exklusiv). Ein Fenster alle STEP Frames."""
    if mn in KASK:
        if kurve_rec is None:
            return np.array([]), []
        ends = list(range(w1 + win, len(kurve_rec) + 1, MM.STEP))
        xs = [kurve_rec[e - win:e].T for e in ends]
    else:
        build = bauer(mn)
        ends = list(range(win, len(rec["raw"]) + 1, MM.STEP))
        xs = [build(rec, e - win, e) for e in ends]
    if not ends:
        return np.array([]), []
    x = torch.tensor(np.stack(xs), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        p = torch.cat([mod(x[i:i + 256]) for i in range(0, len(x), 256)]).argmax(1)
    return p.cpu().numpy(), ends


def je_aufnahme(mod, mn, recs, win, kurven=None, w1=0):
    """Anteil der Aufnahmen, deren Mehrheit der Fenster richtig ist."""
    ok = []
    for r in recs:
        p, _ = vorhersagen(mod, mn, r, win, w1, kurven.get(r["name"]) if kurven else None)
        if len(p):
            ok.append(KLASSEN[Counter(p.tolist()).most_common(1)[0][0]] == int(r["meter"]))
    return float(np.mean(ok)) if ok else np.nan


SPALTEN = ["datensatz", "split", "gehalten", "modell", "fenster", "w1", "w2",
           "vorlauf_s", "seed", "epochen", "block", "dilationen", "params",
           "trials_train", "sek", "bekannt", "ungesehen"]


def plan():
    """-> Liste (split, modell, win, w1) die gerechnet werden."""
    aus = []
    for sp in SPLITS:
        for mn in MODELLE:
            if mn in KASK:
                aus += [(sp, mn, w2, w1) for w1 in KASKADE_W1 for w2 in KASKADE_W2]
            else:
                aus += [(sp, mn, win, 0) for win in FENSTER]
    return aus


def lade_daten():
    recs = DS.lade()
    if NUR_KAMERA_1:
        recs = [r for r in recs if r["name"].endswith("#0")]
    return recs


def zeige_splits(verteilung):
    for nm, (tr, te) in verteilung.items():
        e = Counter(r["person"] for r in tr if not r.get("syn"))
        y = Counter(r["person"] for r in tr if r.get("syn"))
        print(f"  {nm}")
        print(f"      Training {len(tr):>4}   echt {dict(sorted(e.items()))}"
              f"  gemittelt {dict(sorted(y.items()))}")
        for k, v in te.items():
            auf = len({x["name"].split("#")[0] for x in v})
            tk = dict(sorted(Counter(int(x["meter"]) for x in v).items()))
            print(f"      {k:<26}{len(v):>4} Spuren = {auf:>2} Aufnahmen   Takte {tk}")
    print("\n  Kontrollen bestanden: kein Leck zwischen Training und Test, "
          "gehaltene Person\n  nirgends im Training, Teil 3b nur wie vorgesehen.\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    P = plan()
    print(f"Modelle {MODELLE}")
    print(f"Fenster direkt {FENSTER} | indirekt W1 {KASKADE_W1} W2 {KASKADE_W2}")
    print(f"Splits {SPLITS} | {SEEDS} Seeds | {EPOCHEN} Epochen | {DEVICE}")
    print(f"-> {len(P) * SEEDS} Durchgaenge, Ziel {CSV}\n")
    verteilung = splits(lade_daten())
    if "--selfcheck" in sys.argv:
        zeige_splits(verteilung)
        sys.exit()

    neu = not os.path.exists(CSV)
    fertig = set()
    if not neu:
        if next(csv.reader(open(CSV)), []) != SPALTEN:
            sys.exit(f"{CSV} hat einen anderen Spaltensatz - umbenennen oder --csv setzen.")
        for z in csv.DictReader(open(CSV)):
            fertig.add((z["split"], z["gehalten"], z["modell"],
                        int(z["fenster"]), int(z["w1"] or 0), int(z["seed"])))
        print(f"{len(fertig)} Zeilen schon in {CSV}, werden uebersprungen")
    fh = open(CSV, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=SPALTEN)
    if neu:
        w.writeheader(); fh.flush()

    t_all = time.time()
    kur_cache = {}
    for sp, mn, win, w1 in P:
        if sp not in verteilung:
            print(f"  Split '{sp}' unbekannt - uebersprungen"); continue
        tr, tests = verteilung[sp]
        unges = next((k for k in tests if k.startswith("ungesehen")), None)
        geh = unges.split(":")[1].strip() if unges else ""
        for sd in range(SEEDS):
            if (sp, geh, mn, win, w1, sd) in fertig:
                continue
            t0 = time.time()
            torch.manual_seed(sd); np.random.seed(sd)
            if mn in KASK:
                schl = (sp, mn, w1, sd)
                if schl not in kur_cache:          # Stufe 1 fuer alle W2 wiederverwenden
                    kur_cache.clear()
                    graph = mn == "GCN->TCN"
                    mi = stufe1(tr, "ictus", sd, w1, graph)
                    md = stufe1(tr, "down", sd, w1, graph)
                    alle = list(tr) + [r for v in tests.values() for r in v]
                    kur_cache[schl] = ({r["name"]: kurve(mi, md, r, w1, graph) for r in alle},
                                       sum(p.numel() for p in mi.parameters())
                                       + sum(p.numel() for p in md.parameters()))
                    del mi, md
                kurven, p1 = kur_cache[schl]
                X, Y = stufe2_daten(tr, kurven, w1, win)
                if X is None:
                    continue
                torch.manual_seed(sd)
                mod = fit(MD.baue("clf", dils(win), BLOCK_CONVS, BLOCK_WEIGHTNORM,
                                  cin=4).to(DEVICE), X, Y)
                npar = p1 + sum(p.numel() for p in mod.parameters())
            else:
                kurven = None
                X, Y = stapel(tr, mn, win)
                torch.manual_seed(sd)
                mod = fit(netz(mn, win).to(DEVICE), X, Y)
                npar = sum(p.numel() for p in mod.parameters())
            bek = je_aufnahme(mod, mn, tests["bekannt (20%)"], win, kurven, w1)
            ung = je_aufnahme(mod, mn, tests[unges], win, kurven, w1) if unges else np.nan
            w.writerow({
                "datensatz": "/".join(DS.DIRS), "split": sp, "gehalten": geh,
                "modell": mn, "fenster": win, "w1": w1 or "", "w2": win if w1 else "",
                "vorlauf_s": f"{(w1 + win) / MM.FPS:.2f}", "seed": sd, "epochen": EPOCHEN,
                "block": f"{BLOCK_CONVS}conv" + ("+wn" if BLOCK_WEIGHTNORM else ""),
                "dilationen": "-".join(map(str, dils(win))), "params": npar,
                "trials_train": len(tr), "sek": f"{time.time() - t0:.0f}",
                "bekannt": f"{bek:.4f}", "ungesehen": f"{ung:.4f}"})
            fh.flush()
            print(f"  {sp:<22}{mn:<10}"
                  + (f"W1={w1:<3}W2={win:<4}" if w1 else f"WIN={win:<8}")
                  + f"s{sd}  bekannt {bek:.2f}  ungesehen {ung:.2f}  "
                    f"({time.time() - t0:.0f}s)")
            del X, Y, mod
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    fh.close()
    print(f"\nfertig in {(time.time() - t_all) / 60:.0f} min -> {CSV}")
