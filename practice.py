
from math import comb, exp, lgamma, log
import matplotlib.pyplot as plt
import numpy as np


# Shared probability functions

def binomial_pmf(x, n, p):
    return comb(n, x) * p**x * (1 - p)**(n - x) if 0 <= x <= n else 0.0


def neg_binomial_pmf(y, r, p):
    return comb(y - 1, r - 1) * p**r * (1 - p)**(y - r) if y >= r else 0.0


def hypergeometric_pmf(x, N, K, n):
    lo = max(0, n - N + K)
    hi = min(K, n)
    return comb(K, x) * comb(N - K, n - x) / comb(N, n) if lo <= x <= hi else 0.0


def neg_hypergeometric_pmf(y, N, K, r):
    if r <= y <= N - K + r:
        return comb(y - 1, r - 1) * comb(N - y, K - r) / comb(N, K)
    return 0.0


def poisson_pmf(x, mean):
    if x >= 0 and mean > 0:
        return exp(-mean + x * log(mean) - lgamma(x + 1))
    return float(x == 0)


def gamma_pdf(t, r, rate):
    if t > 0:
        return exp(r * log(rate) + (r - 1) * log(t) - rate * t - lgamma(r))
    return 0.0


def seat_open_probability(t, seats, rate):
    return sum(poisson_pmf(k, rate * t) for k in range(seats))


def posterior(prevalence, sensitivity, specificity, result="positive"):
    if result == "positive":
        a = prevalence * sensitivity
        b = (1 - prevalence) * (1 - specificity)
    else:
        a = prevalence * (1 - sensitivity)
        b = (1 - prevalence) * specificity
    return a / (a + b)



# Shared simulation helpers
_sample_cache = {}


def clear_sample_cache():
    """Clear cached random samples, useful when explicitly resetting an app."""
    _sample_cache.clear()


def get_sample(name, settings, seed, build):
    key = (name, tuple(settings), seed)
    if key not in _sample_cache:
        _sample_cache[key] = build(np.random.default_rng(seed))
    return _sample_cache[key]


def measure(seq, n, r):
    """Implementation for X and Y"""
    X = seq[:, :n].sum(axis=1)
    reached = np.cumsum(seq, axis=1) >= r
    return X, reached.argmax(axis=1) + 1


def build_arrivals(rng, rate, seats, runs=10000, horizon=60.0):
    """
    It uses millisecond slots and geometric waiting times, exactly as the
    current Python session does.
    """
    slots = 1000
    columns = max(seats, int(rate * horizon + 4 * np.sqrt(rate * horizon) + 20))

    arrival_slots = rng.geometric(rate / slots, size=(runs, columns)).cumsum(axis=1)

    while np.any(arrival_slots[:, -1] <= horizon * slots):
        extra = (arrival_slots[:, -1, None] + rng.geometric(rate / slots, size=(runs, 32)).cumsum(axis=1))
        arrival_slots = np.concatenate((arrival_slots, extra), axis=1)

    return arrival_slots, horizon


# Shared plotting helper

def population_grid(ax, groups, per_square, N=50, first_width=None):
    q = [int(round(n / per_square)) for n, _, _ in groups]

    w = first_width or max(1, int(np.ceil((q[0] + q[1]) / N)))
    h = int(np.ceil((q[0] + q[1]) / w))

    colours = np.full((N, N), "", dtype="<U7")

    block = [(r, c) for r in range(h) for c in range(w)]
    for k, (r, c) in enumerate(block[: q[0] + q[1]]):
        colours[r, c] = groups[0][1] if k < q[0] else groups[1][1]

    rest = [
        (r, c)
        for r in range(N)
        for c in range(N)
        if colours[r, c] == ""
    ]

    for k, (r, c) in enumerate(rest):
        colours[r, c] = groups[2][1] if k < q[2] else groups[3][1]

    rr, cc = np.divmod(np.arange(N * N), N)
    ax.scatter(cc, -rr, c=list(colours.ravel()), s=36, marker="s", linewidths=0)

    for i, (n, colour, label) in enumerate(groups):
        ax.text(N + 1.5, -3 - 3 * i, f"{n:,}  {label}", color=colour, fontsize=12, va="center")

    ax.set(xlim=(-1, N + 24), ylim=(-N - 5, 1))
    ax.set_aspect("equal")
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Question 1 — Cell 1
# ---------------------------------------------------------------------------

def question1_population_comparison(population=10000, prevalence=0.01, sensitivity=0.80, specificity=0.95, step=0.04):
    population = int(population)
    N = 50
    per_square = population / N**2

    base = {
        "prevalence": prevalence,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

    red = "#b52222"
    orange = "#e5a23b"
    blue = "#1766b5"
    grey = "#b0b0b0"

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(17, 16))

    for ax, changed in zip(axes.flat, [None, "prevalence", "sensitivity", "specificity"]):
        s = dict(base)

        if changed:
            s[changed] = min(1, s[changed] + step)

        n_inf = s["prevalence"] * population
        tp = round(n_inf * s["sensitivity"])
        fn = round(n_inf * (1 - s["sensitivity"]))
        fp = round((population - n_inf) * (1 - s["specificity"]))
        tn = population - tp - fn - fp

        label = changed or "base scenario"
        results[label] = (s, tp, fn, fp, tn)

        population_grid(ax,
            [
                (tp, red, "infected, positive"),
                (fn, orange, "infected, negative"),
                (fp, blue, "not infected, positive"),
                (tn, grey, "not infected, negative"),
            ],
            per_square,
        )

        parts = [
            (f'P(I) = {s["prevalence"]:g}', "prevalence"),
            (f'sens {s["sensitivity"]:.2f}', "sensitivity"),
            (f'spec {s["specificity"]:.3g}', "specificity"),
        ]

        title = "     ".join(text for text, name in parts)

        prefix = ("base scenario:   " if not changed else f"{changed} + {step:g}:   ")

        ax.set_title(prefix + title, fontsize=12, loc="left")

        ax.text(0, -N - 1.5,
            (
                f"positive results: {tp:,} infected + {fp:,} not "
                f"= {tp + fp:,}        P(I | +) = {tp / (tp + fp):.3f}"
            ),
            fontsize=13, va="top",
        )

    fig.suptitle(
        f"population {population:,}, one square = {per_square:g}", fontsize=12, color="grey")
    fig.tight_layout()

    rows = []
    for label, (s, tp, fn, fp, tn) in results.items():
        rows.append(
            {
                "scenario": label,
                "P(I)": s["prevalence"],
                "sensitivity": s["sensitivity"],
                "specificity": s["specificity"],
                "TP": tp,
                "FP": fp,
                "P(I|+) counts": tp / (tp + fp),
                "exact": posterior(
                    s["prevalence"],
                    s["sensitivity"],
                    s["specificity"],
                ),
            }
        )

    return {"figure": fig, "rows": rows}


# ---------------------------------------------------------------------------
# Question 1 — Cell 2
# ---------------------------------------------------------------------------

def question1_parameter_curves(prevalence=0.01, sensitivity=0.80, specificity=0.95, step=0.04):
    base = {
        "prevalence": prevalence,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    grids = [
        ("prevalence", np.logspace(-3, np.log10(0.6), 400)),
        ("sensitivity", np.linspace(0.5, 1, 400)),
        ("specificity", np.linspace(0.90, 1, 400)),
    ]

    for ax, (name, grid) in zip(axes, grids):
        ax.plot(grid, [posterior(**{**base, name: v}) for v in grid], color="#003bd1")

        ax.plot([base[name]], [posterior(**base)], "ko", label="base scenario")

        value = min(1, base[name] + step)

        ax.plot([value], [posterior(**{**base, name: value})], "o", color="#b52222", label=f"{name} + {step:g}")

        ax.set(xlabel=name, ylim=(0, 1), ylabel="P(I | +)")
        ax.legend(fontsize=8)

        if name == "prevalence":
            ax.set_xscale("log")
            ax.set_xticks([0.001, 0.01, 0.1])
            ax.set_xticklabels(["0.001", "0.01", "0.1"])

    fig.suptitle(
        "Which input moves P(I | +) most? "
        "(the other two inputs held at the base values)"
    )
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Question 1 — Cell 3
# ---------------------------------------------------------------------------

def question1_repeated_tests(population=10000, prevalence=0.01, sensitivity=0.80, specificity=0.95):
    population = int(population)
    N = 50
    per_square = population / N**2

    red = "#b52222"
    orange = "#e5a23b"
    blue = "#1766b5"
    grey = "#b0b0b0"

    n_inf = round(prevalence * population)
    n_not = population - n_inf

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    for ax, tests in zip(axes, (1, 2)):
        tp = round(n_inf * sensitivity**tests)
        fp = round(n_not * (1 - specificity) ** tests)

        word = "positive" if tests == 1 else "positive twice"

        population_grid(ax,
            [
                (tp, red, f"infected, {word}"),
                (n_inf - tp, orange, f"infected, not {word}"),
                (fp, blue, f"not infected, {word}"),
                (n_not - fp, grey, f"not infected, not {word}"),
            ],
            per_square,
        )

        ax.set_title(
            (
                f'{tests} test{"s" if tests > 1 else ""}:   '
                f"P(I) = {prevalence:g}     "
                f"sens {sensitivity:.2f}     "
                f"spec {specificity:.3g}"
            ),
            fontsize=12, loc="left",
        )

        plus = "+" if tests == 1 else "+,+"

        ax.text(
            0, -N - 1.5,
            (
                f"{word}: {tp:,} infected + {fp:,} not "
                f"= {tp + fp:,}        P(I | {plus}) = {tp / (tp + fp):.3f}"
            ),
            fontsize=13, va="top",
        )

    fig.suptitle(
        f"population {population:,}, one square = {per_square:g}", fontsize=12, color="grey")
    fig.tight_layout()

    rows = []

    for results in ["+", "++", "+-", "+++"]:
        a = np.prod(
            [
                sensitivity if result == "+" else 1 - sensitivity
                for result in results
            ]
        )
        b = np.prod(
            [
                1 - specificity if result == "+" else specificity
                for result in results
            ]
        )

        tp = round(n_inf * a)
        fp = round(n_not * b)

        from_counts = (tp / (tp + fp) if tp + fp else float("nan"))

        exact = prevalence * a / (prevalence * a + (1 - prevalence) * b)

        rows.append(
            {
                "results": results,
                "P(results | I)": a,
                "P(results | not I)": b,
                "from the counts": from_counts,
                "exact": exact,
            }
        )

    return {"figure": fig, "rows": rows}


# ---------------------------------------------------------------------------
# Question 2 — Cell 4
# ---------------------------------------------------------------------------

def question2_playlist_experiment(
    songs=20,
    by_artist=4,
    played=10,
    target=2,
    runs=10000,
    seed=2029,
):

    songs = int(songs)
    by_artist = int(by_artist)
    played = int(played)
    target = int(target)
    runs = int(runs)

    if runs > 10000:
        raise ValueError("It generates 10,000 runs. Choose runs <= 10,000")

    total = 10000
    p = by_artist / songs

    colours = {
        1: ("#8fa8e8", "#003bd1"),
        2: ("#f0b4a0", "#b52222"),
    }
    player_names = {
        1: "player A",
        2: "player B",
    }

    strip = max(40, int(np.ceil(1.5 * played / 10) * 10))
    rows_count = 12

    def build_runs(rng):
        seq_2 = rng.random((total, songs)).argsort(axis=1) < by_artist

        seq_1 = rng.random((total, 4 * songs)) < p
        while np.any(seq_1.sum(axis=1) < target):
            seq_1 = np.concatenate((seq_1, rng.random((total, 2 * songs)) < p), axis=1)

        return seq_1, seq_2

    seq_1, seq_2 = get_sample("runs", (songs, by_artist, played, target, total), seed, build_runs)

    X1, Y1 = measure(seq_1, played, target)
    X2, Y2 = measure(seq_2, played, target)

    m = runs

    YMAX = int(np.ceil(np.quantile(np.concatenate([Y1, Y2]), 0.995) / 10) * 10)

    XLO = int(np.quantile(np.concatenate([X1, X2]), 0.001))
    XHI = int(np.quantile(np.concatenate([X1, X2]), 0.999))

    lo_x = int(np.floor(played * p - 1.5 * np.sqrt(played * p * (1 - p))))
    hi_x = int(np.ceil(played * p + 1.5 * np.sqrt(played * p * (1 - p))))

    tail_y = int(round((target / p + np.sqrt(target * (1 - p)) / p) / 5) * 5)

    # Figure 1: example runs
    sq = max(4, min(22, int(2600 / strip)))

    strip_fig, axes = plt.subplots(2, 1, figsize=(16, 2 * (0.42 * rows_count + 0.9)), sharex=True)

    for ax, (method, seq) in zip(axes, [(1, seq_1), (2, seq_2)]):
        for row in range(rows_count):
            visible_seq = seq[row][:strip]

            X = int(seq[row][:played].sum())
            Y = int(np.flatnonzero(seq[row])[target - 1] + 1)

            ax.scatter(
                np.arange(1, len(visible_seq) + 1),
                np.full(len(visible_seq), -row),
                s=sq,
                c=np.where(
                    visible_seq,
                    "#1db954",
                    "#dddddd",
                ),
                marker="s",
                linewidths=0,
            )

            if Y <= strip:
                ax.plot(Y, -row, "o", mfc="none", mec=colours[method][1], ms=max(6, sq**0.5 * 2.4), mew=1.5)

            ax.text(1.005, -row, f"X = {X:<4d} Y = {Y}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)

        ax.axvspan(0.5, min(played, strip) + 0.5, color="#fff2b0", zorder=0)

        ax.set(ylabel=player_names[method], xlim=(0.5, strip + 0.5), ylim=(-rows_count + 0.4, 0.6))
        ax.set_yticks(-np.arange(rows_count), np.arange(1, rows_count + 1))

    axes[1].set_xlabel(f"song (first {strip} of each run)")

    strip_fig.suptitle(
        "green: a song by the artist; "
        "yellow: the songs counted in X; "
        "ring: the song counted by Y"
    )

    strip_fig.subplots_adjust(right=0.86, top=0.9, bottom=0.12, hspace=0.25)

    # Exact quantities and PMFs
    exact = {
        1: (played * p, played * p * (1 - p), target / p),
        2: (played * p, played * p * (1 - p) * (songs - played) / (songs - 1), target * (songs + 1) / (by_artist + 1)),
    }

    pmfs = {
        1: (
            lambda x: binomial_pmf(x, played, p),
            lambda y: neg_binomial_pmf(y, target, p),
            f"B({played}, {p:g})",
            f"NB({target}, {p:g})"
        ),
        2: (
            lambda x: hypergeometric_pmf(x, songs, by_artist, played),
            lambda y: neg_hypergeometric_pmf(y, songs, by_artist, target),
            f"HG({songs}, {by_artist}, {played})",
            f"NHG({songs}, {by_artist}, {target})"
        ),
    }

    # Figure 2: simulated distributions vs exact PMFs
    distribution_fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))

    def paired(ax, data, xs, which):
        for k, (method, values) in enumerate(data):
            light, dark = colours[method]

            freq = [(values[:m] == x).mean() for x in xs]

            ax.bar(xs + (k - 0.5) * 0.4, freq, width=0.4, color=light, label=f"{player_names[method]}, {m:,} runs")
            ax.plot(xs, [pmfs[method][which](int(x)) for x in xs], "o", color=dark, ms=4, label=pmfs[method][which + 2])

        ax.set_ylabel("fraction of runs")

    xs = np.arange(max(0, XLO - 1), min(played, XHI + 1) + 1)

    paired(axes[0], [(1, X1), (2, X2)], xs, 0)
    axes[0].set(xlabel="X")
    axes[0].legend(fontsize=8)

    ys = np.arange(target, YMAX + 1)

    paired(axes[1], [(1, Y1), (2, Y2)], ys, 1)
    axes[1].set(xlabel="Y", xlim=(target - 1, YMAX + 1))
    axes[1].legend(fontsize=8)

    distribution_fig.tight_layout()

    headers = ["X=0" if lo_x == 0 else f"X<={lo_x}", f"X>={hi_x}", f"X>={target}", f"Y<={played}", f"Y>{tail_y}"]

    summary_rows = []

    for method, X, Y in [(1, X1, Y1), (2, X2, Y2)]:
        EX, VX, EY = exact[method]
        Xm = X[:m]
        Ym = Y[:m]

        fx = pmfs[method][0]
        fy = pmfs[method][1]

        simulated = [
            (Xm <= lo_x).mean(),
            (Xm >= hi_x).mean(),
            (Xm >= target).mean(),
            (Ym <= played).mean(),
            (Ym > tail_y).mean(),
        ]

        exact_probabilities = [
            sum(fx(x) for x in range(0, lo_x + 1)),
            sum(fx(x) for x in range(hi_x, played + 1)),
            sum(fx(x) for x in range(target, played + 1)),
            sum(fy(y) for y in range(target, played + 1)),

            1 - sum(fy(y) for y in range(target, tail_y + 1))
        ]

        summary_rows.append(
            {
                "player": player_names[method],
                "type": "simulated",
                "mean X": Xm.mean(),
                "var X": Xm.var(),
                "mean Y": Ym.mean(),
                **dict(zip(headers, simulated)),
            }
        )

        summary_rows.append(
            {
                "player": player_names[method],
                "type": "exact",
                "mean X": EX,
                "var X": VX,
                "mean Y": EY,
                **dict(zip(headers, exact_probabilities)),
            }
        )

    first_b = seq_2[0][:played]
    before = np.r_[0, np.cumsum(first_b)[:-1]]
    next_probabilities = (by_artist - before) / (songs - np.arange(played))

    return {
        "strip_figure": strip_fig,
        "distribution_figure": distribution_fig,
        "summary_rows": summary_rows,
        "event_headers": headers,
        "player_b_first_run": first_b,
        "player_b_next_probabilities": next_probabilities,
        "player_a_next_probability": p,
        "X1": X1,
        "Y1": Y1,
        "X2": X2,
        "Y2": Y2,
    }


# ---------------------------------------------------------------------------
# Question 2 — Cell 5
# ---------------------------------------------------------------------------

def question2_size_comparison(lengths=(20, 40, 200, 2000), played=10, fraction_by_artist=0.2, seed=2029):
    sizes = [int(value) for value in lengths]
    n = int(played)
    p = float(fraction_by_artist)
    repeats = 10000

    colours = {
        1: ("#8fa8e8", "#003bd1"),
        2: ("#f0b4a0", "#b52222"),
    }

    player_names = {
        1: "player A",
        2: "player B",
    }

    def build_sizes(rng):
        return {
            N: (
                rng.binomial(
                    n,
                    p,
                    size=repeats,
                ),
                rng.hypergeometric(
                    round(N * p),
                    N - round(N * p),
                    n,
                    size=repeats,
                ),
            )
            for N in sizes
        }

    counts = get_sample("lengths", tuple(sizes) + (n, p, repeats), seed, build_sizes)

    fig, axes = plt.subplots(1, len(sizes), figsize=(3.4 * len(sizes), 3.6), sharey=True, squeeze=False)

    all_x = np.concatenate([values for pair in counts.values() for values in pair])

    xs = np.arange(
        max(0, int(np.quantile(all_x, 0.001)) - 1),
        min(n, int(np.quantile(all_x, 0.999)) + 1) + 1
    )

    rows = []

    for ax, N in zip(axes[0], sizes):
        K = round(N * p)

        options = [
            (
                1,
                counts[N][0],
                lambda x: binomial_pmf(x, n, p),
                f"B({n}, {p:g})",
            ),
            (
                2,
                counts[N][1],
                lambda x, N=N, K=K: hypergeometric_pmf(x, N, K, n),
                f"HG({N}, {K}, {n})",
            ),
        ]

        for k, (method, values, pmf, name) in enumerate(options):
            light, dark = colours[method]

            ax.bar(xs + (k - 0.5) * 0.4, [(values == x).mean() for x in xs], width=0.4, color=light, label=player_names[method])
            ax.plot( xs, [pmf(int(x)) for x in xs], "o", color=dark, ms=4, label=name)

        ax.set(title=f"{N:,} songs", xlabel="X")
        ax.legend(fontsize=7)

        rows.append(
            {
                "songs": N,
                "var X, player A": counts[N][0].var(),
                "exact B": n * p * (1 - p),
                "var X, player B": counts[N][1].var(),
                "exact HG": (n * p * (1 - p) * (N - n) / (N - 1)),
            }
        )

    axes[0][0].set_ylabel("fraction of runs")
    fig.tight_layout()

    return {"figure": fig, "rows": rows}


# ---------------------------------------------------------------------------
# Question 3 — Cell 6
# ---------------------------------------------------------------------------

def question3_registration(rate=2.0, seats=60, time_cutoff=30.0, seed=2029):
    rate = float(rate)
    seats = int(seats)
    t_cut = float(time_cutoff)

    span = max(t_cut, 1.5 * seats / rate)

    arrival_slots, horizon = get_sample(
        "arrivals",
        (rate, seats, span),
        seed,
        lambda rng: build_arrivals(rng, rate, seats, horizon=span),
    )

    T = arrival_slots[:, seats - 1] / 1000

    TMAX = float(np.ceil(max(t_cut * 1.1, np.quantile(T, 0.999)) / 5) * 5)

    R_cut = (arrival_slots <= round(t_cut * 1000)).sum(axis=1)

    open_at_cut = R_cut < seats

    # Figure 1: example opening timelines
    timeline_fig, ax = plt.subplots(figsize=(12, 5))

    for row in range(15):
        colour = ("#003bd1" if open_at_cut[row] else "#999999")

        visible = arrival_slots[row] / 1000
        visible = visible[visible <= TMAX]

        ax.hlines(-row, 0, TMAX, color=colour, alpha=0.3, lw=1)

        ax.scatter(visible, np.full(len(visible), -row), color=colour, s=8)

        if T[row] <= TMAX:
            ax.plot( T[row], -row, "*", color="#b52222", ms=11)

        ax.text(1.01, -row, f"R(t) = {R_cut[row]:<4d}", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=colour)

    ax.axvline(t_cut, color="k", ls="--", lw=1)

    ax.text(t_cut, 1.2, f"t = {t_cut:g}", ha="center")

    ax.set(
        yticks=-np.arange(15),
        yticklabels=np.arange(1, 16),
        xlabel="seconds after opening",
        ylabel="opening",
        xlim=(0, TMAX),
        ylim=(-14.7, 2),
    )

    ax.set_title(
        f"rate {rate:g} per second, {seats} seats; "
        f"dots: requests; star: request {seats}"
    )

    timeline_fig.subplots_adjust(right=0.88, top=0.9, bottom=0.12)

    # Figure 2: distribution of T(seats) and seat-open probability
    summary_fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

    bins = np.linspace(0, TMAX, 61)

    mask = T <= TMAX

    axes[0].hist(T[mask], bins=bins, weights=np.full(mask.sum(), 1 / len(T)), color="#c5cfe6")

    axes[0].axvline(T.mean(), color="#b52222", lw=1.2, label=f"mean {T.mean():.2f} s")

    if (T > TMAX).any():
        axes[0].text(
            0.98,
            0.93,
            f"{(T > TMAX).mean():.1%} beyond {TMAX:g} s",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    axes[0].set(
        xlabel=f"T({seats}), seconds until the course is full",
        ylabel="fraction of openings",
        xlim=(0, TMAX),
    )
    axes[0].legend(fontsize=9)

    grid = np.linspace(0, TMAX, 300)

    axes[1].plot(
        grid,
        [(T > t).mean() for t in grid],
        color="#c5cfe6",
        lw=4,
        label=(
            f"fraction of openings "
            f"with T({seats}) > t"
        ),
    )

    axes[1].plot(
        grid,
        [seat_open_probability(t, seats, rate) for t in grid],
        color="#003bd1",
        lw=1.3,
        label=f"P(R(t) < {seats})",
    )

    axes[1].axvline(t_cut, color="k", ls="--", lw=1)
    axes[1].set(xlabel="t, seconds after opening", ylabel="probability a seat is still open", xlim=(0, TMAX), ylim=(0, 1.02))
    axes[1].legend(fontsize=9)

    summary_fig.tight_layout()

    times = sorted({t_cut, 20.0, 30.0, 40.0})

    rows = [
        {
            "quantity": f"mean of T({seats})",
            "simulated": T.mean(),
            "exact": seats / rate,
        },
        {
            "quantity": f"standard deviation of T({seats})",
            "simulated": T.std(),
            "exact": None,
        },
    ]

    for t in times:
        rows.append(
            {
                "quantity": f"seat open at t = {t:g}",
                "simulated": (T > t).mean(),
                "exact": seat_open_probability(t, seats, rate),
            }
        )

    return {
        "timeline_figure": timeline_fig,
        "summary_figure": summary_fig,
        "rows": rows,
        "T": T,
        "R_cut": R_cut,
        "open_at_cut": open_at_cut,
        "time_cutoff": t_cut,
    }


plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
