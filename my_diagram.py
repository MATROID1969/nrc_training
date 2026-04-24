"""
diagram.py
----------
Speciális diagram-típusokat rajzoló függvények gyűjteménye.

Függvények
----------
- grouped_bar_chart   : csoportosított oszlopdiagram háttér-összeg oszloppal
- bubble_matrix       : buborékdiagram (mátrix elrendezés)
- lollypop_chart      : lollipop diagram
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Csoportosított oszlopdiagram (grouped bar chart with total background)
# ---------------------------------------------------------------------------


def grouped_bar_chart(
    df: pd.DataFrame,
    category_field: str,
    group_field: str,
    title: str = "Grouped Bar Chart",
) -> plt.Figure:
    """
    Csoportosított oszlopdiagram: az x tengelyen a category_field kategóriái
    jelennek meg, az egyes oszlopcsoportokon belül a group_field értékeinek
    darabszáma látható. A szürke háttéroszlop az adott x kategória teljes N-jét mutatja.

    Paraméterek
    ----------
    df : pd.DataFrame
        Bármilyen nyers DataFrame.
    category_field : str
        Az x tengelyen megjelenő kategóriák oszlopának neve.
    group_field : str
        Az egyes x kategórián belüli eloszlást mutató oszlop neve.
    title : str
        A diagram fejléce.

    Visszatér
    ----------
    matplotlib.figure.Figure

    Példa
    -----
    >>> fig = grouped_bar_chart(df, "vegz", "nem", title="Nem megoszlása végzettség szerint")
    >>> fig.show()
    """
    # --- validálás ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("A bemenetnek pandas DataFrame-nek kell lennie.")
    for field in [category_field, group_field]:
        if field not in df.columns:
            raise ValueError(f"A '{field}' oszlop nem található a DataFrame-ben.")

    MAX_CATEGORIES = 30
    for field in [category_field, group_field]:
        n_unique = df[field].nunique()
        if n_unique > MAX_CATEGORIES:
            raise ValueError(
                f"A '{field}' oszlop {n_unique} egyedi értéket tartalmaz. "
                f"Diagramhoz max {MAX_CATEGORIES} kategória javasolt."
            )

    # --- kereszttábla: category_field sorai, group_field oszlopai ---
    ct = df.groupby([category_field, group_field]).size().unstack(fill_value=0)
    totals = ct.sum(axis=1)

    x_labels = ct.index.astype(str).tolist()
    group_labels = ct.columns.astype(str).tolist()
    n_groups = len(group_labels)
    x = np.arange(len(x_labels))
    width = min(0.7 / n_groups, 0.22)

    base_colors = [
        "#1f5d6b",
        "#2f8f9d",
        "#7fcad3",
        "#4fa3a5",
        "#9fd3da",
        "#15616d",
    ]

    # --- rajzolás ---
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 1.2), 6))

    # Szürke háttéroszlop = az adott x kategória teljes N-je
    ax.bar(x, totals.values, width=0.85, color="#e6e6e6", zorder=0)

    # Csoportosított oszlopok group_field értékenként
    for i, grp in enumerate(group_labels):
        offset = (i - (n_groups - 1) / 2) * width
        vals = ct[ct.columns[i]].values
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=grp,
            color=base_colors[i % len(base_colors)],
            zorder=3,
        )
        # Értékfeliratok az oszlopokon
        for xi, val in zip(x, vals):
            if val > 0:
                ax.text(
                    xi + offset,
                    val + totals.max() * 0.01,
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=base_colors[i % len(base_colors)],
                )

    # Teljes N feliratok a háttéroszlopokon
    for i, (xi, tot) in enumerate(zip(x, totals.values)):
        ax.text(
            xi,
            tot - totals.max() * 0.02,
            f"N={tot}",
            ha="center",
            va="top",
            fontsize=9,
            color="#999999",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Darabszám (N)")
    ax.set_title(title)
    ax.set_ylim(0, totals.max() * 1.12)
    ax.legend(title=group_field)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 2. Buborékdiagram (bubble matrix)
# ---------------------------------------------------------------------------


def bubble_matrix(
    df: pd.DataFrame, x_field: str, y_field: str, title: str = "Bubble Matrix"
) -> plt.Figure:
    """
    Buborékdiagram mátrix-elrendezésben: az x_field és y_field kategória-párok
    előfordulási darabszámát ábrázolja buborék méretével és feliratával.

    Bármilyen nyers DataFrame-en működik – a függvény megszámolja, hány sor
    tartozik minden (x_field, y_field) kombinációhoz.

    Paraméterek
    ----------
    df : pd.DataFrame
        Bármilyen nyers DataFrame.
    x_field : str
        Az x tengelyen megjelenő kategóriák oszlopának neve.
    y_field : str
        Az y tengelyen megjelenő kategóriák oszlopának neve.
    title : str
        A diagram fejléce.

    Visszatér
    ----------
    matplotlib.figure.Figure

    Példa
    -----
    >>> fig = bubble_matrix(df, x_field="nem", y_field="vegz",
    ...                     title="Nem × Végzettség")
    >>> fig.show()
    """
    # --- validálás ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("A bemenetnek pandas DataFrame-nek kell lennie.")
    for field in [x_field, y_field]:
        if field not in df.columns:
            raise ValueError(f"A '{field}' oszlop nem található a DataFrame-ben.")

    MAX_CATEGORIES = 30
    for field in [x_field, y_field]:
        n_unique = df[field].nunique()
        if n_unique > MAX_CATEGORIES:
            raise ValueError(
                f"A '{field}' oszlop {n_unique} egyedi értéket tartalmaz. "
                f"Diagramhoz max {MAX_CATEGORIES} kategória javasolt."
            )

    # --- darabszámok kiszámítása ---
    counts = df.groupby([x_field, y_field]).size().reset_index(name="_count")

    x_categories = sorted(df[x_field].dropna().unique().tolist())
    y_categories = sorted(df[y_field].dropna().unique().tolist())

    x_idx = {v: i for i, v in enumerate(x_categories)}
    y_idx = {v: i for i, v in enumerate(y_categories)}

    records = [
        (x_idx[row[x_field]], y_idx[row[y_field]], int(row["_count"]))
        for _, row in counts.iterrows()
    ]

    values = np.array([r[2] for r in records])
    max_val = values.max() if values.max() > 0 else 1
    min_area, max_area = 100, 2200
    sizes = min_area + (values / max_val) * (max_area - min_area)

    # --- rajzolás ---
    fig, ax = plt.subplots(
        figsize=(max(6, len(x_categories) * 1.2), max(4, len(y_categories) * 0.8))
    )

    xs = [r[0] for r in records]
    ys = [r[1] for r in records]

    ax.scatter(xs, ys, s=sizes, color="#5dade2", alpha=0.6, edgecolors="none")

    for xi, yi, val in records:
        ax.text(xi, yi, str(val), ha="center", va="center", fontsize=9)

    ax.set_xticks(range(len(x_categories)))
    ax.set_xticklabels(x_categories)
    ax.set_yticks(range(len(y_categories)))
    ax.set_yticklabels(y_categories)

    ax.set_xlim(-0.6, len(x_categories) - 0.4)
    ax.set_ylim(-0.6, len(y_categories) - 0.4)
    ax.invert_yaxis()

    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title(title)

    ax.set_axisbelow(True)
    ax.grid(True, linewidth=1, alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 3. Lollipop diagram
# ---------------------------------------------------------------------------


def lollypop_chart(
    df: pd.DataFrame, category_field: str, title: str = "Lollipop Chart"
) -> plt.Figure:
    """
    Lollipop diagram: a category_field kategóriánkénti darabszámát ábrázolja.

    Bármilyen nyers DataFrame-en működik – a függvény megszámolja az előfordulásokat,
    és növekvő sorrendbe rendezi őket.

    Paraméterek
    ----------
    df : pd.DataFrame
        Bármilyen nyers DataFrame.
    category_field : str
        A kategóriacímkéket tartalmazó oszlop neve (y tengely).
    title : str
        A diagram fejléce.

    Visszatér
    ----------
    matplotlib.figure.Figure

    Példa
    -----
    >>> fig = lollypop_chart(df, "vegz", title="Megoszlás végzettség szerint")
    >>> fig.show()
    """
    # --- validálás ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("A bemenetnek pandas DataFrame-nek kell lennie.")
    if category_field not in df.columns:
        raise ValueError(f"A '{category_field}' oszlop nem található a DataFrame-ben.")

    MAX_CATEGORIES = 30
    n_unique = df[category_field].nunique()
    if n_unique > MAX_CATEGORIES:
        raise ValueError(
            f"A '{category_field}' oszlop {n_unique} egyedi értéket tartalmaz. "
            f"Diagramhoz max {MAX_CATEGORIES} kategória javasolt."
        )

    # --- darabszámok kiszámítása, növekvő sorrend ---
    counts = df[category_field].value_counts().sort_values().reset_index()
    counts.columns = ["_category", "_count"]

    line_color = "#bdbdbd"
    dot_color = "#f28e2b"

    # --- rajzolás ---
    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.45)))

    ax.hlines(
        y=counts["_category"],
        xmin=0,
        xmax=counts["_count"],
        color=line_color,
        linewidth=2,
    )

    ax.scatter(counts["_count"], counts["_category"], color=dot_color, s=80, zorder=3)

    ax.set_xlim(0, counts["_count"].max() * 1.1)
    ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.5)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#999999")

    ax.set_xlabel("Darabszám (N)")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14, weight="bold")

    ax.tick_params(axis="y", length=0, labelsize=12)
    ax.tick_params(axis="x", colors="#666666")

    fig.tight_layout()
    plt.close(fig)
    return fig
