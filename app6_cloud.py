import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import ast


# ============================================================
# SAJÁT DIAGRAM FÜGGVÉNYEK
# my_diagram.py helyett közvetlenül itt vannak
# ============================================================

def grouped_bar_chart(
    df: pd.DataFrame,
    category_field: str,
    group_field: str,
    title: str = "Grouped Bar Chart",
) -> plt.Figure:

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

    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 1.2), 6))

    ax.bar(x, totals.values, width=0.85, color="#e6e6e6", zorder=0)

    for i, grp in enumerate(group_labels):
        offset = (i - (n_groups - 1) / 2) * width
        vals = ct[ct.columns[i]].values

        ax.bar(
            x + offset,
            vals,
            width,
            label=grp,
            color=base_colors[i % len(base_colors)],
            zorder=3,
        )

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

    for xi, tot in zip(x, totals.values):
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

    if totals.max() > 0:
        ax.set_ylim(0, totals.max() * 1.12)

    ax.legend(title=group_field)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    plt.close(fig)

    return fig


def bubble_matrix(
    df: pd.DataFrame,
    x_field: str,
    y_field: str,
    title: str = "Bubble Matrix",
) -> plt.Figure:

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

    max_val = values.max() if len(values) > 0 and values.max() > 0 else 1
    min_area, max_area = 100, 2200
    sizes = min_area + (values / max_val) * (max_area - min_area)

    fig, ax = plt.subplots(
        figsize=(max(6, len(x_categories) * 1.2), max(4, len(y_categories) * 0.8))
    )

    xs = [r[0] for r in records]
    ys = [r[1] for r in records]

    ax.scatter(
        xs,
        ys,
        s=sizes,
        color="#5dade2",
        alpha=0.6,
        edgecolors="none",
    )

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


def lollypop_chart(
    df: pd.DataFrame,
    category_field: str,
    title: str = "Lollipop Chart",
) -> plt.Figure:

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

    counts = df[category_field].value_counts().sort_values().reset_index()
    counts.columns = ["_category", "_count"]

    line_color = "#bdbdbd"
    dot_color = "#f28e2b"

    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.45)))

    ax.hlines(
        y=counts["_category"],
        xmin=0,
        xmax=counts["_count"],
        color=line_color,
        linewidth=2,
    )

    ax.scatter(
        counts["_count"],
        counts["_category"],
        color=dot_color,
        s=80,
        zorder=3,
    )

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


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Életkor dashboard",
    layout="wide"
)

st.title("Életkor dashboard")

df1 = pd.read_excel("streamlit_base.xlsx")

# Oszlopnevek tisztítása
df1.columns = df1.columns.astype(str).str.strip()


# ------------------------------------------------------------
# Segédfüggvény: lista típusú mezők felismerése
# ------------------------------------------------------------
def is_list_column(series):
    non_null_values = series.dropna().astype(str)

    if non_null_values.empty:
        return False

    first_val = non_null_values.iloc[0].strip()

    return first_val.startswith("[") and first_val.endswith("]")


# ------------------------------------------------------------
# Segédfüggvény: listás mező válaszainak megszámolása
# ------------------------------------------------------------
def count_list_values(df1, column_name):
    counts = {}

    for value in df1[column_name].dropna():

        value_str = str(value).strip()

        if value_str.startswith("[") and value_str.endswith("]"):

            try:
                value_list = ast.literal_eval(value_str)
            except Exception:
                value_list = []

            for item in value_list:
                item = str(item).strip()

                if item != "":
                    counts[item] = counts.get(item, 0) + 1

    result_df1 = pd.DataFrame(
        counts.items(),
        columns=["Válasz", "Darabszám"]
    )

    if not result_df1.empty:
        result_df1 = result_df1.sort_values(
            "Darabszám",
            ascending=True
        )

    return result_df1


# Fülek
tab1, tab2, tab3 = st.tabs([
    "Életkor hisztogram",
    "Kategória diagramok",
    "Lista változók"
])


# ============================================================
# 1. FÜL – HISZTOGRAM
# ============================================================
with tab1:

    filter_col, chart_col = st.columns([1.5, 3])

    with filter_col:
        with st.container(border=True):
            st.subheader("Szűrők")

            veg_options = sorted(df1["Végzettség"].dropna().unique())
            nem_options = sorted(df1["Nem"].dropna().unique())

            veg = st.multiselect(
                "Végzettség",
                veg_options,
                default=veg_options
            )

            nem = st.multiselect(
                "Nem",
                nem_options,
                default=nem_options
            )

            bins = st.slider(
                "Hisztogram oszlopok száma",
                min_value=5,
                max_value=50,
                value=20
            )

    filtered_df1 = df1[
        (df1["Végzettség"].isin(veg)) &
        (df1["Nem"].isin(nem))
    ]

    with chart_col:
        with st.container(border=True):
            st.subheader("Diagramok")

            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=df1["Életkor"],
                    nbinsx=bins,
                    marker=dict(
                        color="lightgrey",
                        line=dict(color="grey", width=1)
                    ),
                    opacity=0.6,
                    name="Teljes sokaság"
                )
            )

            fig.add_trace(
                go.Histogram(
                    x=filtered_df1["Életkor"],
                    nbinsx=bins,
                    marker=dict(
                        color="blue",
                        line=dict(color="grey", width=1)
                    ),
                    opacity=0.6,
                    name="Szűrt"
                )
            )

            fig.update_layout(barmode="overlay")

            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2. FÜL – KATEGÓRIA DIAGRAMOK
# ============================================================
with tab2:

    filter_col2, chart_col2 = st.columns([1.5, 3])

    with filter_col2:
        with st.container(border=True):
            st.subheader("Diagram beállítások")

            category_columns = []

            for col in df1.columns:
                if df1[col].dtype in ["object", "category"]:
                    if not is_list_column(df1[col]):
                        category_columns.append(col)

            category_1 = st.selectbox(
                "Első kategória változó",
                category_columns,
                index=0
            )

            category_2 = st.selectbox(
                "Második kategória változó",
                category_columns,
                index=1 if len(category_columns) > 1 else 0
            )

            diagram_type = st.radio(
                "Diagram típusa",
                ["oszlop", "buborék"],
                index=0
            )

    with chart_col2:
        with st.container(border=True):
            st.subheader("Kategória diagram")

            try:
                if diagram_type == "oszlop":
                    fig2 = grouped_bar_chart(
                        df1,
                        category_field=category_1,
                        group_field=category_2,
                        title=f"{category_2} megoszlása {category_1} szerint"
                    )
                else:
                    fig2 = bubble_matrix(
                        df1,
                        x_field=category_1,
                        y_field=category_2,
                        title=f"{category_1} × {category_2}"
                    )

                st.pyplot(fig2)

            except Exception as e:
                st.error("Nem sikerült megrajzolni a diagramot.")
                st.exception(e)


# ============================================================
# 3. FÜL – LISTA VÁLTOZÓK
# ============================================================
with tab3:

    filter_col3, chart_col3 = st.columns([1.5, 3])

    with filter_col3:
        with st.container(border=True):
            st.subheader("Lista változó beállítások")

            list_columns = []

            for col in df1.columns:
                if df1[col].dtype in ["object", "category"]:
                    if is_list_column(df1[col]):
                        list_columns.append(col)

            selected_list_col = st.selectbox(
                "Lista típusú változó",
                list_columns,
                index=0
            )

    with chart_col3:
        with st.container(border=True):
            st.subheader("Lista válaszok gyakorisága")

            list_counts_df1 = count_list_values(
                df1,
                selected_list_col
            )

            fig3 = go.Figure()

            fig3.add_trace(
                go.Bar(
                    x=list_counts_df1["Darabszám"],
                    y=list_counts_df1["Válasz"],
                    orientation="h",
                    marker=dict(
                        color="steelblue",
                        line=dict(color="grey", width=1)
                    )
                )
            )

            fig3.update_layout(
                xaxis_title="Darabszám",
                yaxis_title="Válasz",
                height=max(500, len(list_counts_df1) * 30)
            )

            st.plotly_chart(fig3, use_container_width=True)

            st.dataframe(list_counts_df1, use_container_width=True)