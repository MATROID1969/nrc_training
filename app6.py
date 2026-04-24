import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ast

from my_diagram import grouped_bar_chart, bubble_matrix


st.set_page_config(
    page_title="Életkor dashboard",
    layout="wide"
)

st.title("Életkor dashboard")

df1 = pd.read_excel("streamlit_base.xlsx")


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
            except:
                value_list = []

            for item in value_list:
                item = str(item).strip()

                if item != "":
                    counts[item] = counts.get(item, 0) + 1

    result_df1 = pd.DataFrame(
        counts.items(),
        columns=["Válasz", "Darabszám"]
    )

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