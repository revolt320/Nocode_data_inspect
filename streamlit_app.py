"""Streamlit Data Inspector - Optimized Version."""

import re
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

# --- Constants ---
NUM_REGEX = r"^\d+\.?\d*$"
SPACE_PATTERNS = {
    "trailing": r"\s$",
    "leading": r"^\s",
    "extra": r"\s{2,}",
    "missing": r"\S\S",
}


# --- Page setup ---
st.set_page_config(page_title="Data Inspector", layout="wide")


# --- Helper functions ---
def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Load dataframe from uploaded file."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype="object")
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, dtype="object")
    else:
        raise ValueError("Unsupported file format")


def column_conditions(column: str, df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Apply column-based conditions to dataframe."""
    condition_map = {
        "is null": lambda: df[df[column].isnull()],
        "is not null": lambda: df[df[column].notnull()],
        "is duplicated": lambda: df[df.duplicated(subset=column)],
        "drop duplicates": lambda: df.drop_duplicates(subset=column),
    }
    return condition_map.get(condition, lambda: df)()


def regex_conditions(fields: List[str], df: pd.DataFrame, regex_type: str) -> pd.DataFrame:
    """Apply regex-based conditions to specified fields."""
    if not fields:
        return df

    indices = set()

    for field in fields:
        field_series = df[field].astype(str)
        
        if regex_type == "is not null":
            indices.update(df[df[field].notnull()].index)
        elif regex_type == "extra spaces":
            indices.update(df[field_series.str.contains(SPACE_PATTERNS["extra"], regex=True, na=False)].index)
        elif regex_type == "leading or trailing spaces":
            pattern = f'{SPACE_PATTERNS["leading"]}|{SPACE_PATTERNS["trailing"]}'
            indices.update(df[field_series.str.contains(pattern, regex=True, na=False)].index)
        elif regex_type == "missing spaces":
            indices.update(df[field_series.str.contains(SPACE_PATTERNS["missing"], regex=True, na=False)].index)

    return df.loc[list(indices)] if indices else df.iloc[0:0]  # Return empty df if no matches


def display_sample_structured(df: pd.DataFrame, n: int = 1) -> None:
    """Display sample(s) in key: value format."""
    if df.empty:
        st.write("No data to display")
        return
        
    sample_df = df.sample(min(n, len(df)))
    for _, row in sample_df.iterrows():
        for col, val in row.items():
            st.markdown(f"**{col}**  \n{val}")
        st.markdown("---")


def spacecheck_ui(dfr: pd.DataFrame, url_column: str) -> pd.DataFrame:
    """Check for space and HTML tag issues in dataframe."""
    results = []
    
    for _, row in dfr.iterrows():
        url_value = row[url_column]
        
        for column_name, value in row.items():
            str_value = str(value)
            
            # Check for various issues
            issues = []
            if re.search(SPACE_PATTERNS["trailing"], str_value):
                issues.append("Trailing space")
            if re.search(SPACE_PATTERNS["leading"], str_value):
                issues.append("Leading space")
            if re.search(SPACE_PATTERNS["extra"], str_value):
                issues.append("Extra spaces")
            if re.search(r"<.*>", str_value):
                issues.append("HTML Tag")
            
            for issue in issues:
                results.append((column_name, issue, str_value, url_value))
    
    return pd.DataFrame(results, columns=["Column", "Issue", "Value", url_column])


def render_basic_info_tab(df: pd.DataFrame, uploaded_file) -> None:
    """Render the Basic Info tab."""
    st.write("**Filename:**", uploaded_file.name if uploaded_file else "No file uploaded")
    st.write("**Number of Rows:**", df.shape[0])
    st.write("**Number of Columns:**", df.shape[1])
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe(include="all").T, use_container_width=True)


def render_preview_tab(df: pd.DataFrame) -> None:
    """Render the Preview tab."""
    st.subheader("Preview Data")
    st.dataframe(df, use_container_width=True)


def render_formatting_checks_tab(df: pd.DataFrame) -> None:
    """Render the Formatting Checks tab."""
    st.subheader("Text Checks for Spaces / HTML Tags")
    url_col = st.selectbox("Select column to display as URL/reference", df.columns)
    
    if st.button("Run Checks"):
        results_df = spacecheck_ui(df, url_col)
        st.dataframe(results_df, use_container_width=True)


def render_unique_values_tab(df: pd.DataFrame) -> None:
    """Render the Unique Values tab."""
    st.subheader("View Unique Values Per Column")

    if "col_index" not in st.session_state:
        st.session_state.col_index = 0

    col_index = st.session_state.col_index
    column_name = df.columns[col_index]

    st.write(f"**Column ({col_index + 1}/{len(df.columns)}): {column_name}**")
    unique_values = df[column_name].dropna().unique()
    st.dataframe(pd.DataFrame(unique_values, columns=[column_name]), use_container_width=True)

    if st.button("Next Column"):
        st.session_state.col_index = (st.session_state.col_index + 1) % len(df.columns)


def render_match_tab(df: pd.DataFrame) -> None:
    """Render the Match tab."""
    st.subheader("Filter Data by Column Values")

    match_cols = st.multiselect("Choose up to 3 column(s) to match", df.columns, max_selections=3)

    if not match_cols:
        return

    filters = []
    for col in match_cols:
        val = st.text_input(f"Value to match in '{col}'", key=f"match_{col}")
        mode = st.selectbox(f"Match mode for '{col}'", ["equals", "contains"], key=f"mode_{col}")
        filters.append((col, val, mode))

    if st.button("Filter Data"):
        filtered_df = df.copy()
        
        for col, val, mode in filters:
            if val.strip():
                col_series = filtered_df[col].astype(str)
                if mode == "equals":
                    filtered_df = filtered_df[col_series == val.strip()]
                elif mode == "contains":
                    filtered_df = filtered_df[col_series.str.contains(val.strip(), na=False)]
        
        st.write(f"Filtered Data ({len(filtered_df)} rows):")
        st.dataframe(filtered_df, use_container_width=True)


def render_explore_tab(df: pd.DataFrame) -> None:
    """Render the Explore tab."""
    st.subheader("Explore Data")

    col_filter = st.selectbox("Choose column for condition", df.columns)
    col_condition = st.radio("Condition", ["is null", "is not null", "is duplicated", "drop duplicates"])

    regex_filter_columns = st.multiselect("Columns for regex checks", df.columns)
    regex_option = st.selectbox("Regex option", ["is not null", "extra spaces", "leading or trailing spaces", "missing spaces"])

    do_sample = st.checkbox("View Sample")

    filtered_df = column_conditions(col_filter, df, col_condition)
    if regex_filter_columns:
        filtered_df = regex_conditions(regex_filter_columns, filtered_df, regex_option)

    if do_sample:
        display_sample_structured(filtered_df, n=1)
    else:
        st.dataframe(filtered_df, use_container_width=True)


def render_group_by_tab(df: pd.DataFrame) -> None:
    """Render the Group By tab."""
    st.subheader("Group By / Summarize")

    group_cols = st.multiselect("Choose column(s) to group by", df.columns)
    agg_cols = st.multiselect("Choose column(s) to aggregate", df.columns)

    agg_funcs = {}
    for col in agg_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            options = ["sum", "mean", "min", "max", "count", "median", "std"]
        else:
            options = ["count"]
        agg_funcs[col] = st.selectbox(f"Aggregation for {col}", options, key=f"agg_{col}")

    if st.button("Run GroupBy"):
        if not group_cols:
            st.warning("Please select at least one column to group by.")
            return

        if not agg_cols:
            # Simple count groupby
            grouped_df = df.groupby(group_cols).size().reset_index(name="Count")
        else:
            # Aggregation groupby
            grouped = df.groupby(group_cols).agg(agg_funcs)

            # Flatten MultiIndex columns if they exist
            if isinstance(grouped.columns, pd.MultiIndex):
                grouped.columns = ["_".join(filter(None, map(str, col))).strip() for col in grouped.columns]

            # Ensure column names don't clash with group columns
            existing_cols = set(group_cols)
            final_cols = []
            
            for col in grouped.columns:
                col_name = col
                counter = 1
                while col_name in existing_cols:
                    col_name = f"{col}_{counter}"
                    counter += 1
                final_cols.append(col_name)
                existing_cols.add(col_name)
            
            grouped.columns = final_cols
            grouped_df = grouped.reset_index()

        st.dataframe(grouped_df, use_container_width=True)


# --- Main app ---
def main() -> None:
    """Main application function."""
    # Sidebar upload
    st.sidebar.title("Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.info("Please upload a CSV or Excel file to begin.")
        return

    try:
        df = load_dataframe(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    st.title("Inspect Data")

    tabs = st.tabs([
        "Basic Info", "Preview", "Formating checks", 
        "Unique Values", "Match", "Explore", "Group by"
    ])

    with tabs[0]:
        render_basic_info_tab(df, uploaded_file)

    with tabs[1]:
        render_preview_tab(df)

    with tabs[2]:
        render_formatting_checks_tab(df)

    with tabs[3]:
        render_unique_values_tab(df)

    with tabs[4]:
        render_match_tab(df)

    with tabs[5]:
        render_explore_tab(df)

    with tabs[6]:
        render_group_by_tab(df)


if __name__ == "__main__":
    main()