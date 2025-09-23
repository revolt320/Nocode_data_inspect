"""Streamlit Data Inspector - Memory-Efficient Version with All Original Functionality."""

import gc
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Constants ---
NUM_REGEX = r"^\d+\.?\d*$"
MEMORY_CHUNK_SIZE = 25000  # Smaller chunks for memory efficiency
MAX_WORKERS = 2  # Conservative to prevent memory issues
TEMP_DIR = tempfile.gettempdir()

SPACE_PATTERNS = {
    "trailing": re.compile(r"\s$"),
    "leading": re.compile(r"^\s"),  
    "extra": re.compile(r"\s{2,}"),
    "missing": re.compile(r"\S\S"),
    "html": re.compile(r"<.*>"),
}

# --- Page setup ---
st.set_page_config(page_title="Data Inspector", layout="wide")

# --- Memory Management ---
def clear_memory():
    """Force garbage collection."""
    gc.collect()

def get_memory_usage():
    """Get memory usage info."""
    try:
        import psutil
        process = psutil.Process()
        return f"{process.memory_info().rss / 1024 / 1024:.1f} MB"
    except:
        return "Unknown"

# --- Large Dataset Handler ---
class MemoryEfficientDataFrame:
    """Memory-efficient wrapper for large datasets."""
    
    def __init__(self, file_path: str, file_type: str):
        self.file_path = file_path
        self.file_type = file_type
        self._sample_df = None
        self._columns = None
        self._total_rows = None
        self._dtypes = None
    
    @property
    def columns(self):
        """Get column names."""
        if self._columns is None:
            if self.file_type == "csv":
                header_df = pd.read_csv(self.file_path, nrows=0, dtype=str)
            else:
                header_df = pd.read_excel(self.file_path, nrows=0, dtype=str)
            self._columns = header_df.columns
            del header_df
            clear_memory()
        return self._columns
    
    @property
    def shape(self):
        """Get shape of dataset."""
        if self._total_rows is None:
            if self.file_type == "csv":
                with open(self.file_path, 'rb') as f:
                    self._total_rows = max(0, sum(1 for _ in f) - 1)
            else:
                # For Excel, count efficiently
                try:
                    xl = pd.ExcelFile(self.file_path)
                    self._total_rows = len(pd.read_excel(self.file_path, usecols=[0], dtype=str))
                except:
                    self._total_rows = 0
        return (self._total_rows, len(self.columns))
    
    def get_working_sample(self, sample_size=50000):
        """Get a working sample for operations."""
        if self._sample_df is None or len(self._sample_df) != min(sample_size, self.shape[0]):
            try:
                if self.file_type == "csv":
                    self._sample_df = pd.read_csv(self.file_path, nrows=sample_size, dtype='object')
                else:
                    self._sample_df = pd.read_excel(self.file_path, nrows=sample_size, dtype='object')
            except Exception as e:
                st.error(f"Error loading sample: {e}")
                self._sample_df = pd.DataFrame()
        return self._sample_df
    
    def iter_chunks(self, chunk_size=MEMORY_CHUNK_SIZE):
        """Iterate through chunks."""
        processed = 0
        while processed < self.shape[0]:
            try:
                if self.file_type == "csv":
                    if processed == 0:
                        chunk = pd.read_csv(self.file_path, nrows=chunk_size, dtype='object')
                    else:
                        chunk = pd.read_csv(self.file_path, skiprows=range(1, processed + 1), 
                                          nrows=chunk_size, dtype='object')
                else:
                    chunk = pd.read_excel(self.file_path, skiprows=processed, 
                                        nrows=chunk_size, dtype='object')
                
                if chunk.empty:
                    break
                    
                yield chunk
                processed += len(chunk)
                del chunk
                clear_memory()
                
            except Exception:
                break
    
    def __getitem__(self, key):
        """Support indexing operations on sample."""
        return self.get_working_sample()[key]
    
    def __len__(self):
        """Return total length."""
        return self.shape[0]

def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily."""
    temp_path = Path(TEMP_DIR) / f"streamlit_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(temp_path)

# --- Original Helper Functions (Memory-Optimized) ---
def column_conditions(column, df, buttons):
    """Original column conditions logic."""
    if buttons == 'is null':
        return df[df[column].isnull()]
    elif buttons == 'is not null':
        return df[df[column].notnull()]
    elif buttons == 'is duplicated':
        return df[df.duplicated(subset=column)]
    elif buttons == 'drop duplicates':
        return df.drop_duplicates(subset=column)
    else:
        return df

def regex_conditions(fields, df, regex):
    """Original regex conditions logic."""
    indices = []
    for f in fields:
        if f not in df.columns:
            continue
            
        if regex == "is not null":
            indices.extend(df[df[f].notnull()].index)
        elif regex == "extra spaces":
            indices.extend(df[df[f].astype(str).str.contains(r"\s{2,}", regex=True, na=False)].index)
        elif regex == "leading or trailing spaces":
            indices.extend(df[df[f].astype(str).str.match(r"^\s|\s$", na=False)].index)
        elif regex == "missing spaces":
            indices.extend(df[df[f].astype(str).str.contains(r"\S\S", regex=True, na=False)].index)
        else:
            indices.extend(df.index)
    
    if indices:
        return df.loc[indices]
    return df.iloc[0:0]

def display_sample_structured(df, n=1):
    """Original sample display logic."""
    if df.empty:
        st.write("No data to display")
        return
        
    sample_df = df.sample(min(n, len(df)))
    for _, row in sample_df.iterrows():
        for col, val in row.items():
            st.markdown(f"**{col}**  \n{val}")
        st.markdown("---")

def spacecheck_ui(dfr, url_column):
    """Original space check logic - chunked for large datasets."""
    results = []
    chunk_size = min(10000, len(dfr))  # Process in smaller chunks
    
    for start_idx in range(0, len(dfr), chunk_size):
        chunk = dfr.iloc[start_idx:start_idx + chunk_size]
        
        for index, row in chunk.iterrows():
            for i in row.keys():
                val = str(row[i])
                if re.search(r'\s$', val):
                    results.append((i, "Trailing space", val, row[url_column]))
                if re.search(r'^\s', val):
                    results.append((i, "Leading space", val, row[url_column]))
                if re.search(r'\s\s', val):
                    results.append((i, "Extra spaces", val, row[url_column]))
                if re.search(r'<.*>', val):
                    results.append((i, "HTML Tag", val, row[url_column]))
        
        # Limit results to prevent memory issues
        if len(results) > 10000:
            results = results[:10000]
            st.warning("Results limited to 10,000 items for memory efficiency")
            break
            
        clear_memory()
    
    return pd.DataFrame(results, columns=["Column", "Issue", "Value", url_column])

# --- UI Rendering Functions ---
def render_basic_info_tab(df, uploaded_file):
    """Original Basic Info tab."""
    st.write("**Filename:**", uploaded_file.name if uploaded_file else "No file uploaded")
    st.write("**Number of Rows:**", df.shape[0])
    st.write("**Number of Columns:**", df.shape[1])
    st.write("**Current Memory Usage:**", get_memory_usage())
    
    # Use sample for describe to prevent memory issues
    sample_df = df.get_working_sample(10000)
    if not sample_df.empty:
        st.write("**Summary Statistics (Sample):**")
        st.dataframe(sample_df.describe(include="all").T, use_container_width=True)

def render_preview_tab(df):
    """Original Preview tab with pagination."""
    st.subheader("Preview Data")
    
    # Add pagination for large datasets
    if hasattr(df, 'shape') and df.shape[0] > 10000:
        st.info(f"Large dataset detected ({df.shape[0]:,} rows). Showing sample of 10,000 rows.")
        sample_df = df.get_working_sample(10000)
        st.dataframe(sample_df, use_container_width=True)
    else:
        sample_df = df.get_working_sample()
        st.dataframe(sample_df, use_container_width=True)

def render_formatting_checks_tab(df):
    """Original Formatting checks tab."""
    st.subheader("Text Checks for Spaces / HTML Tags")
    
    columns_list = list(df.columns)
    url_col = st.selectbox("Select column to display as URL/reference", columns_list)
    
    if st.button("Run Checks"):
        with st.spinner("Running checks on dataset..."):
            # Use sample for large datasets
            sample_df = df.get_working_sample(25000)  # Larger sample for checks
            if not sample_df.empty:
                results_df = spacecheck_ui(sample_df, url_col)
                st.dataframe(results_df, use_container_width=True)
                if df.shape[0] > 25000:
                    st.info("Results based on sample of 25,000 rows for performance.")
            else:
                st.error("Could not load data for analysis")

def render_unique_values_tab(df):
    """Original Unique Values tab."""
    st.subheader("View Unique Values Per Column")

    if "col_index" not in st.session_state:
        st.session_state.col_index = 0

    columns_list = list(df.columns)
    col_index = st.session_state.col_index
    column_name = columns_list[col_index]

    st.write(f"**Column ({col_index + 1}/{len(columns_list)}): {column_name}**")
    
    # Use sample for unique values
    sample_df = df.get_working_sample(50000)
    if not sample_df.empty and column_name in sample_df.columns:
        unique_values = sample_df[column_name].dropna().unique()
        st.dataframe(pd.DataFrame(unique_values, columns=[column_name]), use_container_width=True)
        if df.shape[0] > 50000:
            st.info("Unique values based on sample for performance.")
    else:
        st.error("Could not load column data")

    if st.button("Next Column"):
        if st.session_state.col_index < len(columns_list) - 1:
            st.session_state.col_index += 1
        else:
            st.session_state.col_index = 0

def render_match_tab(df):
    """Original Match tab."""
    st.subheader("Filter Data by Column Values")

    columns_list = list(df.columns)
    match_cols = st.multiselect("Choose up to 3 column(s) to match", columns_list, max_selections=3)

    match_values = []
    match_modes = []

    for col in match_cols:
        val = st.text_input(f"Value to match in '{col}'", key=f"match_{col}")
        match_values.append(val)
        mode = st.selectbox(f"Match mode for '{col}'", ["equals", "contains"], key=f"mode_{col}")
        match_modes.append(mode)

    if st.button("Filter Data"):
        with st.spinner("Filtering data..."):
            sample_df = df.get_working_sample()
            filtered_df = sample_df.copy()
            
            for col, val, mode in zip(match_cols, match_values, match_modes):
                if val.strip() != "":
                    if mode == "equals":
                        filtered_df = filtered_df[filtered_df[col].astype(str) == val.strip()]
                    elif mode == "contains":
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(val.strip(), na=False)]
            
            st.write(f"Filtered Data ({len(filtered_df)} rows):")
            st.dataframe(filtered_df, use_container_width=True)
            
            if df.shape[0] > len(sample_df):
                st.info("Filtering applied to sample for performance.")

def render_explore_tab(df):
    """Original Explore tab."""
    st.subheader("Explore Data")

    columns_list = list(df.columns)
    col_filter = st.selectbox("Choose column for condition", columns_list)
    col_condition = st.radio("Condition", ['is null', 'is not null', 'is duplicated', 'drop duplicates'])

    regex_filter_columns = st.multiselect("Columns for regex checks", columns_list)
    regex_option = st.selectbox("Regex option", ["is not null", "extra spaces", "leading or trailing spaces", "missing spaces"])

    do_sample = st.checkbox("View Sample")

    if st.button("Apply Explore Filters"):
        with st.spinner("Processing..."):
            sample_df = df.get_working_sample()
            
            filtered_df = column_conditions(col_filter, sample_df, col_condition)
            if regex_filter_columns:
                filtered_df = regex_conditions(regex_filter_columns, filtered_df, regex_option)

            if do_sample:
                display_sample_structured(filtered_df, n=1)
            else:
                st.dataframe(filtered_df, use_container_width=True)
                
            if df.shape[0] > len(sample_df):
                st.info("Analysis applied to sample for performance.")

def render_group_by_tab(df):
    """Original Group By tab."""
    st.subheader("Group By / Summarize")

    columns_list = list(df.columns)
    group_cols = st.multiselect("Choose column(s) to group by", columns_list)
    agg_cols = st.multiselect("Choose column(s) to aggregate", columns_list)

    agg_funcs = {}
    sample_df = df.get_working_sample()
    
    for col in agg_cols:
        if col in sample_df.columns:
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                options = ["sum", "mean", "min", "max", "count", "median", "std"]
            else:
                options = ["count"]
            agg_funcs[col] = st.selectbox(f"Aggregation for {col}", options, key=f"agg_{col}")

    if st.button("Run GroupBy"):
        if group_cols and sample_df is not None and not sample_df.empty:
            with st.spinner("Computing aggregations..."):
                try:
                    if not agg_cols:
                        grouped_df = sample_df.groupby(group_cols, observed=True).size().reset_index(name="Count")
                    else:
                        grouped = sample_df.groupby(group_cols, observed=True).agg(agg_funcs)

                        if isinstance(grouped.columns, pd.MultiIndex):
                            grouped.columns = ['_'.join(filter(None, map(str, col))).strip() for col in grouped.columns]

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
                    
                    if df.shape[0] > len(sample_df):
                        st.info("GroupBy computed on sample for performance.")
                        
                except Exception as e:
                    st.error(f"Error in GroupBy operation: {e}")
        elif group_cols:
            grouped_df = sample_df.groupby(group_cols, observed=True).size().reset_index(name="Count")
            st.dataframe(grouped_df, use_container_width=True)
        else:
            st.warning("Please select at least one column to group by.")

# --- Main Application ---
def main():
    """Main application with original logic preserved."""
    # Sidebar upload
    st.sidebar.title("Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "xlsx"])
    
    # Memory info in sidebar
    st.sidebar.info(f"Memory: {get_memory_usage()}")

    if uploaded_file is not None:
        try:
            # Save file temporarily
            temp_file_path = save_uploaded_file(uploaded_file)
            file_type = "csv" if uploaded_file.name.endswith(".csv") else "xlsx"
            
            # Create memory-efficient dataframe wrapper
            df = MemoryEfficientDataFrame(temp_file_path, file_type)
            
            st.sidebar.success("File loaded!")
            st.sidebar.info(f"Rows: {df.shape[0]:,}")
            st.sidebar.info(f"Columns: {df.shape[1]}")
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

        st.title("Inspect Data")

        # Original tabs exactly as specified
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

    else:
        st.info("Please upload a CSV or Excel file to begin.")

if __name__ == "__main__":
    main()