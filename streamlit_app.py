import streamlit as st
import pandas as pd
import numpy as np
import re

# --- Page setup ---
st.set_page_config(page_title="Data Inspector", layout="wide")

# --- Check for required dependencies ---
def check_excel_support():
    try:
        import openpyxl
        return True
    except ImportError:
        return False

# --- Sidebar upload ---
st.sidebar.title("Upload")

# Adjust file types based on available dependencies
if check_excel_support():
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "xlsx"])
else:
    st.sidebar.warning("⚠️ Excel support not available. Please install openpyxl to read Excel files.")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv"])

# --- Sheet selector for Excel files ---
df = None
sheet_name = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            if not check_excel_support():
                st.error("❌ Cannot read Excel files. Please install the 'openpyxl' library or upload a CSV file instead.")
                st.info("To install openpyxl, run: `pip install openpyxl`")
                st.stop()
            
            # Read Excel file to get sheet names
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1:
                sheet_name = st.sidebar.selectbox("Select Sheet", sheet_names)
            else:
                sheet_name = sheet_names[0]
            
            st.sidebar.write(f"**Selected Sheet:** {sheet_name}")
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, dtype='object')
            
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype='object')
            
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("Please check that your file is properly formatted and try again.")
        st.stop()

# --- Helper functions ---
NUM_REGEX = r"^\d+\.?\d*$"

def column_conditions(column, df, buttons):
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
    indices = []
    for f in fields:
        if regex == "is not null":
            indices.extend(df[df[f].notnull()].index)
        elif regex == "extra spaces":
            indices.extend(df[df[f].astype(str).str.contains(r"\s{2,}", regex=True)].index)
        elif regex == "leading or trailing spaces":
            indices.extend(df[df[f].astype(str).str.match(r"^\s|\s$", na=False)].index)
        elif regex == "missing spaces":
            indices.extend(df[df[f].astype(str).str.contains(r"\S\S", regex=True)].index)
        else:
            indices.extend(df.index)
    return df.loc[indices]

def display_sample_structured(df, n=1):
    """Display sample(s) in key: value format."""
    sample_df = df.sample(min(n, len(df)))
    for _, row in sample_df.iterrows():
        for col, val in row.items():
            st.markdown(f"**{col}**  \n{val}")
        st.markdown("---")  # separator between samples

def spacecheck_ui(dfr, url_column):
    results = []
    for index, row in dfr.iterrows():
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
    return pd.DataFrame(results, columns=["Column", "Issue", "Value", url_column])

# --- Main app ---
if uploaded_file is not None and df is not None:
    st.title("Inspect Data")

    tabs = st.tabs([
        "Basic Info", "Preview", "Formating checks", 
        "Unique Values", "Match", "Explore", "Group by"
    ])

    # --- Basic Info ---
    with tabs[0]:
        st.write("**Filename:**", uploaded_file.name)
        if sheet_name:
            st.write("**Sheet Name:**", sheet_name)

        # Display number of rows and columns
        st.write("**Number of Rows:**", df.shape[0])
        st.write("**Number of Columns:**", df.shape[1])

        # Display available sheets for Excel files
        if uploaded_file.name.endswith(".xlsx"):
            excel_file = pd.ExcelFile(uploaded_file)
            st.write("**Available Sheets:**")
            for i, sheet in enumerate(excel_file.sheet_names, 1):
                if sheet == sheet_name:
                    st.write(f"  {i}. **{sheet}** *(current)*")
                else:
                    st.write(f"  {i}. {sheet}")

        # Display summary statistics
        st.write("**Summary Statistics:**")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # --- Preview ---
    with tabs[1]:
        st.subheader("Preview Data")
        if sheet_name:
            st.write(f"*Showing data from sheet: {sheet_name}*")
        st.dataframe(df, use_container_width=True)

    # --- Match ---
    with tabs[4]:
        st.subheader("Filter Data by Column Values")

        # Select up to 3 columns
        match_cols = st.multiselect("Choose up to 3 column(s) to match", df.columns, max_selections=3)

        match_values = []
        match_modes = []

        # Input values and match mode for each selected column
        for col in match_cols:
            val = st.text_input(f"Value to match in '{col}'", key=f"match_{col}")
            match_values.append(val)
            mode = st.selectbox(f"Match mode for '{col}'", ["equals", "contains"], key=f"mode_{col}")
            match_modes.append(mode)

        if st.button("Filter Data"):
            filtered_df = df.copy()
            for col, val, mode in zip(match_cols, match_values, match_modes):
                if val.strip() != "":
                    if mode == "equals":
                        filtered_df = filtered_df[filtered_df[col].astype(str) == val.strip()]
                    elif mode == "contains":
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(val.strip(), na=False)]
            st.write(f"Filtered Data ({len(filtered_df)} rows):")
            st.dataframe(filtered_df, use_container_width=True)

    # --- Group By ---
    with tabs[6]:
        st.subheader("Group By / Summarize")

        # Select columns to group by
        group_cols = st.multiselect("Choose column(s) to group by", df.columns)

        # Select aggregation columns (all columns allowed)
        agg_cols = st.multiselect("Choose column(s) to aggregate", df.columns)

        # Select aggregation functions per column
        agg_funcs = {}
        for col in agg_cols:
            if np.issubdtype(df[col].dtype, np.number):
                options = ["sum", "mean", "min", "max", "count", "median", "std"]
            else:
                options = ["count"]  # Non-numeric columns can only use count
            agg_funcs[col] = st.selectbox(f"Aggregation for {col}", options, key=f"agg_{col}")

        if st.button("Run GroupBy"):
            if group_cols and agg_cols:
                # Perform groupby aggregation
                grouped = df.groupby(group_cols).agg(agg_funcs)

                # Flatten MultiIndex if exists (for multiple agg functions)
                if isinstance(grouped.columns, pd.MultiIndex):
                    grouped.columns = ['_'.join(filter(None, map(str, col))).strip() for col in grouped.columns]

                # Ensure aggregated columns do not clash with group_cols
                final_cols = []
                existing_cols = set(group_cols)
                for col in grouped.columns:
                    col_name = col
                    counter = 1
                    while col_name in existing_cols:
                        col_name = f"{col}_{counter}"
                        counter += 1
                    final_cols.append(col_name)
                    existing_cols.add(col_name)
                grouped.columns = final_cols

                # Reset index after renaming to avoid conflicts
                grouped_df = grouped.reset_index()
                st.dataframe(grouped_df, use_container_width=True)
            elif group_cols:
                grouped_df = df.groupby(group_cols).size().reset_index(name="Count")
                st.dataframe(grouped_df, use_container_width=True)
            else:
                st.warning("Please select at least one column to group by.")

    # --- Explore ---
    with tabs[5]:
        st.subheader("Explore Data")

        col_filter = st.selectbox("Choose column for condition", df.columns)
        col_condition = st.radio("Condition", ['is null', 'is not null', 'is duplicated', 'drop duplicates'])

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

    # --- Text Checks ---
    with tabs[2]:
        st.subheader("Text Checks for Spaces / HTML Tags")
        url_col = st.selectbox("Select column to display as URL/reference", df.columns)
        if st.button("Run Checks"):
            results_df = spacecheck_ui(df, url_col)
            st.dataframe(results_df, use_container_width=True)

    # --- Unique Values with Next Button ---
    with tabs[3]:
        st.subheader("View Unique Values Per Column")

        # Create unique session state key based on sheet name to reset column index when sheet changes
        sheet_key = f"col_index_{sheet_name}" if sheet_name else "col_index"
        if sheet_key not in st.session_state:
            st.session_state[sheet_key] = 0

        col_index = st.session_state[sheet_key]
        column_name = df.columns[col_index]

        st.write(f"**Column ({col_index + 1}/{len(df.columns)}): {column_name}**")
        unique_values = df[column_name].dropna().unique()
        st.dataframe(pd.DataFrame(unique_values, columns=[column_name]), use_container_width=True)

        if st.button("Next Column"):
            if st.session_state[sheet_key] < len(df.columns) - 1:
                st.session_state[sheet_key] += 1
            else:
                st.session_state[sheet_key] = 0  # loop back to first column

else:
    st.info("Please upload a CSV or Excel file to begin.")
