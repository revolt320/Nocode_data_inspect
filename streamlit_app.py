import streamlit as st
import pandas as pd
import numpy as np
import re
import io

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
file_types = ["csv", "json"]
if check_excel_support():
    file_types.append("xlsx")

uploaded_file = st.sidebar.file_uploader("Upload a file", type=file_types)

# --- Sheet selector for Excel files ---
df = None
sheet_name = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            if not check_excel_support():
                st.error("❌ Cannot read Excel files. Please install 'openpyxl'.")
                st.stop()

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

        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file, dtype='object')

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
    sample_df = df.sample(min(n, len(df)))
    for _, row in sample_df.iterrows():
        for col, val in row.items():
            st.markdown(f"**{col}**  \n{val}")
        st.markdown("---")

def spacecheck_ui(dfr, url_column):
    results = []

    # --- Hidden / unwanted Unicode characters pattern ---
    hidden_unicode_pattern = re.compile(
        r'[\u2000-\u200F\u202A-\u202E\u2060\u205F\u00A0\u180E\uFEFF\u3000]'
    )

    for index, row in dfr.iterrows():
        for i in row.keys():
            val = str(row[i])

            # --- Space and HTML checks ---
            if re.search(r'\s$', val):
                results.append((i, "Trailing space", val, row[url_column]))

            if re.search(r'^\s', val):
                results.append((i, "Leading space", val, row[url_column]))

            if re.search(r'\s\s', val):
                results.append((i, "Extra spaces", val, row[url_column]))

            if re.search(r'<.*>', val):
                results.append((i, "HTML Tag", val, row[url_column]))

            # --- Hidden / zero-width Unicode detection ---
            if hidden_unicode_pattern.search(val):
                results.append((i, "Hidden Unicode character", val, row[url_column]))

            # --- General non-ASCII characters (emojis, accents, etc.) ---
            elif re.search(r'[^\x00-\x7F]', val):
                results.append((i, "Unicode character found", val, row[url_column]))

    return pd.DataFrame(results, columns=["Column", "Issue", "Value", url_column])

# --- Main app ---
if uploaded_file is not None and df is not None:
    st.title("Inspect Data")

    tabs = st.tabs([
        "Basic Info", "Preview", "Formating checks",
        "Unique Values", "Match", "Explore", "Group by", "Generate Dataset"
    ])

    # --- Basic Info ---
    with tabs[0]:
        st.write("**Filename:**", uploaded_file.name)
        if sheet_name:
            st.write("**Sheet Name:**", sheet_name)

        st.write("**Number of Rows:**", df.shape[0])
        st.write("**Number of Columns:**", df.shape[1])

        if uploaded_file.name.endswith(".xlsx"):
            excel_file = pd.ExcelFile(uploaded_file)
            st.write("**Available Sheets:**")
            for i, sheet in enumerate(excel_file.sheet_names, 1):
                if sheet == sheet_name:
                    st.write(f"{i}. **{sheet}** *(current)*")
                else:
                    st.write(f"{i}. {sheet}")

        st.write("**Summary Statistics:**")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # --- Preview ---
    with tabs[1]:
        st.subheader("Preview Data")
        if sheet_name:
            st.write(f"*Showing data from sheet: {sheet_name}*")
        st.dataframe(df, use_container_width=True)

    # --- Text Checks ---
    with tabs[2]:
        st.subheader("Text Checks for Spaces / HTML Tags / Unicode")
        url_col = st.selectbox("Select column to display as URL/reference", df.columns)
        if st.button("Run Checks"):
            results_df = spacecheck_ui(df, url_col)
            st.dataframe(results_df, use_container_width=True)

    # --- Unique Values ---
    with tabs[3]:
        st.subheader("View Unique Values Per Column")
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
                st.session_state[sheet_key] = 0

    # --- Match ---
    with tabs[4]:
        st.subheader("Filter Data by Column Values")
        match_cols = st.multiselect("Choose up to 3 column(s) to match", df.columns, max_selections=3)
        match_values, match_modes = [], []
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

    # --- Group By ---
    with tabs[6]:
        st.subheader("Group By / Summarize")
        group_cols = st.multiselect("Choose column(s) to group by", df.columns)
        agg_cols = st.multiselect("Choose column(s) to aggregate", df.columns)
        agg_funcs = {}
        for col in agg_cols:
            if np.issubdtype(df[col].dtype, np.number):
                options = ["sum", "mean", "min", "max", "count", "median", "std"]
            else:
                options = ["count"]
            agg_funcs[col] = st.selectbox(f"Aggregation for {col}", options, key=f"agg_{col}")
        if st.button("Run GroupBy"):
            if group_cols and agg_cols:
                grouped = df.groupby(group_cols).agg(agg_funcs)
                if isinstance(grouped.columns, pd.MultiIndex):
                    grouped.columns = ['_'.join(filter(None, map(str, col))).strip() for col in grouped.columns]
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
                grouped_df = grouped.reset_index()
                st.dataframe(grouped_df, use_container_width=True)
            elif group_cols:
                grouped_df = df.groupby(group_cols).size().reset_index(name="Count")
                st.dataframe(grouped_df, use_container_width=True)
            else:
                st.warning("Please select at least one column to group by.")

    
    # --- Generate Sample Dataset ---
    with tabs[7]:
        st.subheader("Generate Dataset")

        default_filename = uploaded_file.name if uploaded_file else "sample_output.csv"
        filename = st.text_input("Enter output filename (without extension):", value=default_filename.split(".")[0])
        export_format = st.selectbox("Select export format", ["csv", "xlsx", "json"])

        st.write("Select columns to remove (optional):")
        cols_to_remove = st.multiselect("Columns to eliminate from dataset", df.columns)

        remove_dupes = st.checkbox("Remove Duplicates")
        dupe_columns = []
        if remove_dupes:
            dupe_columns = st.multiselect("Select column(s) to check for duplicates", df.columns)
            st.info("If no columns are selected, all columns will be considered for duplicate removal.")

        # --- New option: Clean extra spaces and remove hidden Unicode ---
        clean_text = st.checkbox("Clean extra spaces and remove hidden/unwanted Unicode characters")

        # Ensure we have at least 1 row for the input
        default_rows = max(1, len(df))
        max_rows = max(1, len(df))
        
        num_rows = st.number_input(
            "Number of rows for the sample dataset:",
            min_value=1,
            max_value=max_rows,
            value=default_rows  # default value is the existing number of rows
        )

        st.write("Filter out rows that **do not match** given conditions (up to 3 columns):")
        mismatch_cols = st.multiselect("Select up to 3 column(s)", df.columns, max_selections=3)

        mismatch_conditions = []
        mismatch_values = []
        for col in mismatch_cols:
            val = st.text_input(f"Value (or partial text) that column '{col}' should contain", key=f"mismatch_val_{col}")
            mismatch_values.append(val)
            mode = st.selectbox(f"Match type for '{col}'", ["equals", "contains"], key=f"mismatch_mode_{col}")
            mismatch_conditions.append(mode)

        if st.button("Generate File"):
            sample_df = df.copy()
            if cols_to_remove:
                sample_df = sample_df.drop(columns=cols_to_remove, errors='ignore')

            if remove_dupes:
                if dupe_columns:
                    sample_df = sample_df.drop_duplicates(subset=dupe_columns)
                else:
                    sample_df = sample_df.drop_duplicates()
                st.success(f"✅ Removed duplicates. Remaining rows: {len(sample_df)}")

            # --- Clean extra spaces and remove hidden/unicode ---
            if clean_text:
                hidden_unicode_pattern = re.compile(
                    r'[\u2000-\u200F\u202A-\u202E\u2060\u205F\u00A0\u180E\uFEFF\u3000]'
                )
                for col in sample_df.select_dtypes(include=['object']).columns:
                    sample_df[col] = sample_df[col].astype(str).str.strip()  # remove leading/trailing spaces
                    sample_df[col] = sample_df[col].str.replace(r'\s+', ' ', regex=True)  # replace multiple spaces with single
                    sample_df[col] = sample_df[col].apply(lambda x: hidden_unicode_pattern.sub('', x))  # remove hidden/unicode

            for col, val, mode in zip(mismatch_cols, mismatch_values, mismatch_conditions):
                if val.strip() != "":
                    if mode == "equals":
                        sample_df = sample_df[sample_df[col].astype(str) != val.strip()]
                    elif mode == "contains":
                        sample_df = sample_df[~sample_df[col].astype(str).str.contains(val.strip(), na=False)]

            sample_df = sample_df.sample(n=min(num_rows, len(sample_df)))

            output_filename = f"{filename}.{export_format}"
            if export_format == "csv":
                file_data = sample_df.to_csv(index=False).encode("utf-8")
                mime = "text/csv"
            elif export_format == "xlsx":
                buffer = io.BytesIO()
                sample_df.to_excel(buffer, index=False, engine="openpyxl")
                buffer.seek(0)
                file_data = buffer.getvalue()
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                file_data = sample_df.to_json(orient="records", indent=2).encode("utf-8")
                mime = "application/json"

            st.success("✅ Cleaned dataset ready! Download below:")
            st.download_button(
                label=f"Download {export_format.upper()} File",
                data=file_data,
                file_name=output_filename,
                mime=mime
            )
            st.dataframe(sample_df, use_container_width=True)


else:
    st.info("Please upload a CSV, Excel, or JSON file to begin.")
