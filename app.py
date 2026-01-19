import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import vl_convert as vlc

st.set_page_config(page_title="Defect Analysis Tool", layout="wide")

# --- Theme Configuration ---
col_header, col_theme = st.columns([6, 1])
with col_theme:
    theme_mode = st.toggle("Dark Mode", value=False)

# Define Theme Variables
if theme_mode:
    app_bg_color = "#0e1117"
    app_text_color = "#fafafa"
    header_color = "#64B5F6" # Lighter blue for dark mode
    sub_header_color = "#E0E0E0"
    metric_card_bg = "#262730"
    plt_style = 'dark_background'
    alt_theme = 'dark'
else:
    app_bg_color = "#ffffff"
    app_text_color = "#31333F"
    header_color = "#1E88E5"
    sub_header_color = "#424242"
    metric_card_bg = "#f0f2f6"
    plt_style = 'default'
    alt_theme = 'default'

# Apply Altair Theme
alt.themes.enable(alt_theme)

# Inject CSS based on Theme
st.markdown(f"""
<style>
    .stApp {{
        background-color: {app_bg_color};
        color: {app_text_color};
    }}
    .main-header {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {header_color};
        margin-bottom: 1rem;
    }}
    .sub-header {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {sub_header_color};
        margin-top: 2rem;
    }}
    .metric-card {{
        padding: 1rem;
        background-color: {metric_card_bg};
        border-radius: 8px;
        text-align: center;
        color: {app_text_color};
    }}
    /* Force text color for some Streamlit elements if needed */
    p, h1, h2, h3, h4, h5, h6, li, span, .stDataFrame {{
        color: {app_text_color};
    }}
    /* Ensure Sidebar text is visible in Dark Mode */
    [data-testid="stSidebar"] {{
        background-color: {metric_card_bg};
    }}
    /* Specific overrides for Sidebar text to ensure contrast */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {{
         color: {app_text_color} !important;
    }}
    /* Fix File Uploader Text in Sidebar */
    [data-testid="stSidebar"] .stFileUploader label {{
        color: {app_text_color} !important;
    }}
    [data-testid="stSidebar"] .stFileUploader div {{
         color: {app_text_color};
    }}
</style> 
""", unsafe_allow_html=True)

with col_header:
    st.markdown('<div class="main-header">Manufacturing Defect Analysis Tool</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Manufacturing Data (CSV)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Verify columns
        required_cols = ['line_description', 'bde_order_number', 'GCAS', 'counter_type_description', 'count']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns. Expected: {required_cols}")
            st.stop()
            
        # Data Preprocessing: Ensure DATE is datetime
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            min_date = df['DATE'].min()
            max_date = df['DATE'].max()
        else:
            st.warning("Dataset missing 'DATE' column. Date filtering disabled.")
            min_date = max_date = None

        # --- Sidebar Configuration ---
        st.sidebar.markdown("### Filters")
        
        # 1. Date Range Filter
        if min_date and max_date:
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            # Filter Data by Date
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                df = df[(df['DATE'].dt.date >= start_date) & (df['DATE'].dt.date <= end_date)]

            
        # User Configuration: Select "Good Parts"
        unique_counters = df['counter_type_description'].unique()
        
        # Try to pre-select "Good_In from Line" if it exists
        target_defaults = [
            "Good_In from Line",
            "Bad_In from Line",
            "Total Gross Parts Count",
            "Total Bad Parts Count"
        ]
        default_good = [x for x in unique_counters if x in target_defaults]
        # Fallback: if exact matches fail, keep the old simple "Good_In" scan? 
        # User explicitly asked for these defaults. If they aren't there, default is empty list.
        # But if the user is using my dummy data which ONLY has "Good_In from Line", this works fine for that one.
        # If the user data has the others, they will be selected.
        if not default_good:
             # Fallback to smart detection if none of the specific user defaults are found
             default_good = [x for x in unique_counters if "Good_In" in x]
        
        # 2. Good Parts Selection (Full Width)
        good_parts_selection = st.sidebar.multiselect(
            "Select 'Good Parts'",
            options=unique_counters,
            default=default_good,
            help="Select counters that count produced units."
        )
        
        # Defect Configuration (Separate line or columns if needed, but keeping simple)
        defect_pattern = st.sidebar.text_input(
            "Defect Pattern (Starts with)", 
            value="Hand written",
            help="Case-insensitive string to identify defects in 'counter_type_description'. E.g. 'Hand written' or 'Ungültig'"
        )
            


        # Plot Settings
        st.sidebar.markdown("### Visualization Settings")
        max_ppm_limit = st.sidebar.slider(
            "Graph X-Axis Limit (PPM)", 
            min_value=2500, 
            max_value=50000, 
            value=8000, 
            step=500,
            help="Orders with PPM higher than this value will be excluded from the Gaussian plot distribution to improve visibility."
        )
        
        if not good_parts_selection:
            st.warning("Please select at least one counter type representing 'Good Parts' to proceed.")
            st.stop()

        # --- Logic & Calculations ---
        
        # 1. Identify Defects based on User Input Pattern
        # Case insensitive check
        df['is_defect'] = df['counter_type_description'].str.strip().str.lower().str.startswith(defect_pattern.lower())
        df['is_good'] = df['counter_type_description'].isin(good_parts_selection)
        
        # Filter data: We generally only care about rows that are either Defect or Good for PPM calc
        # However, the prompt implies "Hand written" are defects, and User Selected are Good.
        # Everything else is ignored for the denominator? 
        # -> "Sum of 'Good Parts' counts + Sum of 'Hand written' counts" suggests we focus on these two buckets.
        
        # Aggregation Helper
        def calculate_ppm(grouped_df):
            # Sum counts where is_defect is True
            defect_counts = grouped_df[grouped_df['is_defect']]['count'].sum()
            # Sum counts where is_good is True
            good_counts = grouped_df[grouped_df['is_good']]['count'].sum()
            
            total_parts = good_counts + defect_counts
            
            if total_parts == 0:
                return 0.0
            
            ppm = (defect_counts / total_parts) * 1_000_000
            return ppm

        # --- KPI Overview ---
        st.markdown('<div class="sub-header">Overview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Lines", df['line_description'].nunique())
        with col2:
            st.metric("Total Orders", df['bde_order_number'].nunique())
        with col3:
            total_defect = df[df['is_defect']]['count'].sum()
            total_good = df[df['is_good']]['count'].sum()
            overall_ppm = (total_defect / (total_good + total_defect) * 1_000_000) if (total_good + total_defect) > 0 else 0
            st.metric("Overall PPM", f"{overall_ppm:.0f}")

        # --- Graph 1: Gaussian Distribution per Line (Aggregated by Order) ---
        st.markdown('---')
        st.markdown('<div class="sub-header">1. Order Performance Distribution (Gaussian/KDE)</div>', unsafe_allow_html=True)
        
        selected_line = st.selectbox("Select Production Line", df['line_description'].unique())
        
        line_data = df[df['line_description'] == selected_line]
        
        # Calculate PPM per Order for this line
        # We need to group by order number and sum counts for good vs defect
        order_groups = line_data.groupby('bde_order_number')
        order_ppm_data = []
        
        for order_id, group in order_groups:
            ppm = calculate_ppm(group)
            order_ppm_data.append({'bde_order_number': order_id, 'ppm': ppm})
            
        df_order_ppm = pd.DataFrame(order_ppm_data)
        
        if not df_order_ppm.empty and df_order_ppm['ppm'].sum() > 0:
            # Stats (Calculated on ALL data before filtering for plot)
            mean_ppm = df_order_ppm['ppm'].mean()
            std_ppm = df_order_ppm['ppm'].std()
            
            pass_2500 = (df_order_ppm['ppm'] < 2500).mean() * 100
            pass_5000 = (df_order_ppm['ppm'] < 5000).mean() * 100

            # Filter for Plotting
            filtered_plot_data = df_order_ppm[df_order_ppm['ppm'] <= max_ppm_limit]
            excluded_count = len(df_order_ppm) - len(filtered_plot_data)
            
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
                # Matplotlib Theme Enforcement
                # Explicitly set facecolor and text params to override Streamlit native defaults
                with plt.style.context(plt_style):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor(app_bg_color)
                    ax.set_facecolor(app_bg_color)
                    
                    # Set text colors based on theme
                    text_color = "white" if theme_mode else "black"
                    ax.xaxis.label.set_color(text_color)
                    ax.yaxis.label.set_color(text_color)
                    ax.title.set_color(text_color)
                    ax.tick_params(axis='x', colors=text_color)
                    ax.tick_params(axis='y', colors=text_color)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(text_color)

                    if not filtered_plot_data.empty:
                        # Histogram
                        sns.histplot(filtered_plot_data['ppm'], kde=False, ax=ax, color='skyblue', stat='density', label='Frequency Density')
                        
                        # Add Reverse Cumulative % Line (1 - CDF)
                        # We calculate based on ALL data to show true % of Total Orders
                        sorted_ppm_all = np.sort(df_order_ppm['ppm'])
                        
                        # Create plotting points for the line within the visible range (up to max_ppm_limit)
                        x_vals = sorted_ppm_all
                        y_vals = 100 * (1 - np.arange(len(sorted_ppm_all)) / len(sorted_ppm_all))
                        
                        # Filter for plotting based on max_ppm_limit
                        mask = x_vals <= max_ppm_limit
                        x_plot = x_vals[mask]
                        y_plot = y_vals[mask]
                        
                        ax2 = ax.twinx()
                        ax2.plot(x_plot, y_plot, color='darkblue', linewidth=2, label='Reverse Cumulative % (> X PPM)')
                        ax2.set_ylabel("% of Orders > PPM", color=text_color)
                        ax2.set_ylim(0, 105)
                        ax2.tick_params(axis='y', colors=text_color)
                        ax2.spines['right'].set_color(text_color)
                        ax2.spines['top'].set_visible(False)
                        
                    else:
                        st.warning(f"No data points below {max_ppm_limit} PPM.")
    
                    if excluded_count > 0:
                        # Move text to bottom below X axis, grey and small
                        ax.text(0.99, -0.15, f"Note: Excluded {excluded_count} outliers > {max_ppm_limit}", 
                                transform=ax.transAxes, ha='right', va='top', fontsize=8, color='grey')
                    
                    # Plot customization
                    ax.axvline(2500, color='orange', linestyle='--', linewidth=2, label='Limit: 2500 PPM')
                    ax.axvline(5000, color='red', linestyle='--', linewidth=2, label='Limit: 5000 PPM')
                    
                    # Add Interval Percentage Lines at the Top
                    pct_0_2500 = (df_order_ppm['ppm'] <= 2500).mean() * 100
                    pct_2500_5000 = ((df_order_ppm['ppm'] > 2500) & (df_order_ppm['ppm'] <= 5000)).mean() * 100
                    pct_gt_5000 = (df_order_ppm['ppm'] > 5000).mean() * 100
                    
                    # Create Headroom for the lines
                    y_bottom, y_top_data = ax.get_ylim()
                    ax.set_ylim(y_bottom, y_top_data * 1.25)
                    y_line_pos = y_top_data * 1.08
                    
                    # Line Labels
                    label_color = text_color
                    
                    # Line 0-2500
                    ax.plot([0, 2500], [y_line_pos, y_line_pos], color='grey', linestyle=':', linewidth=1)
                    ax.text(1250, y_line_pos, f"{pct_0_2500:.1f}%", ha='center', va='bottom', fontsize=9, color=label_color, fontweight='bold')
                    
                    # Line 2500-5000
                    ax.plot([2500, 5000], [y_line_pos, y_line_pos], color='grey', linestyle=':', linewidth=1)
                    ax.text(3750, y_line_pos, f"{pct_2500_5000:.1f}%", ha='center', va='bottom', fontsize=9, color=label_color, fontweight='bold')
                    
                    # Line > 5000
                    if max_ppm_limit > 5000:
                        ax.plot([5000, max_ppm_limit], [y_line_pos, y_line_pos], color='grey', linestyle=':', linewidth=1)
                        mid_point = 5000 + (max_ppm_limit - 5000) / 2
                        ax.text(mid_point, y_line_pos, f"{pct_gt_5000:.1f}%", ha='center', va='bottom', fontsize=9, color=label_color, fontweight='bold')
    
                    ax.set_title(f"PPM Distribution - {selected_line}", fontsize=14)
                    ax.set_xlabel("Defect Parts Per Million (PPM)")
                    ax.set_ylabel("Density")
                    
                    # Legend with explicit styling
                    lines_1, labels_1 = ax.get_legend_handles_labels()
                    lines_2, labels_2 = (ax2.get_legend_handles_labels() if not filtered_plot_data.empty else ([], []))
                    legend = ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(1, 0.85), facecolor=app_bg_color, edgecolor=text_color)
                    for text in legend.get_texts():
                        text.set_color(text_color)
                    
                    st.pyplot(fig)
                
            with col_b:
                st.write("**Statistics**")
                st.write(f"Mean: **{mean_ppm:.0f}** PPM")
                st.write(f"Std Dev: **{std_ppm:.0f}** PPM")
                st.write(f"% < 2500: **{pass_2500:.1f}%**")
                st.write(f"% < 5000: **{pass_5000:.1f}%**")
        else:
            st.info("No sufficient data to plot distribution (PPM might be 0 for all orders).")

        # --- High PPM Orders Table (Requested Update) ---
        st.markdown('#### Orders > 2500 PPM')
        # Filter orders > 2500 using the same data as the graph (selected line)
        
        high_ppm_orders = []
        for order_id, group in order_groups:
             ppm = calculate_ppm(group)
             if ppm > 2500:
                 good_parts = group[group['is_good']]['count'].sum()
                 high_ppm_orders.append({
                     'Order Number': order_id,
                     'PPM': ppm,
                     'Gross Parts (Good)': good_parts
                 })
                 
        if high_ppm_orders:
            df_high_ppm = pd.DataFrame(high_ppm_orders).sort_values('PPM', ascending=False)
            
            # View Toggle
            order_view_format = st.radio("View Format", ["Table", "Bar Graph"], horizontal=True, key="high_ppm_view_format")
            
            if order_view_format == "Table":
                # Style High PPM Table
                st.dataframe(
                    df_high_ppm.style.format({'PPM': "{:.0f}", 'Gross Parts (Good)': "{:.0f}"})
                         .set_properties(**{'background-color': app_bg_color, 'color': app_text_color}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                 # Altair Bar Chart
                 import altair as alt
                 
                 # Create a label combining columns if needed or just use Order Number
                 # Ensure Order Number is string for categorical axis
                 df_high_ppm['Order Labels'] = df_high_ppm['Order Number'].astype(str)
                 
                 order_chart = alt.Chart(df_high_ppm).mark_bar().encode(
                     x=alt.X('PPM', title='PPM'),
                     y=alt.Y('Order Labels', sort='-x', title='Order Number', axis=alt.Axis(labelLimit=1000)),
                     tooltip=['Order Number', alt.Tooltip('PPM', format=',.0f'), alt.Tooltip('Gross Parts (Good)', format=',.0f')]
                 ).properties(
                     title=f"High PPM Orders ({selected_line})",
                     height=max(300, len(df_high_ppm) * 30)
                 ).configure(background=app_bg_color)

                 if theme_mode:
                    order_chart = order_chart.configure_axis(
                        labelColor='white', titleColor='white', gridColor='#444'
                    ).configure_title(color='white')
                 else:
                    order_chart = order_chart.configure_axis(
                        labelColor='black', titleColor='black', gridColor='#eee'
                    ).configure_title(color='black')
                 
                 st.altair_chart(order_chart, use_container_width=True)
                 
                 # Download Graph Image
                 try:
                    png_order = vlc.vegalite_to_png(order_chart.to_json(), scale=2)
                    st.download_button("Download Graph as PNG", png_order, "high_ppm_orders_graph.png", "image/png", key='dl_img_order')
                 except Exception as e:
                    st.error(f"Could not generate image: {e}")
            
            # Download Button for High PPM Orders
            csv_high_ppm = df_high_ppm.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download High PPM Orders as CSV",
                csv_high_ppm,
                "high_ppm_orders.csv",
                "text/csv",
                key='download-high-ppm'
            )
        else:
            st.success("No orders exceed 2500 PPM on this line.")

        # --- Graph 2: Scatter Plot of GCAS Performance (PPM vs Size) ---
        st.markdown('---')
        st.markdown('<div class="sub-header">2. GCAS Performance: PPM vs Volume</div>', unsafe_allow_html=True)
        
        # Line Filter for Scatter Graph
        col_scatter_1, col_scatter_2, col_scatter_3 = st.columns([2, 1, 1])
        with col_scatter_1:
            graph_line_option = st.selectbox("Select Line for Graph", ["All Lines"] + list(df['line_description'].unique()))
        
        # Data Preparation
        if graph_line_option == "All Lines":
            graph_data_source = df
        else:
            graph_data_source = df[df['line_description'] == graph_line_option]

        gcas_groups_scatter = graph_data_source.groupby('GCAS')
        scatter_points = []
        
        for gcas_id, group in gcas_groups_scatter:
            ppm = calculate_ppm(group)
            good_parts_count = group[group['is_good']]['count'].sum()
            
            # Get description
            desc = str(gcas_id)
            if 'GCAS_desc' in group.columns:
                 desc_vals = group['GCAS_desc'].dropna().unique()
                 if len(desc_vals) > 0:
                     desc = str(desc_vals[0])
            
            scatter_points.append({
                'GCAS': str(gcas_id), 
                'Description': desc,
                'PPM': ppm, 
                'Total Good Parts': good_parts_count,
            })
            
        df_scatter = pd.DataFrame(scatter_points)
        
        if not df_scatter.empty:
            df_scatter = df_scatter[df_scatter['PPM'] <= max_ppm_limit]
            
            if not df_scatter.empty:
                import altair as alt
                
                max_vol = df_scatter['Total Good Parts'].max()
                default_vol_thresh = int(max_vol / 2) if max_vol > 0 else 0
                
                # Threshold Inputs
                with col_scatter_2:
                    vol_threshold = st.number_input("Volume Threshold (X)", min_value=0, max_value=int(max_vol*1.1), value=default_vol_thresh, step=100)
                with col_scatter_3:
                    ppm_threshold = st.number_input("PPM Threshold (Y)", min_value=0, max_value=int(max_ppm_limit), value=2500, step=50)

                # Assign Quadrants
                # Q1: High PPM, High Vol (Top Right) - CRITICAL
                # Q2: High PPM, Low Vol (Top Left) - Sporadic
                # Q3: Low PPM, Low Vol (Bottom Left) - Insignificant
                # Q4: Low PPM, High Vol (Bottom Right) - GOOD
                
                def get_quadrant(row):
                    if row['PPM'] >= ppm_threshold:
                        if row['Total Good Parts'] >= vol_threshold:
                            return "Q1 (Critical)"
                        else:
                            return "Q2 (Sporadic)"
                    else:
                        if row['Total Good Parts'] >= vol_threshold:
                            return "Q4 (Good)"
                        else:
                            return "Q3 (Small Run)"

                df_scatter['Quadrant'] = df_scatter.apply(get_quadrant, axis=1)
                
                # Calculate Statistics per Quadrant
                total_gcas = len(df_scatter)
                total_volume = df_scatter['Total Good Parts'].sum()
                
                stats = []
                for q in ["Q1 (Critical)", "Q2 (Sporadic)", "Q3 (Small Run)", "Q4 (Good)"]:
                    q_data = df_scatter[df_scatter['Quadrant'] == q]
                    n_gcas = len(q_data)
                    vol_gcas = q_data['Total Good Parts'].sum()
                    stats.append({
                        "Quadrant": q, 
                        "% GCAS": (n_gcas / total_gcas) * 100 if total_gcas > 0 else 0,
                        "% Volume": (vol_gcas / total_volume) * 100 if total_volume > 0 else 0,
                        "Count": n_gcas
                    })
                
                # Plot Base
                base = alt.Chart(df_scatter).encode(
                    x=alt.X('Total Good Parts', title='Total Good Parts (Volume)'),
                    y=alt.Y('PPM', title='Defect Rate (PPM)'),
                    tooltip=['GCAS', 'Description', 'PPM', 'Total Good Parts', 'Quadrant']
                )

                # Points
                points = base.mark_circle(size=60).encode(
                    color=alt.Color('Quadrant', scale=alt.Scale(domain=["Q1 (Critical)", "Q2 (Sporadic)", "Q3 (Small Run)", "Q4 (Good)"], range=['red', 'orange', 'lightgrey', 'green']), legend=None)
                ).interactive()

                # Rule Lines
                h_rule = alt.Chart(pd.DataFrame({'y': [ppm_threshold]})).mark_rule(color='black', strokeDash=[5, 5]).encode(y='y')
                v_rule = alt.Chart(pd.DataFrame({'x': [vol_threshold]})).mark_rule(color='black', strokeDash=[5, 5]).encode(x='x')

                # Altair Theme Enforcement
                chart = (points + h_rule + v_rule).properties(height=500).configure(background=app_bg_color)
                
                if theme_mode: # Dark specifics if not fully handled by Streamlit native
                    chart = chart.configure_axis(
                        labelColor='white', titleColor='white', gridColor='#444'
                    ).configure_legend(
                        labelColor='white', titleColor='white'
                    ).configure_title(color='white')
                else: # Light specifics
                    chart = chart.configure_axis(
                        labelColor='black', titleColor='black', gridColor='#eee'
                    ).configure_legend(
                        labelColor='black', titleColor='black'
                    ).configure_title(color='black')

                st.altair_chart(chart, use_container_width=True)

                # Download Scatter Plot Image
                try:
                    png_scatter = vlc.vegalite_to_png(chart.to_json(), scale=2)
                    st.download_button("Download Scatter Plot as PNG", png_scatter, "gcas_scatter.png", "image/png", key='dl_img_scatter')
                except Exception as e:
                    # Fallback or silent fail if needed, but error is better for debug
                    st.caption(f"Image download unavailable: {e}")
                
                # Display Stats
                st.markdown("#### Quadrant Analysis")
                cols = st.columns(4)
                for i, stat in enumerate(stats):
                    with cols[i]:
                        # Determine label color based on quadrant type
                        if "Q1" in stat['Quadrant']:
                            color = "red"
                        elif "Q2" in stat['Quadrant']:
                            color = "orange"
                        elif "Q3" in stat['Quadrant']:
                            # Use a darker grey in light mode for better visibility, light grey in dark mode
                            color = "grey" if theme_mode else "#555555" 
                        elif "Q4" in stat['Quadrant']:
                            color = "green"
                        else:
                            color = app_text_color

                        st.markdown(f"<h3 style='text-align: center; color: {color};'>{stat['Quadrant']}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center; font-size: 1.1em; font-weight: bold;'>{stat['% GCAS']:.1f}% of GCAS</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center; font-size: 1.1em; font-weight: bold;'>{stat['% Volume']:.1f}% of Volume</p>", unsafe_allow_html=True)

            else:
                st.warning(f"No GCAS items below {max_ppm_limit} PPM limit.")
        else:
             st.info("No data available for graph.")


        # --- Top Offenders Table ---
        st.markdown('---')
        st.markdown('<div class="sub-header">Top GCAS Offenders (> 2500 PPM)</div>', unsafe_allow_html=True)
        
        # Filter for Table
        table_line_option = st.selectbox("Select Line for Top Offenders Table", ["All Lines"] + list(df['line_description'].unique()))
        
        if table_line_option == "All Lines":
            table_data_source = df
        else:
            table_data_source = df[df['line_description'] == table_line_option]
        
        # Re-calculate GCAS data (Same logic as above, reused)
        gcas_groups_filtered = table_data_source.groupby('GCAS')
        gcas_table_data = []
        
        for gcas_id, group in gcas_groups_filtered:
            ppm = calculate_ppm(group)
            good_parts_sum = group[group['is_good']]['count'].sum()
            desc = "N/A"
            if 'GCAS_desc' in group.columns:
                 desc_vals = group['GCAS_desc'].dropna().unique()
                 if len(desc_vals) > 0:
                     desc = desc_vals[0]
            
            gcas_table_data.append({
                'GCAS': gcas_id, 
                'Description': desc,
                'Total Good Parts': good_parts_sum,
                'PPM': ppm
            })
            
        df_gcas_table = pd.DataFrame(gcas_table_data).sort_values('PPM', ascending=False)
        
        offenders = df_gcas_table[df_gcas_table['PPM'] > 2500]
        
        if not offenders.empty:
            st.info(f"Showing {len(offenders)} offenders for {table_line_option}")
            
            # View Toggle
            gcas_view_format = st.radio("View Format", ["Table", "Bar Graph"], horizontal=True, key="gcas_offenders_view_format")
            
            if gcas_view_format == "Table":
                st.dataframe(
                    offenders.style.format({'PPM': "{:.0f}", 'Total Good Parts': "{:.0f}"})
                         .set_properties(**{'background-color': app_bg_color, 'color': app_text_color}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                # Altair Bar Chart
                # Combine GCAS and Description for Label
                offenders['Label'] = offenders['GCAS'].astype(str) + " - " + offenders['Description'].astype(str)
                
                gcas_chart = alt.Chart(offenders).mark_bar().encode(
                    x=alt.X('PPM', title='PPM'),
                    y=alt.Y('Label', sort='-x', title='GCAS - Description', axis=alt.Axis(labelLimit=1000)),
                    tooltip=['GCAS', 'Description', alt.Tooltip('PPM', format=',.0f'), alt.Tooltip('Total Good Parts', format=',.0f')]
                ).properties(
                    title=f"Top GCAS Offenders ({table_line_option})",
                    height=max(400, len(offenders) * 30)
                ).configure(background=app_bg_color)
                
                if theme_mode:
                    gcas_chart = gcas_chart.configure_axis(
                        labelColor='white', titleColor='white', gridColor='#444'
                    ).configure_title(color='white')
                else:
                    gcas_chart = gcas_chart.configure_axis(
                        labelColor='black', titleColor='black', gridColor='#eee'
                    ).configure_title(color='black')
                
                st.altair_chart(gcas_chart, use_container_width=True)
                
                # Download GCAS Graph Image
                try:
                    png_gcas = vlc.vegalite_to_png(gcas_chart.to_json(), scale=2)
                    st.download_button("Download Graph as PNG", png_gcas, "gcas_offenders_graph.png", "image/png", key='dl_img_gcas')
                except Exception as e:
                    st.error(f"Could not generate image: {e}")
            
            # Download Buttons
            csv = offenders.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Table as CSV",
                csv,
                "gcas_offenders.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            if gcas_data_source.empty:
                st.warning("No data found for this selection.")
            else:
                 st.success("No GCAS exceeds the 2500 PPM limit!")


        # --- Graph 3: Weekly PPM Trend ---
        st.markdown('---')
        st.markdown('<div class="sub-header">3. Weekly PPM Trend</div>', unsafe_allow_html=True)
        
        # Line Filter for Weekly Graph
        weekly_line_option = st.selectbox("Select Line for Trend Graph", ["All Lines"] + list(df['line_description'].unique()), key='weekly_filter')
        
        if weekly_line_option == "All Lines":
            weekly_data_source = df
        else:
            weekly_data_source = df[df['line_description'] == weekly_line_option]
            
        if 'DATE' in weekly_data_source.columns:
            weekly_data_source['Week_Num'] = weekly_data_source['DATE'].dt.isocalendar().week
            weekly_data_source['Year'] = weekly_data_source['DATE'].dt.year
            
            # Group by Year-Week
            weekly_groups = weekly_data_source.groupby(['Year', 'Week_Num'])
            weekly_points = []
            
            for (year, week), group in weekly_groups:
                ppm = calculate_ppm(group)
                weekly_points.append({
                    'Year': year,
                    'Week': week,
                    'Year-Week': f"{year}-W{week}",
                    'PPM': ppm
                })
                
            df_weekly = pd.DataFrame(weekly_points)
            
            if not df_weekly.empty:
                 # Sort by Year then Week
                 df_weekly = df_weekly.sort_values(['Year', 'Week'])
                 
                 # Altair Trend Line
                 base_trend = alt.Chart(df_weekly).encode(
                     x=alt.X('Year-Week', sort=None, title='Week'),
                     y=alt.Y('PPM', title='PPM'),
                     tooltip=['Year-Week', 'PPM']
                 )
                 
                 # Main Line (Thicker)
                 trend_line = base_trend.mark_line(point=False, strokeWidth=3).encode(
                     color=alt.value('#1E88E5')  # Consistent blue
                 )
                 
                 # Points on top
                 trend_points = base_trend.mark_circle(size=60).encode(
                     color=alt.value('#1E88E5')
                 )
                 
                 # Limit Lines
                 limit_2500 = alt.Chart(pd.DataFrame({'y': [2500]})).mark_rule(color='orange', strokeDash=[5, 5], strokeWidth=2).encode(y='y')
                 limit_5000 = alt.Chart(pd.DataFrame({'y': [5000]})).mark_rule(color='red', strokeDash=[5, 5], strokeWidth=2).encode(y='y')
                 
                 # Combine
                 final_chart = (trend_line + trend_points + limit_2500 + limit_5000).properties(
                     title=f"Weekly PPM Trend ({weekly_line_option})",
                     height=400
                 ).configure(background=app_bg_color)
                 
                 # Apply Theme
                 if theme_mode:
                    final_chart = final_chart.configure_axis(
                        labelColor='white', titleColor='white', gridColor='#444'
                    ).configure_title(color='white')
                 else:
                    final_chart = final_chart.configure_axis(
                        labelColor='black', titleColor='black', gridColor='#eee'
                    ).configure_title(color='black')
                    
                 st.altair_chart(final_chart, use_container_width=True)
                 
                 # Download Weekly Trend Image
                 try:
                    png_trend = vlc.vegalite_to_png(final_chart.to_json(), scale=2)
                    st.download_button("Download Trend Graph as PNG", png_trend, "weekly_trend.png", "image/png", key='dl_img_trend')
                 except Exception as e:
                    st.caption(f"Image download unavailable: {e}")

            else:
                st.warning("No data available to plot weekly trend.")
        else:
            st.error("DATE column missing. Cannot calculate weekly trend.")


        # --- Section 4: Defect Analysis Table ---
        st.markdown('---')
        st.markdown('<div class="sub-header">4. Hand Written Defect Analysis</div>', unsafe_allow_html=True)
        
        # Filter
        defects_line_option = st.selectbox("Select Line for Defect Analysis", ["All Lines"] + list(df['line_description'].unique()), key='defects_filter')
        
        if defects_line_option == "All Lines":
            defects_data_source = df
        else:
            defects_data_source = df[df['line_description'] == defects_line_option]
            
        # 1. Total Good Parts (Denominator for PPM)
        # We need the sum of all Good Parts in the selected scope to calculate the "Impact" of each defect type relative to total production
        total_good_parts_scope = defects_data_source[defects_data_source['is_good']]['count'].sum()
        
        # 2. Filter for Defects Only
        # All rows marked as is_defect
        defect_rows = defects_data_source[defects_data_source['is_defect']]
        
        if not defect_rows.empty and total_good_parts_scope > 0:
            # Normalize Defect Description to remove Operator names
            # Copy to avoid SettingWithCopyWarning
            defect_analysis_df = defect_rows.copy()
            
            # Remove "Operator 1", "Operator 2", etc. (case insensitive)
            defect_analysis_df['normalized_desc'] = defect_analysis_df['counter_type_description'].astype(str).str.replace(r'operator\s*[0-9]+', '', case=False, regex=True)
            
            # Remove "Hand written errors" (case insensitive)
            defect_analysis_df['normalized_desc'] = defect_analysis_df['normalized_desc'].str.replace(r'hand written errors', '', case=False, regex=True)
            
            # Clean up whitespace
            defect_analysis_df['normalized_desc'] = defect_analysis_df['normalized_desc'].str.strip()
            defect_analysis_df['normalized_desc'] = defect_analysis_df['normalized_desc'].str.replace(r'\s+', ' ', regex=True)

            # Group by Normalized Defect Description
            defect_groups = defect_analysis_df.groupby('normalized_desc')['count'].sum().reset_index()
            
            # Calculate PPM Impact for each defect type
            # PPM = (Defect Count / (Defect Count + Total Good)) * 1,000,000
            # Note: Usage of (Defect + Good) in denominator is standard, 
            # but usually for *specific* defect rate we might just use Total Parts (Good + All Defects). 
            # Let's stick to the simple approximation: Impact PPM = (This Defect Count / Total Good) * 1M for ranking purposes, 
            # or better: (This Defect Count / (Total Good + Total Defects this scope)) * 1M.
            
            total_parts_scope = total_good_parts_scope + defect_rows['count'].sum()
            
            defect_groups['PPM Impact'] = (defect_groups['count'] / total_parts_scope) * 1_000_000
            
            # Rename for display
            defect_groups = defect_groups.rename(columns={'normalized_desc': 'Defect Description', 'count': 'Defect Count'})
            
            # Sort
            defect_groups = defect_groups.sort_values('PPM Impact', ascending=False)
            
            st.info(f"Showing defect breakdown for {defects_line_option}")
            
            # View Toggle
            view_format = st.radio("View Format", ["Table", "Bar Graph"], horizontal=True, key="defect_view_format")
            
            if view_format == "Table":
                st.dataframe(
                    defect_groups.style.format({'PPM Impact': "{:.0f}", 'Defect Count': "{:.0f}"})
                         .set_properties(**{'background-color': app_bg_color, 'color': app_text_color}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                # Bar Graph View
                bar_chart = alt.Chart(defect_groups).mark_bar().encode(
                    x=alt.X('PPM Impact', title='PPM Impact'),
                    y=alt.Y('Defect Description', sort='-x', title='Defect Description', axis=alt.Axis(labelLimit=1000)),
                    tooltip=['Defect Description', 'Defect Count', alt.Tooltip('PPM Impact', format=',.0f')]
                ).properties(
                    title=f"Hand Written Defect Impact ({defects_line_option})",
                    height=max(400, len(defect_groups) * 30) # Dynamic height based on number of bars
                ).configure(background=app_bg_color)
                
                # Apply Theme
                if theme_mode:
                    bar_chart = bar_chart.configure_axis(
                        labelColor='white', titleColor='white', gridColor='#444'
                    ).configure_title(color='white')
                else:
                    bar_chart = bar_chart.configure_axis(
                        labelColor='black', titleColor='black', gridColor='#eee'
                    ).configure_title(color='black')
                
                st.altair_chart(bar_chart, use_container_width=True)
                
                # Download Defect Graph Image
                try:
                    png_defect = vlc.vegalite_to_png(bar_chart.to_json(), scale=2)
                    st.download_button("Download Graph as PNG", png_defect, "defect_analysis_graph.png", "image/png", key='dl_img_defect')
                except Exception as e:
                    st.error(f"Could not generate image: {e}")
            
            # Download Button for Defect Analysis
            csv_defects = defect_groups.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Defect Analysis as CSV",
                csv_defects,
                "defect_analysis.csv",
                "text/csv",
                key='download-defect-analysis'
            )
        else:
             if total_good_parts_scope == 0:
                 st.warning("Total Good Parts is 0, cannot calculate PPM impact.")
             else:
                 st.success("No defects found in this selection.")


    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Awaiting CSV file upload in the sidebar.")
    st.write("If you don't have a file, one has been generated as `dummy_data.csv` in your folder.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; padding: 20px 0;'>
        Copyrights Khaled Senan 19 Jan 2026
    </div>
    """,
    unsafe_allow_html=True
)
