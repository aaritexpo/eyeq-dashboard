import os
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, Input, Output

BASE_DIR = './EyeQ'

# --- DATA LOADING FUNCTION ---
def load_data(base_dir):
    eyeq_data = pd.read_csv(os.path.join(base_dir, 'EyeQ Data - Master.csv'))
    vimssq_data = eyeq_data[['Participant ID', 'VIMSSQ-Intake', 'Age']].drop_duplicates(subset=['Participant ID'])

    # Build location lookup: map Morning->1, Afternoon->2 to match Daily Session
    session_map = {'Morning': 1, 'Afternoon': 2}
    location_data = eyeq_data[['Participant ID', 'Day', 'Session', 'Location']].dropna(subset=['Location']).drop_duplicates()
    location_data['Daily Session'] = location_data['Session'].map(session_map)
    location_data = location_data.rename(columns={'Day': 'Study Day'})
    location_data = location_data[['Participant ID', 'Study Day', 'Daily Session', 'Location']]

    print(f"Searching for data in {base_dir}...")
    all_files = [] 
    
    # Walk through directories
    for root, dirs, files in os.walk(base_dir):
        #Skip the "DQ" and "Dry runs"
        if 'DQ' in dirs:
            dirs.remove('DQ')
        if 'Dry runs' in dirs:
            dirs.remove('Dry runs')
            
        #Check if we are in a 'final' folder
        if os.path.basename(root) == 'final':
            #Check if the parent folder (PID) is a number
            parent_folder = os.path.basename(os.path.dirname(root))
            if parent_folder.isdigit():
                for file in files:
                    #Check if the file name contains 'responses_final'
                    if 'responses_final' in file and file.endswith(".csv"):
                        all_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_files)} valid CSV files.")
    
    df_list = []
    for filename in all_files:
        try:
            temp_df = pd.read_csv(filename)
            
            #Basic Cleaning
            temp_df['Response'] = pd.to_numeric(temp_df['Response'], errors='coerce')
            
            # Filter Missed
            if 'Missed' in temp_df.columns:
                temp_df = temp_df[temp_df['Missed'] != True]
            
            temp_df = temp_df.dropna(subset=['Response'])
            
            #Use Target Time for binning (Seconds -> Minutes)
            if 'Target Time (seconds)' in temp_df.columns:
                temp_df['Time_Minutes'] = temp_df['Target Time (seconds)'] / 60
            else:
                # Fallback if column is missing, though it shouldn't be based on specs
                temp_df['Time_Minutes'] = temp_df['Global Time (seconds)'] / 60
            
            temp_df = temp_df.merge(vimssq_data[['Participant ID', 'VIMSSQ-Intake', 'Age']], on='Participant ID', how='left')
            temp_df = temp_df.rename(columns={'VIMSSQ-Intake': 'VIMSSQ Score'})
            temp_df = temp_df.merge(location_data, on=['Participant ID', 'Study Day', 'Daily Session'], how='left')
            df_list.append(temp_df)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not df_list:
        # Create an empty dummy DF to prevent app crash on load if no data found
        return pd.DataFrame(columns=['Participant ID', 'Survey Type', 'Time_Minutes', 'Response', 'Device Type', 'Session Type', 'VIMSSQ Score'])

    master_df = pd.concat(df_list, ignore_index=True)
    
    # Create combined Session Type x Location variable
    master_df['Session Type x Location'] = master_df.apply(
        lambda r: f"{r['Session Type']} ({r['Location']})"
        if pd.notna(r.get('Location')) else r['Session Type'],
        axis=1
    )

    # Age bin: Young (20-35) vs Older (50+)
    master_df['Age Bin'] = master_df['Age'].apply(
        lambda x: 'Young (20-35)' if pd.notna(x) and x <= 35
        else ('Older (50+)' if pd.notna(x) and x >= 50 else None)
    )

    # Ensure categorical columns are strings for cleaner plotting
    master_df['Participant ID'] = master_df['Participant ID'].astype(str)
    return master_df

# --- SSQ DATA LOADING ---
def load_ssq_data(base_dir, response_df):
    """Load SSQ data from master CSV, reshape to long format, and merge session metadata."""
    eyeq_data = pd.read_csv(os.path.join(base_dir, 'EyeQ Data - Master.csv'))

    ssq_cols_pre = ['SSQ-T-Pre', 'SSQ-N-Pre', 'SSQ-O-Pre', 'SSQ-D-Pre']
    ssq_cols_post = ['SSQ-T-Post', 'SSQ-N-Post', 'SSQ-O-Post', 'SSQ-D-Post']
    id_cols = ['Participant ID', 'Day', 'Session']

    # Only keep PIDs that are in the response data
    valid_pids = response_df['Participant ID'].unique()
    valid_pids_int = [int(p) for p in valid_pids]
    ssq_raw = eyeq_data[eyeq_data['Participant ID'].isin(valid_pids_int)].copy()

    # Map Morning/Afternoon to Daily Session 1/2
    session_map = {'Morning': 1, 'Afternoon': 2}
    ssq_raw['Daily Session'] = ssq_raw['Session'].map(session_map)
    ssq_raw = ssq_raw.rename(columns={'Day': 'Study Day'})

    # Get session metadata from response data
    session_lookup = response_df[['Participant ID', 'Study Day', 'Daily Session',
                                   'Device Type', 'Session Type', 'Moderator Initials',
                                   'Location', 'Session Type x Location',
                                   'VIMSSQ Score', 'Age', 'Age Bin']].drop_duplicates()
    session_lookup['Participant ID'] = session_lookup['Participant ID'].astype(int)

    ssq_raw = ssq_raw.merge(session_lookup,
                            on=['Participant ID', 'Study Day', 'Daily Session'],
                            how='left')

    meta_cols = ['Participant ID', 'Study Day', 'Daily Session', 'Device Type',
                 'Session Type', 'Moderator Initials', 'Location',
                 'Session Type x Location', 'VIMSSQ Score', 'Age', 'Age Bin']
    # Only keep meta columns that actually exist after the merge
    meta_cols = [c for c in meta_cols if c in ssq_raw.columns]

    # Melt Pre scores
    ssq_pre = ssq_raw[meta_cols + ssq_cols_pre].melt(
        id_vars=meta_cols, var_name='Subscale', value_name='Score')
    ssq_pre['Timing'] = 'Pre'
    ssq_pre['Subscale'] = ssq_pre['Subscale'].str.replace('SSQ-', '').str.replace('-Pre', '')

    # Melt Post scores
    ssq_post = ssq_raw[meta_cols + ssq_cols_post].melt(
        id_vars=meta_cols, var_name='Subscale', value_name='Score')
    ssq_post['Timing'] = 'Post'
    ssq_post['Subscale'] = ssq_post['Subscale'].str.replace('SSQ-', '').str.replace('-Post', '')

    # Enforce pairing: only keep sessions where BOTH Pre and Post exist
    pair_keys = ['Participant ID', 'Study Day', 'Daily Session', 'Subscale']
    pre_valid = ssq_pre.dropna(subset=['Score'])[pair_keys]
    post_valid = ssq_post.dropna(subset=['Score'])[pair_keys]
    valid_pairs = pre_valid.merge(post_valid, on=pair_keys)

    ssq_pre = ssq_pre.merge(valid_pairs, on=pair_keys)
    ssq_post = ssq_post.merge(valid_pairs, on=pair_keys)

    ssq_long = pd.concat([ssq_pre, ssq_post], ignore_index=True)

    subscale_labels = {'T': 'Total', 'N': 'Nausea', 'O': 'Oculomotor', 'D': 'Disorientation'}
    ssq_long['Subscale'] = ssq_long['Subscale'].map(subscale_labels)
    ssq_long['Participant ID'] = ssq_long['Participant ID'].astype(str)

    return ssq_long


# Load Data Once on Startup
df = load_data(BASE_DIR)
ssq_df = load_ssq_data(BASE_DIR, df)
available_pids = sorted(df['Participant ID'].unique())
survey_types = list(df['Survey Type'].unique()) + ['SSQ']

# --- DASH APP SETUP ---
app = dash.Dash(__name__)
server = app.server  # Expose server for Render (Gunicorn)

# Options for dropdowns
variable_options = [
    {'label': 'None', 'value': 'None'},
    {'label': 'Device Type', 'value': 'Device Type'},
    {'label': 'Session Type', 'value': 'Session Type'},
    {'label': 'Participant ID', 'value': 'Participant ID'},
    {'label': 'Moderator Initials', 'value': 'Moderator Initials'},
    {'label': 'Study Day', 'value': 'Study Day'},
    {'label': 'Daily Session', 'value': 'Daily Session'},
    {'label': 'VIMSSQ Bin', 'value': 'VIMSSQ Bin'},
    {'label': 'Session Type x Location', 'value': 'Session Type x Location'},
    {'label': 'Age Bin', 'value': 'Age Bin'},
]

# --- LAYOUT ---
app.layout = html.Div([
    html.H1("EyeQ Study Results Dashboard", style={'textAlign': 'center'}),

    html.Div([
        # Left side: Graph Area
        html.Div([
            dcc.Graph(
                id='main-graph', 
                style={'height': '85vh', 'width': '100%'},
                config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'scale': 3,  # Higher DPI (3x default resolution)
                        'filename': 'eyeq_plot'
                    }
                }
            )
        ], style={
            'width': '75%', 
            'display': 'inline-block', 
            'verticalAlign': 'top'
        }),

        # Right side: Control Panel
        html.Div([
            html.H3("Controls", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.Label("1. Select Survey Type:"),
            dcc.Dropdown(
                id='survey-dropdown',
                options=[{'label': s.upper() if s == 'SSQ' else s.capitalize(), 'value': s} for s in survey_types],
                value=survey_types[0] if len(survey_types) > 0 else None,
                clearable=False
            ),
            html.Br(),
            
            html.Label("2. Select Participants (Leave empty for ALL):"),
            dcc.Dropdown(
                id='pid-dropdown',
                options=[{'label': pid, 'value': pid} for pid in available_pids],
                value=[], # Default empty means all
                multi=True
            ),
            html.Br(),
            
            html.Label("VIMSSQ Threshold (for binning):"),
            dcc.Input(
                id='vimssq-threshold',
                type='number',
                value=22,
                placeholder='Enter threshold value',
                style={'width': '100%'}
            ),
            html.Br(),
            html.Br(),
            
            dcc.Checklist(
                id='exclude-vomiting-checkbox',
                options=[{'label': ' Exclude vomiting (MISC = 10)', 'value': 'exclude'}],
                value=[]
            ),
            html.Br(),
            dcc.Checklist(
                id='show-sem-checkbox',
                options=[{'label': ' Show SEM', 'value': 'show'}],
                value=['show']  # Checked by default
            ),
            html.Br(),
            html.Hr(),
            
            html.Label("3. Color Lines By:"),
            dcc.Dropdown(
                id='color-dropdown',
                options=variable_options,
                value='Device Type',
                clearable=False
            ),
            html.Br(),
            
            html.Label("4. Facet Columns By:"),
            dcc.Dropdown(
                id='facet-col-dropdown',
                options=variable_options,
                value='None',
                clearable=False
            ),
            html.Br(),
            
            html.Label("5. Facet Rows By:"),
            dcc.Dropdown(
                id='facet-row-dropdown',
                options=variable_options,
                value='None',
                clearable=False
            ),
        ], style={
            'width': '25%', 
            'display': 'inline-block', 
            'verticalAlign': 'top', 
            'padding': '20px',
            'borderLeft': '2px solid #ccc',
            'height': '85vh',
            'overflowY': 'auto'
        }),
    ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'row'})
])

# --- CALLBACKS ---
@app.callback(
    Output('main-graph', 'figure'),
    [Input('survey-dropdown', 'value'),
     Input('pid-dropdown', 'value'),
     Input('color-dropdown', 'value'),
     Input('facet-col-dropdown', 'value'),
     Input('facet-row-dropdown', 'value'),
     Input('exclude-vomiting-checkbox', 'value'),
     Input('vimssq-threshold', 'value'),
     Input('show-sem-checkbox', 'value')]
)
def update_graph(selected_survey, selected_pids, color_var, facet_col, facet_row, exclude_vomiting, vimssq_threshold, show_sem):

    if selected_survey == 'SSQ':
        return build_ssq_figure(selected_pids, color_var, facet_row,
                                vimssq_threshold, show_sem)

    # --- Standard survey path (misc, eyestrain, etc.) ---
    # 1. Filter by Survey
    filtered_df = df[df['Survey Type'] == selected_survey].copy()

    # Create VIMSSQ Bin column based on threshold
    if vimssq_threshold is not None:
        filtered_df['VIMSSQ Bin'] = filtered_df['VIMSSQ Score'].apply(
            lambda x: 'High' if pd.notna(x) and x >= vimssq_threshold else 'Low' if pd.notna(x) else None
        )

    # 2. Filter by PIDs (if any selected)
    if selected_pids:
        filtered_df = filtered_df[filtered_df['Participant ID'].isin(selected_pids)]

    # 3. Exclude vomiting if checkbox is checked
    if 'exclude' in exclude_vomiting:
        outlier_threshold = 10
        filtered_df = filtered_df[filtered_df['Response'] < outlier_threshold]

    # 4. Dynamic Aggregation
    # We always group by Time
    group_cols = ['Time_Minutes']

    # Add dynamic grouping variables if they are not 'None'
    if color_var != 'None': group_cols.append(color_var)
    if facet_col != 'None': group_cols.append(facet_col)
    if facet_row != 'None': group_cols.append(facet_row)

    # Remove duplicates in group_cols (e.g. if Color and Facet Col are same)
    group_cols = list(set(group_cols))

    # Calculate Mean and Standard Error
    agg_df = filtered_df.groupby(group_cols)['Response'].agg(['mean', 'sem', 'count']).reset_index()

    # 5. Plotting
    c = color_var if color_var != 'None' else None
    fc = facet_col if facet_col != 'None' else None
    fr = facet_row if facet_row != 'None' else None

    title = f"Average {selected_survey.capitalize()} Response over Time"
    if selected_pids:
        title += f" (n={len(selected_pids)})"
    else:
        title += f" (n={filtered_df['Participant ID'].nunique()})"

    error_y_val = 'sem' if 'show' in show_sem else None

    fig = px.line(
        agg_df,
        x='Time_Minutes',
        y='mean',
        error_y=error_y_val,
        color=c,
        facet_col=fc,
        facet_row=fr,
        markers=True,
        title=title,
        hover_data={'count': True},
        labels={
            'Time_Minutes': 'Time (minutes)',
            'mean': 'Score',
            'sem': 'Standard Error',
            'Device Type': 'Device Type',
            'Session Type': 'Session Type',
            'Participant ID': 'Participant ID',
            'Moderator Initials': 'Moderator Initials',
            'Study Day': 'Study Day',
            'Daily Session': 'Daily Session',
            'VIMSSQ Bin': 'VIMSSQ Bin',
            'Session Type x Location': 'Session Type x Location',
            'Age Bin': 'Age Bin'
        }
    )

    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        title_xanchor='center',
        font=dict(size=14),
        title_font=dict(size=20),
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            title_font=dict(size=16),
            font=dict(size=14),
            title=dict(
                side='top',
                font=dict(size=16)
            )
        ),
        xaxis=dict(title_font=dict(size=16)),
        yaxis=dict(title_font=dict(size=16))
    )

    for annotation in fig.layout.annotations:
        if "=" in annotation.text:
            split_text = annotation.text.split("=")
            if len(split_text) == 2:
                annotation.text = split_text[1].strip()

    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=12))
    fig.update_yaxes(matches='y', title_font=dict(size=16), tickfont=dict(size=12))

    return fig


def build_ssq_figure(selected_pids, color_var, facet_row, vimssq_threshold, show_sem):
    """SSQ Pre vs Post, faceted by Subscale (columns), with optional color and row facet."""

    filtered = ssq_df.copy()

    # VIMSSQ binning
    if vimssq_threshold is not None:
        filtered['VIMSSQ Bin'] = filtered['VIMSSQ Score'].apply(
            lambda x: 'High' if pd.notna(x) and x >= vimssq_threshold
            else ('Low' if pd.notna(x) else None)
        )

    # Filter by PIDs
    if selected_pids:
        filtered = filtered[filtered['Participant ID'].isin(selected_pids)]

    # Aggregation: always group by Subscale + Timing
    c = color_var if color_var != 'None' else None
    fr = facet_row if facet_row != 'None' else None

    group_cols = ['Subscale', 'Timing']
    if c:
        group_cols.append(c)
    if fr:
        group_cols.append(fr)
    group_cols = list(set(group_cols))

    agg = filtered.groupby(group_cols)['Score'].agg(['mean', 'sem', 'count']).reset_index()

    error_y_val = 'sem' if 'show' in show_sem else None

    fig = px.line(
        agg,
        x='Timing',
        y='mean',
        error_y=error_y_val,
        color=c,
        facet_col='Subscale',
        facet_row=fr,
        markers=True,
        category_orders={
            'Timing': ['Pre', 'Post'],
            'Subscale': ['Total', 'Nausea', 'Oculomotor', 'Disorientation'],
        },
        hover_data={'count': True},
        labels={
            'mean': 'SSQ Score',
            'sem': 'Standard Error',
            'Timing': '',
        }
    )

    title = "SSQ Pre vs Post"
    if selected_pids:
        title += f" (n={len(selected_pids)})"
    else:
        title += f" (n={filtered['Participant ID'].nunique()})"

    fig.update_traces(mode="lines+markers", marker_size=10)
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20)),
        font=dict(size=14),
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            title_font=dict(size=16),
            font=dict(size=14),
            title=dict(side='top', font=dict(size=16))
        ),
        xaxis=dict(title_font=dict(size=16)),
        yaxis=dict(title_font=dict(size=16)),
    )

    # Clean up facet labels (remove "Subscale=")
    for annotation in fig.layout.annotations:
        if "=" in annotation.text:
            split_text = annotation.text.split("=")
            if len(split_text) == 2:
                annotation.text = split_text[1].strip()

    # Let each subscale have its own y-axis scale
    fig.update_yaxes(matches=None, showticklabels=True,
                     title_font=dict(size=16), tickfont=dict(size=12))
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=12))

    return fig

if __name__ == '__main__':
    # Debug mode is great for local dev, turn off for prod if desired
    app.run(debug=True, use_reloader=False, port=8052)