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
    vimssq_data = eyeq_data[['Participant ID', 'VIMSSQ-Intake']].drop_duplicates(subset=['Participant ID'])

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
            
            temp_df = temp_df.merge(vimssq_data[['Participant ID', 'VIMSSQ-Intake']], on='Participant ID', how='left')
            temp_df = temp_df.rename(columns={'VIMSSQ-Intake': 'VIMSSQ Score'})
            df_list.append(temp_df)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not df_list:
        # Create an empty dummy DF to prevent app crash on load if no data found
        return pd.DataFrame(columns=['Participant ID', 'Survey Type', 'Time_Minutes', 'Response', 'Device Type', 'Session Type', 'VIMSSQ Score'])

    master_df = pd.concat(df_list, ignore_index=True)
    
    # Ensure categorical columns are strings for cleaner plotting
    master_df['Participant ID'] = master_df['Participant ID'].astype(str)
    return master_df

# Load Data Once on Startup
df = load_data(BASE_DIR)
available_pids = sorted(df['Participant ID'].unique())
survey_types = df['Survey Type'].unique()

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
                options=[{'label': s.capitalize(), 'value': s} for s in survey_types],
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
    # We use numeric_only=True to ensure we don't try to average string columns
    agg_df = filtered_df.groupby(group_cols)['Response'].agg(['mean', 'sem', 'count']).reset_index()

    # 5. Plotting
    # Map 'None' inputs to Python None for Plotly
    c = color_var if color_var != 'None' else None
    fc = facet_col if facet_col != 'None' else None
    fr = facet_row if facet_row != 'None' else None

    title = f"Average {selected_survey.capitalize()} Response over Time"
    if selected_pids:
        title += f" (n={len(selected_pids)})"
    else:
        title += f" (n={filtered_df['Participant ID'].nunique()})"

    # Conditionally add error bars based on checkbox
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
            'VIMSSQ Bin': 'VIMSSQ Bin'
        }
    )
    
    # Improve styling
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        template="plotly_white",
        title_x=0.5,  # Center the title
        title_xanchor='center',  # Anchor title at center
        # Increase font sizes
        font=dict(size=14),
        title_font=dict(size=20),
        # Legend positioning and styling
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
        # Increase axis label font sizes
        xaxis=dict(title_font=dict(size=16)),
        yaxis=dict(title_font=dict(size=16))
    )
    
    for annotation in fig.layout.annotations:
        if "=" in annotation.text:
            split_text = annotation.text.split("=")
            if len(split_text) == 2:
                annotation.text = split_text[1].strip()

    # Update all axes in case of faceting
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=12))
    fig.update_yaxes(matches='y', title_font=dict(size=16), tickfont=dict(size=12))
    
    return fig

if __name__ == '__main__':
    # Debug mode is great for local dev, turn off for prod if desired
    app.run(debug=True, use_reloader=False, port=8052)