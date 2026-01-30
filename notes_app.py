import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State

BASE_DIR = './EyeQ'

# --- DATA LOADING FUNCTION ---
def load_notes_data(base_dir):
    """
    Load all notes CSV files from participant directories.
    Returns a dataframe with columns: Participant ID, Timestamp, Global Time (seconds), Note, Day, Session
    """
    print(f"Searching for notes data in {base_dir}...")
    
    # Find all subdirectories that are numeric (participant IDs)
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d)) and re.match(r'^\d+$', d)]
    
    all_notes_list = []
    
    for subdir in subdirs:
        final_dir = os.path.join(base_dir, subdir, 'final')
        if os.path.exists(final_dir):
            for file in os.listdir(final_dir):
                if file.endswith('.csv') and 'notes' in file.lower():
                    file_path = os.path.join(final_dir, file)
                    
                    # Extract Day and Session from filename (e.g., "1029_D2S2_notes_final_...")
                    match = re.search(r'_D(\d+)S(\d+)_', file)
                    if match:
                        day = int(match.group(1))
                        session = int(match.group(2))
                    else:
                        # Fallback if pattern not found
                        day = None
                        session = None
                    
                    try:
                        temp_df = pd.read_csv(file_path)
                        temp_df['Day'] = day
                        temp_df['Session'] = session
                        temp_df['Source File'] = file
                        pid = temp_df['Participant ID'][0]
                        #print(pid)
                        all_notes_list.append(temp_df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    if not all_notes_list:
        print("No notes files found.")
        return pd.DataFrame(columns=['Participant ID', 'Timestamp', 'Global Time (seconds)', 'Note', 'Day', 'Session'])
    
    all_notes_df = pd.concat(all_notes_list, ignore_index=True)
    
    # Ensure proper types
    all_notes_df['Participant ID'] = all_notes_df['Participant ID'].astype(str)
    all_notes_df['Day'] = all_notes_df['Day'].astype('Int64')  # Nullable int
    all_notes_df['Session'] = all_notes_df['Session'].astype('Int64')
    
    # Convert time to minutes
    all_notes_df['Time_Minutes'] = all_notes_df['Global Time (seconds)'] / 60
    
    print(f"Loaded {len(all_notes_df)} notes from {len(all_notes_list)} files.")
    return all_notes_df


# Load Data Once on Startup
df = load_notes_data(BASE_DIR)
df['Participant ID'] = df['Participant ID'].astype(float)
available_pids = sorted(df['Participant ID'].unique())

# --- DASH APP SETUP ---
notes_app = dash.Dash(__name__)
server = notes_app.server  # Expose server for deployment

# --- LAYOUT ---
notes_app.layout = html.Div([
    html.H1("EyeQ Session Notes Timeline", style={'textAlign': 'center'}),

    html.Div([
        # Left side: Graph Area
        html.Div([
            dcc.Graph(
                id='notes-timeline', 
                style={'height': '85vh', 'width': '100%'},
                config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'scale': 3,
                        'filename': 'eyeq_notes_timeline'
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
            
            html.Label("1. Select Participant ID:"),
            dcc.Dropdown(
                id='pid-dropdown',
                options=[{'label': pid, 'value': pid} for pid in available_pids],
                value=available_pids[0] if len(available_pids) > 0 else None,
                clearable=False
            ),
            html.Br(),
            
            html.Label("2. Select Day:"),
            dcc.Dropdown(
                id='day-dropdown',
                options=[],  # Populated by callback
                value=None,
                clearable=False
            ),
            html.Br(),
            
            html.Label("3. Select Session:"),
            dcc.Dropdown(
                id='session-dropdown',
                options=[],  # Populated by callback
                value=None,
                clearable=False
            ),
            html.Br(),
            html.Br(),
            
            html.Button(
                'Show Notes', 
                id='show-notes-btn', 
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '10px',
                    'fontSize': '16px',
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer'
                }
            ),
            html.Br(),
            html.Br(),
            
            html.Hr(),
            
            # Info panel showing note count
            html.Div(id='info-panel', style={'marginTop': '20px', 'fontSize': '14px'})
            
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

# Callback 1: Update Day dropdown based on selected PID
@notes_app.callback(
    [Output('day-dropdown', 'options'),
     Output('day-dropdown', 'value')],
    [Input('pid-dropdown', 'value')]
)
def update_day_options(selected_pid):
    if selected_pid is None:
        return [], None
    
    # Filter data for selected PID
    pid_df = df[df['Participant ID'] == selected_pid]
    available_days = sorted(pid_df['Day'].dropna().unique())
    
    options = [{'label': f'Day {d}', 'value': d} for d in available_days]
    default_value = available_days[0] if len(available_days) > 0 else None
    
    return options, default_value


# Callback 2: Update Session dropdown based on selected PID and Day
@notes_app.callback(
    [Output('session-dropdown', 'options'),
     Output('session-dropdown', 'value')],
    [Input('pid-dropdown', 'value'),
     Input('day-dropdown', 'value')]
)
def update_session_options(selected_pid, selected_day):
    if selected_pid is None or selected_day is None:
        return [], None
    
    # Filter data for selected PID and Day
    filtered_df = df[(df['Participant ID'] == selected_pid) & (df['Day'] == selected_day)]
    available_sessions = sorted(filtered_df['Session'].dropna().unique())
    
    options = [{'label': f'Session {s}', 'value': s} for s in available_sessions]
    default_value = available_sessions[0] if len(available_sessions) > 0 else None
    
    return options, default_value


# Callback 3: Update the timeline graph when button is clicked
@notes_app.callback(
    [Output('notes-timeline', 'figure'),
     Output('info-panel', 'children')],
    [Input('show-notes-btn', 'n_clicks')],
    [State('pid-dropdown', 'value'),
     State('day-dropdown', 'value'),
     State('session-dropdown', 'value')]
)
def update_timeline(n_clicks, selected_pid, selected_day, selected_session):
    # Create empty figure for initial state or invalid selections
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_white",
        title="Select a participant, day, and session, then click 'Show Notes'",
        title_x=0.5,
        title_xanchor='center',
        font=dict(size=14),
        title_font=dict(size=18)
    )
    
    if n_clicks == 0 or selected_pid is None or selected_day is None or selected_session is None:
        return empty_fig, ""
    
    # Filter data for the selected combination
    filtered_df = df[
        (df['Participant ID'] == selected_pid) & 
        (df['Day'] == selected_day) & 
        (df['Session'] == selected_session)
    ].copy()
    
    if filtered_df.empty:
        empty_fig.update_layout(title="No notes found for this selection")
        return empty_fig, "No notes found."
    
    # Sort by time
    filtered_df = filtered_df.sort_values('Time_Minutes')
    
    # Create y=0 for all points (flat timeline)
    filtered_df['y'] = 0
    
    # Truncate notes for hover display (full note can be very long)
    filtered_df['Note_Display'] = filtered_df['Note'].apply(
        lambda x: x if len(str(x)) <= 100 else str(x)[:100] + '...'
    )
    
    # Create the timeline figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=filtered_df['Time_Minutes'],
        y=filtered_df['y'],
        mode='markers',
        marker=dict(
            symbol='line-ns',  # Vertical line marker
            size=20,
            line=dict(width=2, color='#007bff')
        ),
        hovertemplate=(
            '<b>Time:</b> %{x:.1f} min<br>'
            '<b>Note:</b> %{customdata}<extra></extra>'
        ),
        customdata=filtered_df['Note']
    ))
    
    # Add a horizontal baseline
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
    
    # Style the figure
    title = f"Notes Timeline: PID {selected_pid} - Day {selected_day}, Session {selected_session}"
    
    fig.update_layout(
        template="plotly_white",
        title=title,
        title_x=0.5,
        title_xanchor='center',
        font=dict(size=14),
        title_font=dict(size=20),
        xaxis=dict(
            title="Time (minutes)",
            title_font=dict(size=16),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            visible=False,  # Hide y-axis since all points are at y=0
            range=[-1, 1]
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Arial"
        )
    )
    
    # Info panel content
    info_text = [
        html.P(f"Showing {len(filtered_df)} notes"),
        html.P(f"Time range: {filtered_df['Time_Minutes'].min():.1f} - {filtered_df['Time_Minutes'].max():.1f} min")
    ]
    
    return fig, info_text


if __name__ == '__main__':
    notes_app.run(debug=True, port=8051)  # Different port to run alongside main app
