import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

BASE_DIR = './EyeQ'

# --- Metric definitions ---
# Metrics where "Unable to perform" should be shown as failure markers
FAILURE_METRICS = {
    'acc facility OU', 'acc facility OD', 'acc facility OS',
    'mem OD', 'mem OS',
}

METRIC_OPTIONS = [
    'aca', 'cac', 'verg fac OU',
    'npc #1 blur', 'npc #1 recovery', 'npc #1 break',
    'acc facility OU', 'acc facility OD', 'acc facility OS',
    'mem OD', 'mem OS',
]


# --- DATA LOADING ---
def load_optometry_data():
    # Load optometry excel
    opto = pd.read_excel('Optometry_Cleaned_2172026.xlsx')

    # Fix known date typo and parse dates
    opto['Date'] = opto['Date'].apply(
        lambda x: pd.Timestamp('2026-01-28') if isinstance(x, str) else pd.Timestamp(x)
    )

    # Derive Study Day (1-4) from date rank per participant
    opto['Study Day'] = opto.groupby('PT')['Date'].transform(
        lambda x: x.rank(method='dense').astype(int)
    )

    # Rename to match existing data conventions
    opto = opto.rename(columns={'PT': 'Participant ID', 'Session': 'Daily Session'})

    # Load pre-generated session lookup (avoids scanning hundreds of CSVs)
    session_lookup = pd.read_csv('session_lookup.csv')

    # Merge session metadata and VIMSSQ scores
    opto = opto.merge(session_lookup, on=['Participant ID', 'Study Day', 'Daily Session'], how='left')
    opto = opto.rename(columns={'VIMSSQ-Intake': 'VIMSSQ Score'})

    # Parse metric columns: flag failures, coerce to numeric
    for col in METRIC_OPTIONS:
        raw = opto[col].astype(str).str.strip().str.lower()
        opto[col + '_failed'] = raw.isin(['unable to perform', 'utp', ''])
        opto[col + '_numeric'] = pd.to_numeric(opto[col], errors='coerce')

    opto['Participant ID'] = opto['Participant ID'].astype(str)

    return opto


# Load once at startup
df = load_optometry_data()
available_pids = sorted(df['Participant ID'].unique())

# --- DASH APP SETUP ---
optometry_app = dash.Dash(__name__)
server = optometry_app.server

# Grouping options for color/facet dropdowns
variable_options = [
    {'label': 'None', 'value': 'None'},
    {'label': 'Device Type', 'value': 'Device Type'},
    {'label': 'Session Type', 'value': 'Session Type'},
    {'label': 'Participant ID', 'value': 'Participant ID'},
    {'label': 'Moderator Initials', 'value': 'Moderator Initials'},
    {'label': 'Study Day', 'value': 'Study Day'},
    {'label': 'Daily Session', 'value': 'Daily Session'},
    {'label': 'VIMSSQ Bin', 'value': 'VIMSSQ Bin'},
    {'label': 'Optometrist', 'value': 'Optometrist'},
]

# --- LAYOUT ---
optometry_app.layout = html.Div([
    html.H1("EyeQ Optometry Pre/Post Dashboard", style={'textAlign': 'center'}),

    html.Div([
        # Left side: Graph
        html.Div([
            dcc.Graph(
                id='main-graph',
                style={'height': '85vh', 'width': '100%'},
                config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'scale': 3,
                        'filename': 'optometry_plot'
                    }
                }
            )
        ], style={
            'width': '75%',
            'display': 'inline-block',
            'verticalAlign': 'top'
        }),

        # Right side: Controls
        html.Div([
            html.H3("Controls", style={'textAlign': 'center', 'marginBottom': '20px'}),

            html.Label("1. Select Metric:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[{'label': m, 'value': m} for m in METRIC_OPTIONS],
                value='aca',
                clearable=False
            ),
            html.Br(),

            html.Label("2. Select Participants (Leave empty for ALL):"),
            dcc.Dropdown(
                id='pid-dropdown',
                options=[{'label': pid, 'value': pid} for pid in available_pids],
                value=[],
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
                id='show-paired-lines',
                options=[{'label': ' Show paired Pre\u2194Post lines', 'value': 'show'}],
                value=[]
            ),
            html.Br(),
            dcc.Checklist(
                id='show-mean-checkbox',
                options=[{'label': ' Show group mean', 'value': 'show'}],
                value=['show']
            ),
            html.Br(),
            html.Hr(),

            html.Label("3. Color By:"),
            dcc.Dropdown(
                id='color-dropdown',
                options=variable_options,
                value='None',
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


# --- Plotly default color sequence ---
COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA',
    '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
    '#FF97FF', '#FECB52',
]


def make_jitter(n, amount=0.08):
    """Small random horizontal jitter so points don't stack."""
    rng = np.random.default_rng(42)
    return rng.uniform(-amount, amount, size=n)


# --- CALLBACK ---
@optometry_app.callback(
    Output('main-graph', 'figure'),
    [Input('metric-dropdown', 'value'),
     Input('pid-dropdown', 'value'),
     Input('color-dropdown', 'value'),
     Input('facet-col-dropdown', 'value'),
     Input('facet-row-dropdown', 'value'),
     Input('vimssq-threshold', 'value'),
     Input('show-paired-lines', 'value'),
     Input('show-mean-checkbox', 'value')]
)
def update_graph(metric, selected_pids, color_var, facet_col, facet_row,
                 vimssq_threshold, show_paired, show_mean):

    filtered = df.copy()

    # VIMSSQ binning
    if vimssq_threshold is not None:
        filtered['VIMSSQ Bin'] = filtered['VIMSSQ Score'].apply(
            lambda x: 'High' if pd.notna(x) and x >= vimssq_threshold
            else ('Low' if pd.notna(x) else None)
        )

    # Filter by selected PIDs
    if selected_pids:
        filtered = filtered[filtered['Participant ID'].isin(selected_pids)]

    val_col = metric + '_numeric'
    fail_col = metric + '_failed'
    has_failures = metric in FAILURE_METRICS

    # Map Pre/Post to numeric x positions
    pre_post_x = {'Pre': 0, 'Post': 1}
    filtered['x_base'] = filtered['Pre/Post'].map(pre_post_x)

    # Determine facet groups
    fc = facet_col if facet_col != 'None' else None
    fr = facet_row if facet_row != 'None' else None
    c = color_var if color_var != 'None' else None

    # Build facet grid
    facet_col_vals = sorted(filtered[fc].dropna().unique()) if fc else [None]
    facet_row_vals = sorted(filtered[fr].dropna().unique()) if fr else [None]
    n_cols = len(facet_col_vals)
    n_rows = len(facet_row_vals)

    from plotly.subplots import make_subplots
    subplot_titles = []
    for rv in facet_row_vals:
        for cv in facet_col_vals:
            parts = []
            if rv is not None:
                parts.append(str(rv))
            if cv is not None:
                parts.append(str(cv))
            subplot_titles.append(' | '.join(parts) if parts else '')

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles if any(subplot_titles) else None,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    color_vals = sorted(filtered[c].dropna().unique()) if c else [None]
    legend_added = set()

    for ri, rv in enumerate(facet_row_vals):
        for ci, cv in enumerate(facet_col_vals):
            # Filter to this facet cell
            cell = filtered.copy()
            if fc and cv is not None:
                cell = cell[cell[fc] == cv]
            if fr and rv is not None:
                cell = cell[cell[fr] == rv]

            for ki, color_val in enumerate(color_vals):
                subset = cell.copy()
                if c and color_val is not None:
                    subset = subset[subset[c] == color_val]

                color_hex = COLORS[ki % len(COLORS)]
                legend_name = str(color_val) if color_val is not None else 'All'
                show_legend = legend_name not in legend_added

                # Split into success and failure
                success = subset[subset[val_col].notna() & ~subset[fail_col]] if has_failures else subset[subset[val_col].notna()]
                failures = subset[subset[fail_col]] if has_failures else pd.DataFrame()

                # --- Success points ---
                if len(success) > 0:
                    jitter = make_jitter(len(success))
                    fig.add_trace(go.Scatter(
                        x=success['x_base'] + jitter,
                        y=success[val_col],
                        mode='markers',
                        marker=dict(color=color_hex, size=6, opacity=0.5),
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=show_legend,
                        hovertemplate=(
                            'PT %{customdata[0]}<br>Day %{customdata[1]} Sess %{customdata[2]}'
                            '<br>%{customdata[3]}: %{y:.2f}<extra></extra>'
                        ),
                        customdata=success[['Participant ID', 'Study Day', 'Daily Session', 'Pre/Post']].values,
                    ), row=ri + 1, col=ci + 1)
                    if show_legend:
                        legend_added.add(legend_name)

                # --- Failure markers ---
                if len(failures) > 0:
                    jitter_f = make_jitter(len(failures))
                    fail_legend = legend_name + ' (failed)'
                    show_fail_legend = fail_legend not in legend_added
                    fig.add_trace(go.Scatter(
                        x=failures['x_base'] + jitter_f,
                        y=[0] * len(failures),
                        mode='markers',
                        marker=dict(
                            color='red', size=8, opacity=0.6,
                            symbol='x',
                        ),
                        name=fail_legend,
                        legendgroup=fail_legend,
                        showlegend=show_fail_legend,
                        hovertemplate=(
                            'PT %{customdata[0]}<br>Day %{customdata[1]} Sess %{customdata[2]}'
                            '<br>%{customdata[3]}: Unable to perform<extra></extra>'
                        ),
                        customdata=failures[['Participant ID', 'Study Day', 'Daily Session', 'Pre/Post']].values,
                    ), row=ri + 1, col=ci + 1)
                    if show_fail_legend:
                        legend_added.add(fail_legend)

                # --- Group mean lines ---
                if 'show' in show_mean and len(success) > 0:
                    for xval, label in [(0, 'Pre'), (1, 'Post')]:
                        grp = success[success['x_base'] == xval]
                        if len(grp) > 0:
                            mean_val = grp[val_col].mean()
                            fig.add_trace(go.Scatter(
                                x=[xval - 0.15, xval + 0.15],
                                y=[mean_val, mean_val],
                                mode='lines',
                                line=dict(color=color_hex, width=3),
                                showlegend=False,
                                legendgroup=legend_name,
                                hovertemplate=f'{legend_name} {label} mean: {mean_val:.2f}<extra></extra>',
                            ), row=ri + 1, col=ci + 1)

                # --- Paired connecting lines ---
                if 'show' in show_paired and len(success) > 0:
                    # Pair by Participant ID + Study Day + Daily Session
                    pre_pts = success[success['Pre/Post'] == 'Pre'].set_index(
                        ['Participant ID', 'Study Day', 'Daily Session']
                    )[[val_col, 'x_base']]
                    post_pts = success[success['Pre/Post'] == 'Post'].set_index(
                        ['Participant ID', 'Study Day', 'Daily Session']
                    )[[val_col, 'x_base']]

                    paired = pre_pts.join(post_pts, lsuffix='_pre', rsuffix='_post', how='inner')

                    if len(paired) > 0:
                        # Draw each pair as a separate line using None separator
                        xs = []
                        ys = []
                        for _, row in paired.iterrows():
                            xs.extend([0, 1, None])
                            ys.extend([row[val_col + '_pre'], row[val_col + '_post'], None])

                        fig.add_trace(go.Scatter(
                            x=xs, y=ys,
                            mode='lines',
                            line=dict(color=color_hex, width=0.8),
                            opacity=0.3,
                            showlegend=False,
                            legendgroup=legend_name,
                            hoverinfo='skip',
                        ), row=ri + 1, col=ci + 1)

    # --- Axis and layout styling ---
    title = f"{metric} — Pre vs Post"
    if selected_pids:
        title += f" (n={len(selected_pids)} participants)"
    else:
        title += f" (n={filtered['Participant ID'].nunique()} participants)"

    fig.update_layout(
        template='plotly_white',
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20)),
        font=dict(size=14),
        legend=dict(
            yanchor='middle', y=0.5,
            xanchor='left', x=1.02,
            title_font=dict(size=16),
            font=dict(size=14),
        ),
        height=max(500, 400 * n_rows),
    )

    # Set x-axis tick labels on all subplots
    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=['Pre', 'Post'],
        range=[-0.4, 1.4],
        title_font=dict(size=16),
        tickfont=dict(size=12),
    )
    fig.update_yaxes(
        title_text=metric,
        title_font=dict(size=16),
        tickfont=dict(size=12),
    )

    return fig


if __name__ == '__main__':
    optometry_app.run(debug=True, use_reloader=False, port=8053)
