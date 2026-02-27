import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from plotly.subplots import make_subplots

BASE_DIR = './EyeQ'

# --- Metric definitions ---
# Metrics where "Unable to perform" should be shown as failure markers
FAILURE_METRICS = {
    'acc facility OU', 'acc facility OD', 'acc facility OS',
    'mem OD', 'mem OS',
}

# Columns that need VA parsing (x/y format -> extract y)
VA_METRICS = [
    'VA Distance 10 feet OU',
    'VA Distance 10 feet OD',
    'VA Distance 10 feet OS',
]

# Maps raw column names -> clean display labels for the dropdown
METRIC_LABELS = {
    'aca': 'AC/A Ratio',
    'cac': 'CA/C Ratio',
    'verg fac OU': 'Vergence Facility — Both Eyes',
    'npc #1 blur': 'Near Point of Convergence — Blur',
    'npc #1 recovery': 'Near Point of Convergence — Recovery',
    'npc #1 break': 'Near Point of Convergence — Break',
    'acc facility OU': 'Accommodative Facility — Both Eyes',
    'acc facility OD': 'Accommodative Facility — Right Eye',
    'acc facility OS': 'Accommodative Facility — Left Eye',
    'mem OD': 'MEM Retinoscopy — Right Eye',
    'mem OS': 'MEM Retinoscopy — Left Eye',
    'VA Distance 10 feet OU': 'Visual Acuity (Distance) — Both Eyes',
    'VA Distance 10 feet OD': 'Visual Acuity (Distance) — Right Eye',
    'VA Distance 10 feet OS': 'Visual Acuity (Distance) — Left Eye',
}

METRIC_OPTIONS = list(METRIC_LABELS.keys())

PAIR_KEYS = ['Participant ID', 'Study Day', 'Daily Session']


# --- DATA LOADING ---
def load_optometry_data():
    opto = pd.read_csv('Optometry_Cleaned.csv')

    # Fix known date typo then parse dates
    opto['Date'] = opto['Date'].replace('1/28:2026', '2026-01-28')
    opto['Date'] = pd.to_datetime(opto['Date'], format='mixed')

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

    # Parse VA columns: extract denominator from "20/Y+Z" or "20/Y-Z" format
    for col in VA_METRICS:
        opto[col] = opto[col].astype(str).str.extract(r'\d+/(\d+)', expand=False)

    # Parse metric columns: flag failures, coerce to numeric
    for col in METRIC_OPTIONS:
        raw = opto[col].astype(str).str.strip().str.lower()
        opto[col + '_failed'] = raw.isin(['unable to perform', 'utp', ''])
        opto[col + '_numeric'] = pd.to_numeric(opto[col], errors='coerce')

    # Age bin: Young (20-35) vs Older (50+)
    opto['Age Bin'] = opto['Age'].apply(
        lambda x: 'Young (20-35)' if pd.notna(x) and x <= 35
        else ('Older (50+)' if pd.notna(x) and x >= 50 else None)
    )

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
    {'label': 'Age Bin', 'value': 'Age Bin'},
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

            html.Label("1. Plot Mode:"),
            dcc.Dropdown(
                id='plot-mode-dropdown',
                options=[
                    {'label': 'Delta Distribution (Post \u2212 Pre)', 'value': 'delta'},
                    {'label': 'Point Cloud (Pre vs Post)', 'value': 'pointcloud'},
                ],
                value='delta',
                clearable=False
            ),
            html.Br(),

            html.Label("2. Select Metric:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[{'label': METRIC_LABELS[m], 'value': m} for m in METRIC_OPTIONS],
                value='aca',
                clearable=False
            ),
            html.Br(),

            html.Label("3. Select Participants (Leave empty for ALL):"),
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

            html.Label("4. Color By:"),
            dcc.Dropdown(
                id='color-dropdown',
                options=variable_options,
                value='None',
                clearable=False
            ),
            html.Br(),

            html.Label("5. Facet Columns By:"),
            dcc.Dropdown(
                id='facet-col-dropdown',
                options=variable_options,
                value='None',
                clearable=False
            ),
            html.Br(),

            html.Label("6. Facet Rows By:"),
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


def compute_deltas(filtered, val_col, fail_col, has_failures):
    """Compute Post - Pre deltas for paired observations."""
    pre = filtered[filtered['Pre/Post'] == 'Pre'].set_index(PAIR_KEYS)
    post = filtered[filtered['Pre/Post'] == 'Post'].set_index(PAIR_KEYS)

    # Keep metadata columns from the pre rows (they're the same for pre/post)
    meta_cols = [c for c in filtered.columns if c not in
                 [val_col, fail_col, 'Pre/Post', 'x_base', 'Date', 'Time',
                  'Optometrist'] + PAIR_KEYS
                 and c in pre.columns]

    pre_vals = pre[[val_col, fail_col] + meta_cols]
    post_vals = post[[val_col, fail_col]]

    paired = pre_vals.join(post_vals, lsuffix='_pre', rsuffix='_post', how='inner')

    if has_failures:
        # If either Pre or Post was a failure, mark the delta as failed
        paired['delta_failed'] = paired[fail_col + '_pre'] | paired[fail_col + '_post']
        # Only compute delta where both are successful
        mask = ~paired['delta_failed']
        paired['delta'] = np.nan
        paired.loc[mask, 'delta'] = (
            paired.loc[mask, val_col + '_post'] - paired.loc[mask, val_col + '_pre']
        )
    else:
        paired['delta_failed'] = False
        paired['delta'] = paired[val_col + '_post'] - paired[val_col + '_pre']

    return paired.reset_index()


# --- CALLBACK ---
@optometry_app.callback(
    Output('main-graph', 'figure'),
    [Input('plot-mode-dropdown', 'value'),
     Input('metric-dropdown', 'value'),
     Input('pid-dropdown', 'value'),
     Input('color-dropdown', 'value'),
     Input('facet-col-dropdown', 'value'),
     Input('facet-row-dropdown', 'value'),
     Input('vimssq-threshold', 'value'),
     Input('show-paired-lines', 'value'),
     Input('show-mean-checkbox', 'value')]
)
def update_graph(plot_mode, metric, selected_pids, color_var, facet_col, facet_row,
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

    if plot_mode == 'delta':
        return build_delta_figure(filtered, metric, val_col, fail_col, has_failures,
                                  color_var, facet_col, facet_row, selected_pids, show_mean)
    else:
        return build_pointcloud_figure(filtered, metric, val_col, fail_col, has_failures,
                                       color_var, facet_col, facet_row, selected_pids,
                                       show_paired, show_mean)


def gaussian_kde(data, n_points=200, bw_factor=1.0):
    """Compute a simple Gaussian KDE without scipy dependency."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return np.array([]), np.array([])

    # Silverman's rule of thumb for bandwidth
    std = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    bw = 0.9 * min(std, iqr / 1.34) * len(data) ** (-0.2) * bw_factor
    if bw <= 0:
        bw = std * 0.5 if std > 0 else 1.0

    x_min = data.min() - 3 * bw
    x_max = data.max() + 3 * bw
    x_grid = np.linspace(x_min, x_max, n_points)

    # Vectorized KDE
    density = np.zeros(n_points)
    for d in data:
        density += np.exp(-0.5 * ((x_grid - d) / bw) ** 2)
    density /= (len(data) * bw * np.sqrt(2 * np.pi))

    return x_grid, density


def build_delta_figure(filtered, metric, val_col, fail_col, has_failures,
                       color_var, facet_col, facet_row, selected_pids, show_mean):
    """Delta distribution: KDE curves of (Post - Pre) with zero reference line."""

    deltas = compute_deltas(filtered, val_col, fail_col, has_failures)

    fc = facet_col if facet_col != 'None' else None
    fr = facet_row if facet_row != 'None' else None
    c = color_var if color_var != 'None' else None

    facet_col_vals = sorted(deltas[fc].dropna().unique()) if fc else [None]
    facet_row_vals = sorted(deltas[fr].dropna().unique()) if fr else [None]
    n_cols = len(facet_col_vals)
    n_rows = len(facet_row_vals)

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
        shared_xaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    color_vals = sorted(deltas[c].dropna().unique()) if c else [None]
    legend_added = set()

    for ri, rv in enumerate(facet_row_vals):
        for ci, cv in enumerate(facet_col_vals):
            cell = deltas.copy()
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

                success = subset[subset['delta'].notna() & ~subset['delta_failed']]
                failures = subset[subset['delta_failed']] if has_failures else pd.DataFrame()

                # KDE curve
                if len(success) >= 2:
                    x_grid, density = gaussian_kde(success['delta'].values)
                    fig.add_trace(go.Scatter(
                        x=x_grid, y=density,
                        mode='lines',
                        line=dict(color=color_hex, width=2.5),
                        fill='tozeroy',
                        fillcolor=f'rgba({int(color_hex[1:3], 16)}, {int(color_hex[3:5], 16)}, {int(color_hex[5:7], 16)}, 0.15)',
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=show_legend,
                        hoverinfo='skip',
                    ), row=ri + 1, col=ci + 1)
                    if show_legend:
                        legend_added.add(legend_name)

                # Rug plot: small tick marks along y=0 for individual observations
                if len(success) > 0:
                    fig.add_trace(go.Scatter(
                        x=success['delta'],
                        y=[0] * len(success),
                        mode='markers',
                        marker=dict(color=color_hex, size=6, symbol='line-ns',
                                    line=dict(width=1.5, color=color_hex), opacity=0.5),
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=False,
                        hovertemplate=(
                            'PT %{customdata[0]}<br>Day %{customdata[1]} Sess %{customdata[2]}'
                            '<br>\u0394 = %{x:.2f}<extra></extra>'
                        ),
                        customdata=success[['Participant ID', 'Study Day', 'Daily Session']].values,
                    ), row=ri + 1, col=ci + 1)

                # Failure markers on the rug
                if len(failures) > 0:
                    fail_legend = legend_name + ' (failed)'
                    show_fail_legend = fail_legend not in legend_added
                    fig.add_trace(go.Scatter(
                        x=[0] * len(failures),
                        y=[0] * len(failures),
                        mode='markers',
                        marker=dict(color='red', size=8, opacity=0.6, symbol='x'),
                        name=fail_legend,
                        legendgroup=fail_legend,
                        showlegend=show_fail_legend,
                        hovertemplate=(
                            'PT %{customdata[0]}<br>Day %{customdata[1]} Sess %{customdata[2]}'
                            '<br>Pre or Post: Unable to perform<extra></extra>'
                        ),
                        customdata=failures[['Participant ID', 'Study Day', 'Daily Session']].values,
                    ), row=ri + 1, col=ci + 1)
                    if show_fail_legend:
                        legend_added.add(fail_legend)

                # Mean vertical line
                if 'show' in show_mean and len(success) > 0:
                    mean_val = success['delta'].mean()
                    fig.add_vline(
                        x=mean_val,
                        line=dict(color=color_hex, width=2, dash='dot'),
                        annotation_text=f'mean={mean_val:.2f} (n={len(success)})',
                        annotation_font=dict(size=11, color=color_hex),
                        annotation_position='top right',
                        row=ri + 1, col=ci + 1,
                    )

            # Zero reference line
            fig.add_vline(
                x=0,
                line=dict(color='gray', width=1, dash='dash'),
                row=ri + 1, col=ci + 1,
            )

    # Title
    n_pairs = len(deltas[deltas['delta'].notna()])
    title = f"{METRIC_LABELS.get(metric, metric)} \u2014 Post \u2212 Pre (\u0394)"
    if selected_pids:
        title += f" (n={len(selected_pids)} participants, {n_pairs} pairs)"
    else:
        title += f" (n={deltas['Participant ID'].nunique()} participants, {n_pairs} pairs)"

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

    fig.update_xaxes(
        title_text=f'\u0394 {METRIC_LABELS.get(metric, metric)} (Post \u2212 Pre)',
        title_font=dict(size=16),
        tickfont=dict(size=12),
    )
    fig.update_yaxes(
        title_text='Density',
        title_font=dict(size=16),
        tickfont=dict(size=12),
    )

    return fig


def build_pointcloud_figure(filtered, metric, val_col, fail_col, has_failures,
                            color_var, facet_col, facet_row, selected_pids,
                            show_paired, show_mean):
    """Original point cloud: Pre and Post as separate x positions."""

    pre_post_x = {'Pre': 0, 'Post': 1}
    filtered['x_base'] = filtered['Pre/Post'].map(pre_post_x)

    fc = facet_col if facet_col != 'None' else None
    fr = facet_row if facet_row != 'None' else None
    c = color_var if color_var != 'None' else None

    facet_col_vals = sorted(filtered[fc].dropna().unique()) if fc else [None]
    facet_row_vals = sorted(filtered[fr].dropna().unique()) if fr else [None]
    n_cols = len(facet_col_vals)
    n_rows = len(facet_row_vals)

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

                success = subset[subset[val_col].notna() & ~subset[fail_col]] if has_failures else subset[subset[val_col].notna()]
                failures = subset[subset[fail_col]] if has_failures else pd.DataFrame()

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

                if len(failures) > 0:
                    jitter_f = make_jitter(len(failures))
                    fail_legend = legend_name + ' (failed)'
                    show_fail_legend = fail_legend not in legend_added
                    fig.add_trace(go.Scatter(
                        x=failures['x_base'] + jitter_f,
                        y=[0] * len(failures),
                        mode='markers',
                        marker=dict(color='red', size=8, opacity=0.6, symbol='x'),
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

                if 'show' in show_paired and len(success) > 0:
                    pre_pts = success[success['Pre/Post'] == 'Pre'].set_index(PAIR_KEYS)[[val_col, 'x_base']]
                    post_pts = success[success['Pre/Post'] == 'Post'].set_index(PAIR_KEYS)[[val_col, 'x_base']]
                    paired = pre_pts.join(post_pts, lsuffix='_pre', rsuffix='_post', how='inner')

                    if len(paired) > 0:
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

    title = f"{METRIC_LABELS.get(metric, metric)} \u2014 Pre vs Post"
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

    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=['Pre', 'Post'],
        range=[-0.4, 1.4],
        title_font=dict(size=16),
        tickfont=dict(size=12),
    )
    fig.update_yaxes(
        title_text=METRIC_LABELS.get(metric, metric),
        title_font=dict(size=16),
        tickfont=dict(size=12),
    )

    return fig


if __name__ == '__main__':
    optometry_app.run(debug=True, use_reloader=False, port=8053)
