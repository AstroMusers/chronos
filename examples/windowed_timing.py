"""Illustrate Chronos timing searches with explicitly synthetic windows."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u

REPOSITORY_PATH = Path(__file__).resolve().parents[1]
if str(REPOSITORY_PATH) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_PATH))

from etv_sampler import etv_noise, is_in_tess_obs, search_timing_period
from paths import get_repository_path
from tdpy.verbosity import print


def simulate_timing_example(seed=42):
    """Use a sinusoidal delay, independent Gaussian noise, and three toy windows."""
    time_days = np.arange(0.0, 1200.0, 2.5) # [day]
    period_days = 420.0 # [day]
    amplitude_seconds = 35.0 # [s]
    eclipse_duration_days = 0.15 # [day]
    timing_error_seconds = etv_noise(0.1, 1e-4, eclipse_duration_days) * u.day.to(u.s) # [s]
    truth_seconds = amplitude_seconds * np.sin(2 * np.pi * time_days / period_days) # [s]
    observed_seconds = truth_seconds + np.random.default_rng(seed).normal(
        0.0, timing_error_seconds, len(time_days),
    ) # [s]
    windows = pd.DataFrame({
        'Sector': [1, 2, 3],
        'StartD': [0.0, 500.0, 1000.0], # [day]
        'EndD': [180.0, 720.0, 1180.0], # [day]
    })
    selected = np.asarray(is_in_tess_obs(time_days, [1, 2, 3], windows))
    return {
        'time_days': time_days,
        'period_days': period_days,
        'amplitude_seconds': amplitude_seconds,
        'timing_error_seconds': timing_error_seconds,
        'truth_seconds': truth_seconds,
        'observed_seconds': observed_seconds,
        'selected': selected,
        'windows': windows,
        'complete_search': search_timing_period(time_days, observed_seconds),
        'windowed_search': search_timing_period(time_days[selected], observed_seconds[selected]),
    }


def _announce_output(path):
    try:
        name = path.resolve().relative_to(get_repository_path()).as_posix()
    except ValueError:
        name = path.name
    print(f'Writing to {name}...')


def plot_timing_example(result, output_directory=None, typefileplot='png', typeplotback='white', interactive=False):
    """Save full-baseline timing and period-search views from the same result."""
    if typefileplot not in {'png', 'pdf'} or typeplotback not in {'white', 'dark'}:
        raise ValueError('Use png/pdf output and white/dark backgrounds')
    output_directory = Path(output_directory) if output_directory is not None else get_repository_path() / 'examples' / 'visuals'
    output_directory.mkdir(parents=True, exist_ok=True)
    background, foreground = ('black', 'white') if typeplotback == 'dark' else ('white', 'black')
    timing_path = output_directory / f'synthetic_windowed_timing_{typeplotback}.{typefileplot}'
    period_path = output_directory / f'synthetic_timing_period_search_{typeplotback}.{typefileplot}'
    time_days = result['time_days']
    selected = result['selected']
    model_days = np.linspace(time_days[0], time_days[-1], 2000) # [day]
    model_seconds = result['windowed_search']['lomb_scargle'].model(
        model_days, result['windowed_search']['frequency_per_day'],
    ) # [s]
    truth_seconds = result['amplitude_seconds'] * np.sin(2 * np.pi * model_days / result['period_days']) # [s]
    style = {
        'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
        'figure.facecolor': background, 'axes.facecolor': background,
        'savefig.facecolor': background, 'text.color': foreground,
        'axes.labelcolor': foreground, 'axes.edgecolor': foreground,
        'xtick.color': foreground, 'ytick.color': foreground, 'axes.grid': False,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(figsize=(8, 4.8), layout='constrained')
        axes.scatter(time_days, result['observed_seconds'], color='#89939a', s=8, alpha=0.45, label='All synthetic observations')
        axes.errorbar(time_days[selected], result['observed_seconds'][selected],
                      yerr=result['timing_error_seconds'], fmt='.', color='#007fa3',
                      markersize=4, linewidth=0.6, label='Selected synthetic windows')
        axes.plot(model_days, truth_seconds, color=foreground, linestyle='--', label='Injected sinusoidal delay')
        axes.plot(model_days, model_seconds, color='#c73546', label='Windowed best-fit sinusoid')
        axes.set(xlabel='Time [day]', ylabel='Timing residual [s]', xlim=(time_days[0], time_days[-1]))
        axes.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2,
                    fancybox=True, framealpha=1, facecolor=background, edgecolor=foreground)
        try:
            _announce_output(timing_path)
            figure.savefig(timing_path, dpi=300, bbox_inches='tight')
        finally:
            plt.close(figure)

        figure, axes = plt.subplots(figsize=(8, 4.2), layout='constrained')
        for key, color, label in (
            ('complete_search', '#89939a', 'All synthetic observations'),
            ('windowed_search', '#c73546', 'Selected synthetic windows'),
        ):
            search = result[key]
            axes.plot(1 / search['frequencies_per_day'], search['power'], color=color, label=label)
        axes.axvline(result['period_days'], color=foreground, linestyle='--', label='Injected period')
        axes.set(xlabel='Trial period [day]', ylabel='Lomb-Scargle power', xlim=(0, time_days[-1] / 2))
        axes.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fancybox=True,
                    framealpha=1, facecolor=background, edgecolor=foreground)
        try:
            _announce_output(period_path)
            figure.savefig(period_path, dpi=300, bbox_inches='tight')
        finally:
            plt.close(figure)

    outputs = {'timing': timing_path, 'periodogram': period_path}
    if interactive:
        from bokeh.embed import file_html
        from bokeh.layouts import column
        from bokeh.models import ColumnDataSource, Whisker
        from bokeh.plotting import figure
        from bokeh.resources import INLINE

        timing = figure(height=390, sizing_mode='stretch_width', x_axis_label='Time [day]',
                        y_axis_label='Timing residual [s]', x_range=(float(time_days[0]), float(time_days[-1])))
        timing.scatter(time_days, result['observed_seconds'], size=3, color='#89939a', alpha=0.45, legend_label='All synthetic observations')
        timing.scatter(time_days[selected], result['observed_seconds'][selected], size=4,
                       color='#007fa3', legend_label='Selected synthetic windows')
        error_source = ColumnDataSource({
            'time': time_days[selected],
            'lower': result['observed_seconds'][selected] - result['timing_error_seconds'],
            'upper': result['observed_seconds'][selected] + result['timing_error_seconds'],
        })
        timing.add_layout(Whisker(source=error_source, base='time', lower='lower', upper='upper', line_color='#007fa3'))
        timing.line(model_days, truth_seconds, color=foreground, line_dash='dashed', legend_label='Injected sinusoidal delay')
        timing.line(model_days, model_seconds, color='#c73546', legend_label='Windowed best-fit sinusoid')
        periods = figure(height=390, sizing_mode='stretch_width', x_axis_label='Trial period [day]',
                         y_axis_label='Lomb-Scargle power', x_range=(0, float(time_days[-1] / 2)))
        for key, color, label in (
            ('complete_search', '#89939a', 'All synthetic observations'),
            ('windowed_search', '#c73546', 'Selected synthetic windows'),
        ):
            search = result[key]
            periods.line(1 / search['frequencies_per_day'], search['power'], color=color, legend_label=label)
        periods.line([result['period_days']] * 2, [0, 1], color=foreground,
                     line_dash='dashed', legend_label='Injected period')
        for plot in (timing, periods):
            plot.background_fill_color = plot.border_fill_color = background
            plot.outline_line_color = foreground
            plot.grid.visible = False
            plot.axis.axis_line_color = plot.axis.major_tick_line_color = foreground
            plot.axis.axis_label_text_color = plot.axis.major_label_text_color = foreground
            plot.axis.axis_label_text_font_size = plot.axis.major_label_text_font_size = '11pt'
            legend = plot.legend[0]
            legend.background_fill_color, legend.background_fill_alpha = background, 1
            legend.border_line_color, legend.border_radius = foreground, 4
            legend.label_text_color, legend.label_text_font_size = foreground, '11pt'
            plot.add_layout(legend, 'above')
        html_path = output_directory / f'synthetic_timing_search_{typeplotback}.html'
        _announce_output(html_path)
        html_path.write_text(file_html(column(timing, periods, sizing_mode='stretch_width'), INLINE,
                                      'Synthetic eclipse-timing searches'), encoding='utf-8')
        outputs['interactive'] = html_path
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--typefileplot', choices=['png', 'pdf'], default='png')
    parser.add_argument('--typeplotback', choices=['white', 'dark'], default='white')
    parser.add_argument('--bokeh', action='store_true')
    arguments = parser.parse_args()
    plot_timing_example(simulate_timing_example(), typefileplot=arguments.typefileplot,
                        typeplotback=arguments.typeplotback, interactive=arguments.bokeh)