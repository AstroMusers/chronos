import builtins
import ast
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.timeseries import LombScargle
from tdpy.verbosity import print

import etv_sampler


def test_import_has_no_sampler_or_file_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHRONOS_PATH", str(tmp_path))
    monkeypatch.delenv("TDPY_VERBOSITY", raising=False)
    original_import = builtins.__import__
    optional_modules = {"rebound_model", "lite_delay", "tess_stars2px", "ticgen"}

    def block_optional_imports(name, *args, **kwargs):
        if name.split(".")[0] in optional_modules:
            raise AssertionError("Import must not require the full orbit workflow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_optional_imports)
    source_path = Path(__file__).with_name("etv_sampler.py")
    specification = importlib.util.spec_from_file_location("chronos_import_check", source_path)
    sampler = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(sampler)

    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().out == ""
    assert sampler.tess_obs_range is None
    np.testing.assert_allclose(sampler.etv_noise(0.1, 1e-4, 0.15), 7.5e-5)


def test_search_matches_original_recovery_periodogram():
    time_days = np.arange(101.0) # [day]
    delay_seconds = 20 * np.sin(2 * np.pi * time_days / 10) # [s]
    search = etv_sampler.search_timing_period(time_days, delay_seconds)
    original = LombScargle(time_days, delay_seconds, normalization='standard')
    frequencies, power = original.autopower(
        samples_per_peak=200,
        maximum_frequency=1 / (np.min(np.diff(time_days)) * 2),
        minimum_frequency=1 / ((np.max(time_days) - np.min(time_days)) / 2),
    )

    np.testing.assert_array_equal(search['frequencies_per_day'], frequencies)
    np.testing.assert_array_equal(search['power'], power)
    assert search['period_days'] == pytest.approx(10, abs=0.02)
    np.testing.assert_allclose(
        search['lomb_scargle'].model(time_days, search['frequency_per_day']),
        delay_seconds, atol=0.1,
    )


def test_search_converts_time_and_delay_quantities():
    time_days = np.arange(101.0) # [day]
    delay_seconds = np.sin(2 * np.pi * time_days / 10) # [s]
    numeric = etv_sampler.search_timing_period(time_days, delay_seconds)
    quantities = etv_sampler.search_timing_period(
        (time_days * u.day).to(u.hour), (delay_seconds * u.s).to(u.minute),
    )
    np.testing.assert_allclose(quantities['power'], numeric['power'], atol=1e-13)
    assert quantities['period_days'] == numeric['period_days']


@pytest.mark.parametrize('times,delays', [
    ([0, 1], [0, 1]),
    ([0, 1, 2], [0, 1]),
    ([[0, 1], [2, 3]], [[0, 1], [1, 0]]),
    ([0, 1, np.nan], [0, 1, 0]),
    ([0, 1, 2], [0, np.inf, 0]),
    ([0, 2, 1], [0, 1, 0]),
    ([0, 0, 2], [0, 1, 0]),
    ([0, 1, 2], [0, 1, 0]),
    (np.arange(10), np.zeros(10)),
    (np.ma.array([0, 1, 2], mask=[False, True, False]), [0, 1, 0]),
    ([0, 1, 2], np.ma.array([0, 1, 0], mask=[False, True, False])),
])
def test_search_rejects_invalid_timing_observations(times, delays):
    with pytest.raises(ValueError):
        etv_sampler.search_timing_period(times, delays)


def test_observation_windows_use_selected_sectors_and_include_boundaries():
    windows = pd.DataFrame({'Sector': [1, 2], 'StartD': [0, 10], 'EndD': [2, 12]})
    mask = etv_sampler.is_in_tess_obs([-1, 0, 1, 2, 10, 12, 13], [1], windows)
    assert mask == [False, True, True, True, False, False, False]


def test_observation_windows_must_be_supplied_outside_recovery(monkeypatch):
    monkeypatch.setattr(etv_sampler, 'tess_obs_range', None)
    with pytest.raises(ValueError, match='observation_windows'):
        etv_sampler.is_in_tess_obs([1], [1])


def test_recovery_startup_schedules_all_requested_draws(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(etv_sampler, 'DATA_PATH', tmp_path)
    monkeypatch.setattr(etv_sampler, 'etv_sample_path', tmp_path / 'etv_sample.csv')
    monkeypatch.setattr(etv_sampler, 'total_number_of_samples', 100)
    monkeypatch.setattr(etv_sampler, 'num_cpu', 38)
    original_import = etv_sampler.importlib.import_module

    def optional_import(name, *args, **kwargs):
        if name in {'rebound_model', 'lite_delay', 'tess_stars2px', 'ticgen'}:
            return None
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(etv_sampler.importlib, 'import_module', optional_import)
    tess_path = tmp_path / 'TESS_orbit_times.csv'
    print(f'Writing to {tess_path}...')
    pd.DataFrame({'Sector': [1], 'StartD': [0], 'EndD': [10]}).to_csv(tess_path, index=False)
    scheduled = []

    class RecordingPool:
        def __init__(self, workers):
            assert workers == 38

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, batches):
            assert function is etv_sampler.etv_recovery
            scheduled.extend(int(batch) for batch in batches)

    monkeypatch.setattr(etv_sampler, 'Pool', RecordingPool)
    monkeypatch.setenv('TDPY_VERBOSITY', '1')
    capsys.readouterr()
    etv_sampler.main()
    output = capsys.readouterr().out.splitlines()
    assert output[:2] == [
        f'Reading from {tess_path}...',
        f'Writing to {etv_sampler.etv_sample_path}...',
    ]
    assert len(scheduled) == 38
    assert sum(scheduled) == 100
    assert max(scheduled) - min(scheduled) == 1
    print(f'Reading from {etv_sampler.etv_sample_path}...')
    assert list(pd.read_csv(etv_sampler.etv_sample_path).columns)[-1] == 'amplitude_lombscargle'


def test_missing_orbit_dependencies_preserve_existing_catalog(tmp_path, monkeypatch):
    catalog = tmp_path / 'etv_sample.csv'
    catalog.write_text('existing catalog\n', encoding='utf-8')
    monkeypatch.setattr(etv_sampler, 'etv_sample_path', catalog)

    def missing_dependency(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(etv_sampler.importlib, 'import_module', missing_dependency)
    with pytest.raises(ModuleNotFoundError):
        etv_sampler.main()
    assert catalog.read_text(encoding='utf-8') == 'existing catalog\n'


def _load_timing_example():
    source_path = Path(__file__).parent / 'examples' / 'windowed_timing.py'
    specification = importlib.util.spec_from_file_location('chronos_timing_example', source_path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    return example


def test_synthetic_example_uses_production_search_and_windows():
    result = _load_timing_example().simulate_timing_example()
    np.testing.assert_array_equal(
        result['selected'],
        etv_sampler.is_in_tess_obs(result['time_days'], [1, 2, 3], result['windows']),
    )
    assert 0 < np.count_nonzero(result['selected']) < len(result['selected'])
    assert result['timing_error_seconds'] == pytest.approx(6.48)
    for key in ('complete_search', 'windowed_search'):
        assert result[key]['period_days'] == pytest.approx(result['period_days'], rel=0.05)


@pytest.mark.parametrize('theme', ['white', 'dark'])
def test_example_figures_preserve_full_baseline_and_plot_style(tmp_path, monkeypatch, theme):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    example = _load_timing_example()
    result = example.simulate_timing_example()
    original_savefig = Figure.savefig
    saved_axes = []

    def inspect_figure(figure, *args, **kwargs):
        FigureCanvasAgg(figure)
        figure.canvas.draw()
        axes = figure.axes[0]
        saved_axes.append(axes)
        assert not any(line.get_visible() for line in axes.get_xgridlines() + axes.get_ygridlines())
        legend = axes.get_legend()
        assert legend.get_frame().get_alpha() == 1
        assert not axes.get_window_extent().overlaps(legend.get_window_extent())
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', inspect_figure)
    outputs = example.plot_timing_example(result, tmp_path, typeplotback=theme)
    np.testing.assert_allclose(saved_axes[0].get_xlim(), [result['time_days'][0], result['time_days'][-1]])
    assert all(path.parent == tmp_path for path in outputs.values())
    assert all(path.stat().st_size > 1000 for path in outputs.values())


@pytest.mark.parametrize('theme', ['white', 'dark'])
def test_optional_bokeh_example_uses_same_data_without_local_paths(tmp_path, theme):
    pytest.importorskip('bokeh')
    example = _load_timing_example()
    result = example.simulate_timing_example()
    outputs = example.plot_timing_example(result, tmp_path, typeplotback=theme, interactive=True)
    html = outputs['interactive'].read_text(encoding='utf-8')
    assert 'Selected synthetic windows' in html
    assert str(tmp_path) not in html
    assert str(Path.home()) not in html


def test_saved_notebook_executes_pipeline_and_embeds_safe_accessible_figures():
    notebook_path = Path(__file__).parent / 'examples' / 'WindowedTiming.ipynb'
    serialized = notebook_path.read_text(encoding='utf-8')
    notebook = json.loads(serialized)
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert code_cells and all(cell['execution_count'] is not None for cell in code_cells)
    source = '\n'.join(''.join(cell['source']) for cell in code_cells)
    syntax = ast.parse(source)
    calls = [node for node in ast.walk(syntax) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert {'simulate_timing_example', 'plot_timing_example'}.issubset({call.func.id for call in calls})
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(syntax))
    image_calls = [call for call in calls if call.func.id == 'Image']
    assert len(image_calls) == 2
    assert all(any(keyword.arg == 'alt' and isinstance(keyword.value, ast.Constant)
                   and keyword.value.value for keyword in call.keywords) for call in image_calls)
    outputs = [output for cell in code_cells for output in cell['outputs']]
    assert not any(output['output_type'] == 'error' for output in outputs)
    images = [output['data']['image/png'] for output in outputs if 'image/png' in output.get('data', {})]
    assert len(images) == 2
    assert all(base64.b64decode(''.join(image)).startswith(b'\x89PNG\r\n\x1a\n') for image in images)
    assert '/Users/' not in serialized and '/private/var/folders/' not in serialized