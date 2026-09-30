"""Compare backend runs with tolerances and verify source provenance."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir
    names = ('cpu_f32', 'cuda_f32', 'cpu_f64')
    reports = {name: json.loads((root / f'measure_{name}' / 'measurement.json').read_text()) for name in names}
    base = reports['cpu_f32']
    baseline = np.load(root / 'measure_cpu_f32' / 'final_state.npz')
    # Absolute tolerances in simulation units; velocity amplifies position rounding by 1/h.
    limits = {'positions': 1e-5, 'velocity': 5e-4, 'fracture_strain': 1e-5, 'fracture_time': 1e-10}
    result = {'tolerances': limits, 'pairs': {}, 'source_hashes_match': True}
    for path, digest in base['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'source changed: {path}'
    for name in names:
        assert reports[name]['source_sha256'] == base['source_sha256'], 'mixed source revisions'
        assert reports[name]['topology']['diagnostics_report']['passed']
        assert reports[name]['integrity']['finite_all_frames']
    for name in names[1:]:
        other = reports[name]
        data = np.load(root / f'measure_{name}' / 'final_state.npz')
        values = {}
        for field in ('positions', 'velocity'):
            assert data[field].shape == baseline[field].shape
            error = float(np.max(np.abs(data[field] - baseline[field])))
            values[field + '_max_abs_error'] = error
            assert np.allclose(data[field], baseline[field], atol=limits[field], rtol=0)
        assert np.array_equal(data['triangles'], baseline['triangles'])
        a, b = base['fracture']['log'], other['fracture']['log']
        assert len(a) == len(b) > 0
        assert [event['edge'] for event in a] == [event['edge'] for event in b]
        for key, limit_name in (('strain', 'fracture_strain'), ('time', 'fracture_time')):
            error = max(abs(x[key] - y[key]) for x, y in zip(a, b))
            values[key + '_max_abs_error'] = error
            assert error <= limits[limit_name]
        values['same_triangle_indices_and_tear_edges'] = True
        values['passed'] = True
        result['pairs']['cpu_f32_vs_' + name] = values
    result['passed'] = True
    (root / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
