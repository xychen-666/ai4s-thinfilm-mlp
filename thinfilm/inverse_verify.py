"""Recompute inverse model predictions, TMM spectra and every reported metric."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import make_dataset,make_design_candidates
from .inverse import InverseMLP,predict_inverse,evaluate_inverse
from .physics import tmm_reflectance,tmm_rt


def verify(output: Path) -> dict:
    config=Config(); torch.set_num_threads(1)
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    with np.load(output/'data'/'dataset.npz') as z:data={k:z[k] for k in z.files}
    for key,array in make_dataset(config).__dict__.items():np.testing.assert_array_equal(data[key],array)
    with np.load(output/'data'/'test_results.npz') as z:arrays={k:z[k] for k in z.files}
    np.testing.assert_array_equal(arrays['target_spectra'],data['y_test'])
    np.testing.assert_array_equal(arrays['true_thickness'],data['x_test'])
    np.testing.assert_array_equal(arrays['supervised_thickness'],np.clip(arrays['supervised_raw_thickness'],40,180))
    for method in ('supervised','physics'):
        model=InverseMLP(bounded=method=='physics')
        model.load_state_dict(torch.load(output/'models'/f'{method}_4000.pt',map_location='cpu',weights_only=True))
        predicted=predict_inverse(model,data['y_test'],config)
        key='supervised_raw_thickness' if method=='supervised' else 'physics_thickness'
        np.testing.assert_array_equal(predicted,arrays[key])
    for method in ('supervised','physics','refined','nearest'):
        d=arrays[f'{method}_thickness']; assert np.all((d>=40)&(d<=180))
        metrics,spectra=evaluate_inverse(data['y_test'],data['x_test'],d,config)
        np.testing.assert_allclose(spectra,arrays[f'{method}_spectra'],atol=1e-13,rtol=0)
        for key,value in metrics.items():np.testing.assert_allclose(value,summary['methods'][method][key],atol=1e-12,rtol=0)
        r,t=tmm_rt(d,config.wavelengths_nm);np.testing.assert_allclose(r+t,1,atol=2e-14,rtol=0)
    assert np.all(np.diff(arrays['refinement_best_mse_history'],axis=0)<=1e-14)
    true_mse=np.mean((arrays['refined_spectra']-data['y_test'])**2,axis=1)
    np.testing.assert_allclose(arrays['refinement_best_mse_history'][-1],true_mse,atol=1e-13,rtol=0)
    # The lookup baseline's reference library is the training split only.
    nearest=arrays['nearest_training_indices'];assert np.all((nearest>=0)&(nearest<4000))
    np.testing.assert_array_equal(arrays['nearest_thickness'],data['x_train'][nearest])
    for row in summary['sizes_results']:
        size=int(row['training_samples']);model=InverseMLP()
        model.load_state_dict(torch.load(output/'models'/f'supervised_{size}.pt',map_location='cpu',weights_only=True))
        d=predict_inverse(model,data['y_test'],config)
        metrics,_=evaluate_inverse(data['y_test'],data['x_test'],np.clip(d,40,180),config)
        for key,value in metrics.items():np.testing.assert_allclose(value,row[key],atol=1e-12,rtol=0)
        history=json.loads((output/'histories'/f'supervised_{size}.json').read_text())
        assert int(np.argmin(history['validation_loss']))+1==row['best_epoch']
    ph=json.loads((output/'histories'/'physics_4000.json').read_text())
    assert int(np.argmin(ph['validation_loss']))+1==summary['best_epoch_physics']
    with np.load(output/'data'/'independent_targets.npz') as z:pool={k:z[k] for k in z.files}
    candidates=make_design_candidates(config);np.testing.assert_array_equal(pool['true_thickness'],candidates)
    np.testing.assert_allclose(pool['target_spectra'],tmm_reflectance(candidates,config.wavelengths_nm),atol=1e-13,rtol=0)
    for method in ('supervised','physics'):
        metrics,r=evaluate_inverse(pool['target_spectra'],candidates,pool[f'{method}_thickness'],config)
        np.testing.assert_allclose(r,pool[f'{method}_spectra'],atol=1e-13,rtol=0)
        for key,value in metrics.items():np.testing.assert_allclose(value,summary['independent_10000'][method][key],atol=1e-12,rtol=0)
    with (output/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==4*config.n_test
    for row in rows:
        method=row['method'];idx=int(row['test_index'])
        mae=float(np.mean(abs(arrays[f'{method}_spectra'][idx]-data['y_test'][idx])))
        np.testing.assert_allclose(float(row['spectral_mae']),mae,atol=1e-13,rtol=0)
    check={'status':'passed','fixed_dataset_samples':5000,'held_out_targets':500,
           'independent_targets':10000,'tested_models':5,'tmm_energy_conservation':True,
           'test_metrics_recomputed':True,'all_final_thicknesses_feasible':True,
           'refinement_best_loss_monotone':True,'nearest_library_training_only':True}
    (output/'verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(check,ensure_ascii=False,indent=2))
    return check


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('results/inverse'))
    args=p.parse_args();verify(args.output)
if __name__=='__main__':main()
