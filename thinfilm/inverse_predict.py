"""Predict four thicknesses from a 41-row wavelength/reflectance CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .experiment import write_csv
from .inverse import InverseMLP, predict_inverse, make_mlp_starts, refine_inverse
from .physics import tmm_reflectance


def load_target(path: Path, config: Config) -> np.ndarray:
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    try:
        wave=np.array([float(r['wavelength_nm']) for r in rows])
        values=np.array([float(r['reflectance']) for r in rows])
    except (KeyError,ValueError) as exc:
        raise ValueError('CSV columns must be wavelength_nm and reflectance, with numeric values') from exc
    if wave.shape != config.wavelengths_nm.shape or not np.array_equal(wave,config.wavelengths_nm):
        raise ValueError('Expected exactly 41 rows, ordered 400,410,...,800 nm')
    if not np.all(np.isfinite(values)) or np.any((values<0)|(values>1)):
        raise ValueError('Reflectance must be finite and in [0,1]')
    return values[None,:]


def run(spectrum: Path, model_dir: Path, output: Path, method: str = 'physics',
        refine_steps: int = 0, refinement_learning_rate: float = .02) -> dict:
    config=Config(); torch.set_num_threads(1)
    target=load_target(spectrum,config)
    predictions={}
    for kind in ('supervised','physics'):
        model=InverseMLP(bounded=kind=='physics')
        model.load_state_dict(torch.load(model_dir/f'{kind}_4000.pt',map_location='cpu',weights_only=True))
        predictions[kind]=predict_inverse(model,target,config)
    raw=predictions[method]
    feasible=np.clip(raw,config.thickness_min_nm,config.thickness_max_nm)
    one_pass=tmm_reflectance(feasible,config.wavelengths_nm,config.refractive_indices,config.ambient_index,config.substrate_index)[0]
    final_d=feasible[0]; final_r=one_pass; refined=False; selected_start=None
    if refine_steps:
        result=refine_inverse(target,make_mlp_starts(predictions['supervised'],predictions['physics'],config),config,
                              steps=refine_steps,learning_rate=refinement_learning_rate)
        final_d=result['thickness_nm'][0]; final_r=result['spectra'][0]
        refined=True; selected_start=int(result['start_index'][0])
    output.mkdir(parents=True,exist_ok=True)
    result={'student_name':config.student_name,'student_id':config.student_id,
            'input_spectrum_csv':str(spectrum),'method':method,'layer_order':['H','L','H','L'],
            'raw_one_pass_thickness_nm':raw[0].tolist(),'feasible_one_pass_thickness_nm':feasible[0].tolist(),
            'raw_out_of_bounds_count':int(np.count_nonzero((raw<40)|(raw>180))),
            'one_pass_spectral_mae':float(np.mean(abs(one_pass-target[0]))),
            'one_pass_spectral_mse':float(np.mean((one_pass-target[0])**2)),
            'tmm_refinement_applied':refined,'refinement_steps':refine_steps,
            'refinement_learning_rate':refinement_learning_rate,'selected_refinement_start':selected_start,
            'final_thickness_nm':final_d.tolist(),'final_spectral_mae':float(np.mean(abs(final_r-target[0]))),
            'final_spectral_mse':float(np.mean((final_r-target[0])**2)),
            'target_wavelength_nm':config.target_wavelength_nm,
            'target_reflectance_at_520nm':float(target[0,config.target_index]),
            'final_tmm_reflectance_at_520nm':float(final_r[config.target_index]),
            'target_520nm_abs_error':float(abs(final_r[config.target_index]-target[0,config.target_index]))}
    (output/'prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(output/'predicted_thickness.csv',[{'layer':j+1,'type':['H','L','H','L'][j],
               'refractive_index':config.refractive_indices[j],'raw_mlp_thickness_nm':float(raw[0,j]),
               'feasible_mlp_thickness_nm':float(feasible[0,j]),'final_thickness_nm':float(final_d[j])} for j in range(4)])
    write_csv(output/'reconstructed_spectrum.csv',[{'wavelength_nm':float(w),'target_reflectance':float(t),
               'mlp_tmm_reflectance':float(m),'final_tmm_reflectance':float(r)}
               for w,t,m,r in zip(config.wavelengths_nm,target[0],one_pass,final_r)])
    plt.rcParams.update({'font.family':['Microsoft YaHei','Arial'],'axes.unicode_minus':False,
                         'pdf.fonttype':42,'svg.fonttype':'none','font.size':8,
                         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,ax=plt.subplots(figsize=(6.65,3.0))
    ax.plot(config.wavelengths_nm,target[0],color='#263441',lw=1.5,label='目标光谱')
    ax.plot(config.wavelengths_nm,one_pass,color='#7D9B87',lw=1.25,ls=(0,(4,2)),label='反向MLP → TMM')
    if refined:ax.plot(config.wavelengths_nm,final_r,color='#3879AB',lw=1.3,ls=(0,(2,2)),label='多起点优化 → TMM')
    ax.set(xlim=(400,800),ylim=(0,1),xlabel='波长 (nm)',ylabel='反射率')
    ax.legend(loc='upper right',fontsize=7.5)
    ax.text(.02,.96,f"光谱 MAE = {result['final_spectral_mae']:.3e}",transform=ax.transAxes,va='top',fontsize=7.5)
    fig.subplots_adjust(left=.10,right=.97,bottom=.23,top=.93)
    fig.text(.5,.055,'最终膜厚 (nm)：'+', '.join(f'{v:.2f}' for v in final_d),ha='center',fontsize=8)
    for ext in ('png','pdf','svg'):fig.savefig(output/f'inverse_prediction.{ext}',dpi=600)
    plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main() -> None:
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spectrum',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=Path('results/inverse/models'))
    p.add_argument('--output',type=Path,default=Path('results/inverse/custom_prediction'))
    p.add_argument('--method',choices=['supervised','physics'],default='physics')
    p.add_argument('--refine-steps',type=int,default=0)
    p.add_argument('--refinement-learning-rate',type=float,default=.02)
    args=p.parse_args()
    if args.refine_steps<0:p.error('--refine-steps must be nonnegative')
    run(args.spectrum,args.model_dir,args.output,args.method,args.refine_steps,args.refinement_learning_rate)


if __name__=='__main__':main()
