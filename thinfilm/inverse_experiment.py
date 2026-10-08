"""Run the complete inverse-design add-on without modifying forward results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .config import Config
from .data import make_dataset, make_design_candidates
from .experiment import write_csv
from .inverse import (train_inverse, predict_inverse, evaluate_inverse,
                      nearest_training_structure, make_mlp_starts, refine_inverse)
from .inverse_plots import make_inverse_figures
from .physics import tmm_reflectance


def load_fixed_dataset(source: Path, config: Config) -> tuple[dict, str | None]:
    path = source / 'data' / 'dataset.npz'
    expected = make_dataset(config)
    if path.exists():
        with np.load(path) as saved:
            data = {key: saved[key] for key in expected.__dict__}
        for key, value in expected.__dict__.items():
            np.testing.assert_array_equal(data[key], value)
        return data, hashlib.sha256(path.read_bytes()).hexdigest()
    return expected.__dict__, None


def save_fit(output: Path, fit, size: int, method: str, config: Config) -> None:
    torch.save(fit.model.state_dict(), output / 'models' / f'{method}_{size}.pt')
    (output / 'histories' / f'{method}_{size}.json').write_text(
        json.dumps({**fit.history, 'best_epoch': fit.best_epoch,
                    'best_validation_loss': fit.best_validation_loss, 'loss_kind': fit.loss_kind}), encoding='utf-8')


def report_markdown(output: Path, summary: dict) -> None:
    names = {'supervised': '直接反向MLP（边界投影后）', 'physics': '物理约束反向MLP（一次推理）',
             'refined': 'MLP多起点＋TMM优化', 'nearest': '训练库光谱最近邻'}
    rows = []
    for method in names:
        m = summary['methods'][method]
        rows.append(f"| {names[method]} | {m['thickness_mae_nm']:.4f} | {m['spectral_mse']:.7g} | {m['spectral_mae']:.6f} | {m['target_520nm_mae']:.6f} | {m['p90_spectrum_mae']:.6f} |")
    baseline = summary['methods']['supervised']; physics = summary['methods']['physics']; refined = summary['methods']['refined']
    sizes = '\n'.join(f"| {r['training_samples']} | {r['raw_thickness_mae_nm']:.4f} | {r['thickness_mae_nm']:.4f} | {r['spectral_mae']:.6f} | {r['best_epoch']} |" for r in summary['sizes_results'])
    report = f"""# 附加实验 目标反射光谱到四层膜厚的反向预测

姓名：陈新堉　学号：2023270069

## 1 任务与条件

输入为400、410、…、800 nm的41点目标反射率，输出从空气侧依次排列的四层膜厚[d1,d2,d3,d4]。本轮反向实验依据新增任务开展。结构为Air/H/L/H/L/Glass，正入射、无损、忽略色散，折射率依次为1.00/2.30/1.45/2.30/1.45/1.52；玻璃为半无限基底。膜厚范围40–180 nm。复用原5000组数据和4000/500/500划分，seed=270069、design_seed=270070，个性化波长520 nm。

## 2 模型与评价

直接反向MLP为41–128–128–64–4，隐藏层ReLU，输出线性层。输入使用原始反射率[0,1]，膜厚标签按2(d−40)/140−1归一化。与正向实验一致，Adam学习率0.001、batch=128、训练{summary['epochs']}轮；按500组验证集最小归一化膜厚MSE选权重。依次使用原训练池前500/1000/2000/4000组，测试集固定500组。

直接模型的原始输出膜厚MAE为{baseline['raw_thickness_mae_nm']:.4f} nm，2000个膜厚值中{baseline['raw_out_of_bounds_count']}个越界；原始输出及越界统计已完整保存。实现合法膜结构时将输出投影到[40,180]，表中的直接模型光谱指标由投影后的结构经TMM回算。投影前后指标分别记录，未把投影隐去。

额外训练物理约束反向MLP：隐藏宽度、数据划分、初始化种子和训练参数相同；输出增加tanh以保证膜厚范围。其损失为精确可微TMM的回算光谱与输入光谱之间的MSE，不使用膜厚标签监督；按验证光谱MSE选权重。该模型与直接模型的损失不同，不能比较两个损失值的数值大小。

TMM优化为独立推理步骤：使用直接MLP、物理MLP及物理MLP的5/10/20 nm固定种子扰动，共5个起点，进行{summary['refinement_steps']}步有边界Adam优化，归一化膜厚学习率{summary['refinement_learning_rate']}。每个起点保存历史最小光谱MSE，最后依据已输入的目标光谱选取最小误差结构。优化既不读取目标的真实膜厚，也不更新MLP权重。最近邻对照仅检索4000组训练光谱。

## 3 固定500组测试结果

| 方法 | 膜厚MAE (nm) | TMM回算光谱MSE | TMM回算光谱MAE | 520 nm MAE | 单样本光谱MAE的P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

物理约束MLP一次推理的光谱MAE为{physics['spectral_mae']:.6f}，本轮未改善直接MLP的整体光谱MAE；它的中位数较低，但大误差尾部更长。最近邻检索的光谱MAE为{summary['methods']['nearest']['spectral_mae']:.6f}，优于两种MLP的一次推理均值，因此当前结果不支持反向MLP已经优于简单检索的结论。

多起点TMM优化后的光谱MAE为{refined['spectral_mae']:.6f}，单样本中位数为{refined['median_spectrum_mae']:.3e}，最大值为{refined['max_spectrum_mae']:.6f}。优化后光谱MAE小于0.01的样本比例为{100*refined['spectra_mae_below_0_01_fraction']:.1f}%。这些优化结果属于组合流程，不能描述为MLP一次推理的精度；测试目标均由相同TMM模型生成，属于可实现目标，不代表实验测量精度。

## 4 训练样本量效应

| 训练样本数 | 原始膜厚MAE (nm) | 合法结构膜厚MAE (nm) | 回算光谱MAE | 最优验证轮次 |
| ---: | ---: | ---: | ---: | ---: |
{sizes}

## 5 理论解释与结果边界

反射率只保留幅度信息，逆映射可能存在多个或近似等价结构。因此膜厚误差与实现目标光谱的误差必须同时考察，不能把原膜厚标签当作唯一合法答案。所有光谱指标均由真实TMM复算，未使用正向MLP代理评分。研究边界为无损、无色散、正入射及固定H/L/H/L层序；对任意输入光谱不保证该膜系能够实现，也不保证有边界局部优化得到全局最优。结果来自固定单一随机种子，不提供未经重复实验估计的置信区间。

已额外使用design_seed独立生成10000组新结构和目标光谱，评价两种反向MLP一次推理的泛化；结果记录于summary.json的independent_10000字段。新目标不参与训练或权重选择。examples中提供原正向筛选Top5候选的完整目标光谱及反向设计结果，可直接用CLI预测。

## 6 图表

"""
    captions = ['反向任务流程与物理复核', '直接与物理约束模型的原始训练损失',
                '四层膜厚恢复及近似等价结构', '按固定误差分位选择的代表性光谱',
                '相同划分下的训练样本量效应', '500组测试误差分布与优化历史', '优化后最大误差案例']
    for i, (filename, title) in enumerate(zip(summary['figure_files'], captions), start=1):
        report += f'### 图{i} {title}\n\n![{title}](figures/{filename})\n\n'
    report += """## 7 运行与复核

在仓库目录运行：

```powershell
python -m thinfilm.inverse_experiment --source results --output results/inverse
python -m thinfilm.inverse_verify --output results/inverse
python -m thinfilm.inverse_predict --spectrum results/inverse/examples/target_candidate_8667.csv --output results/inverse/custom_prediction --refine-steps 400
```

模型文件保存在models，原始损失在histories，逐测试样本结果在test_predictions.csv，数组及优化历史在data/test_results.npz。完整精度以CSV和JSON为准，图表只按显示精度标注。

## 理论参考

- Liu D, Tan Y, Khoram E, Yu Z. Training Deep Neural Networks for the Inverse Design of Nanophotonic Structures. ACS Photonics 2018, 5:1365–1369. https://doi.org/10.1021/acsphotonics.7b01377
- Ma TG, Wang H, Guo LJ. OptoGPT: A foundation model for inverse design in optical multilayer thin film structures. Opto-Electronic Advances 2024, 7:240062. https://doi.org/10.29026/oea.2024.240062
"""
    (output / '附加题_反向预测结果.md').write_text(report, encoding='utf-8')


def run(source: Path, output: Path, epochs: int = 300, batch_size: int = 128,
        learning_rate: float = 1e-3, refinement_steps: int = 400,
        refinement_learning_rate: float = .02) -> dict:
    config = Config(); output.mkdir(parents=True, exist_ok=True)
    for name in ('data', 'models', 'histories', 'examples'):
        (output / name).mkdir(exist_ok=True)
    data, source_hash = load_fixed_dataset(source, config)
    np.savez_compressed(output / 'data' / 'dataset.npz', **data)
    (output / 'config.json').write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    sizes_results = []; supervised_fit = None; supervised_raw = None
    for size in config.training_sizes:
        print(f'Supervised inverse MLP: {size} training samples, {epochs} epochs', flush=True)
        start = perf_counter()
        fit = train_inverse(data['y_train'][:size], data['x_train'][:size], data['y_validation'], data['x_validation'], config,
                            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        seconds = perf_counter() - start
        raw = predict_inverse(fit.model, data['y_test'], config)
        legal = np.clip(raw, config.thickness_min_nm, config.thickness_max_nm)
        metrics, _ = evaluate_inverse(data['y_test'], data['x_test'], legal, config)
        metrics.update(training_samples=size, best_epoch=fit.best_epoch, validation_loss=fit.best_validation_loss,
                       raw_thickness_mae_nm=float(np.mean(np.abs(raw-data['x_test']))),
                       raw_thickness_rmse_nm=float(np.sqrt(np.mean((raw-data['x_test'])**2))),
                       raw_out_of_bounds_count=int(np.count_nonzero((raw<40)|(raw>180))),
                       training_seconds=seconds)
        sizes_results.append(metrics); save_fit(output, fit, size, 'supervised', config)
        print(f"  test thickness MAE={metrics['thickness_mae_nm']:.4f} nm; roundtrip spectral MAE={metrics['spectral_mae']:.6f}", flush=True)
        if size == config.n_train:
            supervised_fit = fit; supervised_raw = raw
    assert supervised_fit is not None and supervised_raw is not None
    print('Physics-constrained inverse MLP: exact TMM loss, 4000 training spectra', flush=True)
    start = perf_counter()
    physics_fit = train_inverse(data['y_train'], data['x_train'], data['y_validation'], data['x_validation'], config,
                                physics_loss=True, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    physics_seconds = perf_counter() - start
    save_fit(output, physics_fit, config.n_train, 'physics', config)
    supervised_d = np.clip(supervised_raw, 40, 180)
    physics_d = predict_inverse(physics_fit.model, data['y_test'], config)
    nearest_d, nearest_indices = nearest_training_structure(data['y_test'], data['y_train'], data['x_train'])
    print(f'Refining all {config.n_test} held-out targets with 5 MLP-derived starts', flush=True)
    start = perf_counter()
    refinement = refine_inverse(data['y_test'], make_mlp_starts(supervised_raw, physics_d, config), config,
                                steps=refinement_steps, learning_rate=refinement_learning_rate)
    refinement_seconds = perf_counter() - start
    assert np.all(refinement['final_mse'] <= refinement['initial_mse'] + 1e-12)
    assert np.all(np.diff(refinement['best_mse_history'], axis=0) <= 1e-14)
    methods = {}; spectra = {}
    structures = {'supervised':supervised_d, 'physics':physics_d, 'refined':refinement['thickness_nm'], 'nearest':nearest_d}
    for name, d in structures.items():
        methods[name], spectra[name] = evaluate_inverse(data['y_test'], data['x_test'], d, config)
    methods['supervised'].update(raw_thickness_mae_nm=sizes_results[-1]['raw_thickness_mae_nm'],
                                 raw_thickness_rmse_nm=sizes_results[-1]['raw_thickness_rmse_nm'],
                                 raw_out_of_bounds_count=sizes_results[-1]['raw_out_of_bounds_count'])
    sample_mae = np.mean(np.abs(spectra['refined']-data['y_test']), axis=1)
    ordered = np.argsort(sample_mae, kind='stable')
    representatives = [int(ordered[round(q*(config.n_test-1))]) for q in (.25,.5,.75)]
    failure = int(np.argmax(sample_mae))
    arrays = {'true_thickness':data['x_test'], 'target_spectra':data['y_test'],
              'supervised_raw_thickness':supervised_raw, 'nearest_training_indices':nearest_indices,
              'training_sizes':np.asarray(config.training_sizes),
              'size_thickness_mae_nm':np.array([m['thickness_mae_nm'] for m in sizes_results]),
              'size_spectral_mae':np.array([m['spectral_mae'] for m in sizes_results]),
              'loss_supervised_train':np.array(supervised_fit.history['train_loss']),
              'loss_supervised_validation':np.array(supervised_fit.history['validation_loss']),
              'loss_physics_train':np.array(physics_fit.history['train_loss']),
              'loss_physics_validation':np.array(physics_fit.history['validation_loss']),
              'refinement_best_mse_history':refinement['best_mse_history'],
              'refinement_start_index':refinement['start_index']}
    for name in structures:
        arrays[f'{name}_thickness'] = structures[name]; arrays[f'{name}_spectra'] = spectra[name]
    np.savez_compressed(output/'data'/'test_results.npz', **arrays)

    test_rows = []
    for name, d in structures.items():
        for idx in range(config.n_test):
            error = spectra[name][idx]-data['y_test'][idx]
            row = {'method':name,'test_index':idx,'spectral_mae':float(np.mean(abs(error))),
                   'spectral_mse':float(np.mean(error**2)), 'target_520nm_abs_error':float(abs(error[config.target_index])),
                   'thickness_mae_nm':float(np.mean(abs(d[idx]-data['x_test'][idx])))}
            for j in range(4):
                row[f'd{j+1}_true_nm']=float(data['x_test'][idx,j]); row[f'd{j+1}_predicted_nm']=float(d[idx,j])
            test_rows.append(row)
    write_csv(output/'test_predictions.csv', test_rows)
    write_csv(output/'training_size_metrics.csv', [{k:v for k,v in m.items() if not isinstance(v,list)} for m in sizes_results])
    write_csv(output/'method_metrics.csv', [{'method':name,**{k:v for k,v in metrics.items() if not isinstance(v,list)}} for name,metrics in methods.items()])

    # New targets from the same independent 10000-structure pool as the forward task.
    print('Evaluating one-pass inverses on 10000 independent targets', flush=True)
    candidates = make_design_candidates(config)
    goals = tmm_reflectance(candidates, config.wavelengths_nm, config.refractive_indices, config.ambient_index, config.substrate_index)
    pool_raw = predict_inverse(supervised_fit.model, goals, config)
    pool_supervised = np.clip(pool_raw,40,180); pool_physics = predict_inverse(physics_fit.model,goals,config)
    independent = {}
    independent['supervised'], pool_sr = evaluate_inverse(goals,candidates,pool_supervised,config)
    independent['physics'], pool_pr = evaluate_inverse(goals,candidates,pool_physics,config)
    np.savez_compressed(output/'data'/'independent_targets.npz',true_thickness=candidates,target_spectra=goals,
                        supervised_raw_thickness=pool_raw, supervised_thickness=pool_supervised,
                        physics_thickness=pool_physics,supervised_spectra=pool_sr,physics_spectra=pool_pr)
    forward_top5 = source/'top5_designs.csv'
    if forward_top5.exists():
        with forward_top5.open(encoding='utf-8-sig',newline='') as handle:
            selected = [int(row['candidate_index']) for row in csv.DictReader(handle)]
        example_rule = 'original_forward_Top5_candidates'
    else:
        selected = np.argsort(-goals[:,config.target_index],kind='stable')[:5].tolist()
        example_rule = 'independent_pool_TMM_top5'
    example_refinement = refine_inverse(goals[selected],make_mlp_starts(pool_raw[selected],pool_physics[selected],config),config,
                                        steps=refinement_steps,learning_rate=refinement_learning_rate)
    example_rows=[]
    for position, idx in enumerate(selected):
        write_csv(output/'examples'/f'target_candidate_{idx}.csv',
                  [{'wavelength_nm':float(w),'reflectance':float(r)} for w,r in zip(config.wavelengths_nm,goals[idx])])
        for name,d,r in [('supervised',pool_supervised[idx],pool_sr[idx]),('physics',pool_physics[idx],pool_pr[idx]),
                         ('refined',example_refinement['thickness_nm'][position],example_refinement['spectra'][position])]:
            err=r-goals[idx]
            row={'candidate_index':idx,'method':name,'spectral_mae':float(np.mean(abs(err))),
                 'spectral_mse':float(np.mean(err**2)),'target_R520':float(goals[idx,config.target_index]),
                 'reconstructed_R520':float(r[config.target_index])}
            for j in range(4): row[f'd{j+1}_true_nm']=float(candidates[idx,j]);row[f'd{j+1}_predicted_nm']=float(d[j])
            example_rows.append(row)
    write_csv(output/'examples'/'inverse_design_examples.csv',example_rows)

    summary={'identity':{'student_name':config.student_name,'student_id':config.student_id},
             'task':'41-point target reflectance spectrum -> four film thicknesses',
             'architecture':[41,128,128,64,4], 'split':[4000,500,500], 'seed':config.seed,'design_seed':config.design_seed,
             'target_wavelength_nm':config.target_wavelength_nm,'epochs':epochs,'batch_size':batch_size,
             'learning_rate':learning_rate,'source_dataset_sha256':source_hash,
             'input_scaling':'raw reflectance in [0,1]', 'thickness_scaling':'2*(d-40)/140-1',
             'supervised_output':'linear; raw outputs saved; feasibility projection reported separately',
             'physics_output':'tanh normalized thickness; exact TMM spectrum loss, no thickness labels used',
             'sizes_results':sizes_results, 'methods':methods,
             'best_epoch_supervised':supervised_fit.best_epoch,'best_epoch_physics':physics_fit.best_epoch,
             'physics_training_seconds':physics_seconds,'refinement_seconds':refinement_seconds,
             'refinement_steps':refinement_steps,'refinement_learning_rate':refinement_learning_rate,
             'refinement_starts':5,'refinement_start_rule':['supervised','physics','physics+normal(0,5nm)','physics+normal(0,10nm)','physics+normal(0,20nm)'],
             'refinement_seed':config.design_seed,'refinement_uses_true_thickness':False,
             'representative_test_indices':representatives,'failure_test_index':failure,
             'failure_spectrum_mae':float(sample_mae[failure]),
             'independent_10000':independent,'example_candidate_indices':selected,'example_selection_rule':example_rule,
             'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__}}
    summary['figure_files']=make_inverse_figures(output,config,arrays,summary)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    report_markdown(output,summary)
    print(json.dumps({'methods':methods,'output':str(output)},ensure_ascii=False,indent=2),flush=True)
    return summary


def main() -> None:
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results'))
    parser.add_argument('--output',type=Path,default=Path('results/inverse'))
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--learning-rate',type=float,default=1e-3)
    parser.add_argument('--refinement-steps',type=int,default=400)
    parser.add_argument('--refinement-learning-rate',type=float,default=.02)
    args=parser.parse_args()
    run(args.source,args.output,args.epochs,args.batch_size,args.learning_rate,args.refinement_steps,args.refinement_learning_rate)


if __name__=='__main__': main()
