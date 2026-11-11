"""Controlled serving-shape costs; synthetic inputs are not accuracy evidence.

Run on an idle CUDA device. Router timings include base features, CPU routing,
weight replacement and expert forward; prefill/classification only, no decoding.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'merge_study'))
from common import environment, read_config, write_json
from engine import DeltaEvaluator


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    values = []
    with torch.inference_mode():
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000)
    return {'wall_ms': values, 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/protocol-v2-a100.json')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--rounds', type=int, default=3)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not run this benchmark on the local CPU')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    root = CODE.parent
    cfg = read_config(args.config)
    plan = {'seed': 42, 'input_seed': 950109, 'dtype': 'float32',
            'batch_sizes': [1, 4], 'sequence_lengths': [32, 128],
            'warmup': args.warmup, 'repeats': args.repeats, 'rounds': args.rounds,
            'task_head': 'sst2 (known)', 'inputs': 'fixed synthetic token IDs, all tokens valid',
            'timing': 'synchronized wall time; excludes tokenizer, H2D input transfer, model loading and one-time merge',
            'router': 'frozen base mean features -> archived linear source classifier -> expert; known task head',
            'scope': 'single GPU, seed42 bank; not accuracy evaluation or multi-seed latency inference',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(out / 'plan.json', plan)  # Frozen before model loading or measurements.
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(plan['input_seed'])
    engine = DeltaEvaluator.from_bank(cfg, Path(args.bank))
    engine.model.eval()
    if next(engine.model.parameters()).dtype != torch.float32:
        raise AssertionError('Expected float32 model')
    selections = read_config(root / 'artifacts/coefficient-v1-s42-2026-10-25.json')['selection']
    tasks = cfg['tasks']
    zero = {k: torch.zeros_like(v) for k, v in engine.banks[tasks[0]].items()}
    kl = selections['layer_kl']
    weights = torch.tensor(kl['weights'])
    merged = {k: sum(engine.banks[t][k] * float(weights[int(k.split('.')[2]), i]) * kl['gain']
                     for i, t in enumerate(tasks)) for k in engine.base}
    uniform = {k: sum(engine.banks[t][k] for t in tasks) / len(tasks) for k in engine.base}
    archived_router = read_config(root / 'artifacts/router-v1-summary-2026-10-25.json')['selection']
    mean, scale, coef, intercept = [np.array(archived_router[k]) for k in
                                  ('scaler_mean', 'scaler_scale', 'linear_coef', 'linear_intercept')]
    router_choices = []

    @torch.inference_mode()
    def features(inputs):
        hidden = engine.model.model(**inputs).last_hidden_state.float()
        mask = inputs['attention_mask'].unsqueeze(-1)
        return ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu().numpy()

    @torch.inference_mode()
    def route(inputs, bank, zero_state):
        engine.set_state('sst2', zero_state)
        x = features(inputs)
        chosen = (((x - mean) / scale) @ coef.T + intercept).argmax(1)
        router_choices.append(chosen.tolist())
        logits = torch.empty((len(chosen), 2), device=engine.device)
        for i in np.unique(chosen):
            idx = np.flatnonzero(chosen == i)
            engine.set_state('sst2', bank[tasks[int(i)]])
            sub = {k: v[torch.as_tensor(idx, device=v.device)] for k, v in inputs.items()}
            logits[torch.as_tensor(idx, device=logits.device)] = engine.model(**sub).logits
        return logits

    rows = []
    rng = random.Random(950109)
    def inputs_for(batch_size, length):
        generator = torch.Generator(device=engine.device).manual_seed(plan['input_seed'] + batch_size + length)
        ids = torch.randint(10, 50000, (batch_size, length), device=engine.device, generator=generator)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    for batch_size in plan['batch_sizes']:
        for length in plan['sequence_lengths']:
            inputs = inputs_for(batch_size, length)
            assert int(inputs['input_ids'].max()) < engine.model.config.vocab_size
            # Equal synthetic input shapes isolate execution costs from text length.
            candidates = [('individual_materialized', engine.banks['sst2']),
                          ('average_materialized', uniform), ('layer_kl_materialized', merged)]
            for round_id in range(args.rounds):
                rng.shuffle(candidates)
                for name, delta in candidates:
                    engine.set_state('sst2', delta)
                    with torch.inference_mode():
                        first = engine.model(**inputs).logits
                        assert torch.isfinite(first).all()
                    result = measure(lambda: engine.model(**inputs).logits, args.warmup, args.repeats)
                    rows.append(dict(method=name, batch_size=batch_size, sequence_length=length,
                                     round=round_id, **result))
                result = measure(lambda: route(inputs, engine.banks, zero), args.warmup, args.repeats)
                rows.append(dict(method='router_cpu_delta_switch', batch_size=batch_size,
                                 sequence_length=length, round=round_id, **result))
                write_json(out / 'progress.json', {'state': 'running', 'rows': rows})
    # GPU-cached dense deltas trade additional resident memory for faster swaps.
    cached = {t: {k: v.to(engine.device) for k, v in bank.items()} for t, bank in engine.banks.items()}
    zero_cached = {k: v.to(engine.device) for k, v in zero.items()}
    for batch_size in plan['batch_sizes']:
        for length in plan['sequence_lengths']:
            inputs = inputs_for(batch_size, length)
            with torch.inference_mode():
                a = route(inputs, engine.banks, zero)
                b = route(inputs, cached, zero_cached)
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
            for round_id in range(args.rounds):
                result = measure(lambda: route(inputs, cached, zero_cached), args.warmup, args.repeats)
                rows.append(dict(method='router_gpu_cached_dense_delta_switch', batch_size=batch_size,
                                 sequence_length=length, round=round_id, **result))
    # Storage: distinguish serialized files, tensor payloads and parameter bytes.
    adapter_bytes = {t: sum(f.stat().st_size for f in (Path(args.bank) / t / 'adapter').rglob('*') if f.is_file()) for t in tasks}
    model_bytes = sum(p.numel() * p.element_size() for p in engine.model.parameters())
    storage = {'base_classifier_parameter_bytes': model_bytes,
               'adapter_directory_bytes': adapter_bytes,
               'four_adapter_directory_bytes': sum(adapter_bytes.values()),
               'four_task_head_parameter_bytes': sum(t.numel()*t.element_size() for t in engine.heads.values()),
               'linear_router_and_scaler_float64_payload_bytes': sum(a.nbytes for a in (mean, scale, coef, intercept)),
               'layer_coefficients_trainable_scalars': 97,
               'layer_coefficients_float32_payload_bytes': 97 * 4,
               'cached_dense_delta_gpu_bytes': sum(v.numel()*v.element_size() for d in cached.values() for v in d.values())}
    groups = []
    for method, batch, length in sorted({(r['method'],r['batch_size'],r['sequence_length']) for r in rows}):
        selected = [r for r in rows if (r['method'],r['batch_size'],r['sequence_length']) == (method,batch,length)]
        values = [v for r in selected for v in r['wall_ms']]
        med = float(np.median(values))
        groups.append({'method': method, 'batch_size': batch, 'sequence_length': length,
                       'median_ms': med, 'p95_ms': float(np.percentile(values,95)),
                       'samples_per_second': batch * 1000 / med,
                       'round_medians_ms': [r['median_ms'] for r in selected],
                       'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in selected),
                       'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in selected)})
    result = {'state':'passed', 'plan':plan, 'environment':environment(), 'rows':rows,
              'summary':groups,'storage':storage,'route_choices':router_choices,
              'limitations':['Synthetic token IDs: no task performance claim.',
                 'GPU-cached measurements run after CPU-switch measurements; not thermally interleaved.',
                 'Memory includes evaluator base snapshots and caches; not a minimal serving implementation.',
                 'Dense delta replacement is this prototype, not optimized PEFT hot swapping.',
                 'One merged model still needs four task heads; unknown task/head selection is unsolved.',
                 'Timing repeats measure system variability, not independent training seeds.']}
    write_json(out/'summary.json',result)
    print('DEPLOYMENT_BENCHMARK_COMPLETE',out)


if __name__ == '__main__':
    main()
