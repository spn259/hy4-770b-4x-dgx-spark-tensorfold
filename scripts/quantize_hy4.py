#!/usr/bin/env python3
"""Streaming GPU NVFP4 quantizer for tencent/Hy4-FP8 -> 4-rank sharded TensorFold checkpoint."""
import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = 'tencent/Hy4-FP8'
BASE_URL = f'https://huggingface.co/{REPO}/resolve/main'

# E2M1 positive representable values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
# Decision midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)

EXPERT_RE = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale)$')
SHARED_OR_DENSE_RE = re.compile(r'^model\.layers\.(\d+)\.mlp\.(?:shared_experts\.)?(gate_proj|up_proj|down_proj)\.(weight|weight_scale)$')


def dequant_block_fp8(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize 128x128 block FP8 (E4M3) to FP32 on GPU."""
    M, K = w_fp8.shape
    w = w_fp8.to(torch.float32)
    s = scale.to(torch.float32)
    s_exp = s.repeat_interleave(128, dim=0)[:M, :].repeat_interleave(128, dim=1)[:, :K]
    return w * s_exp


def quantize_nvfp4(w_f32: torch.Tensor, midpoints: torch.Tensor):
    """Quantize [M, K] FP32 tensor to ModelOpt NVFP4 (uint8 [M, K//2], float8_e4m3fn [M, K//16], f32 scalar)."""
    M, K = w_f32.shape
    assert K % 16 == 0, f'K={K} not divisible by 16'
    amax = w_f32.abs().max()
    if amax.item() == 0.0:
        w_packed = torch.zeros((M, K // 2), dtype=torch.uint8, device=w_f32.device)
        w_scale = torch.zeros((M, K // 16), dtype=torch.float8_e4m3fn, device=w_f32.device)
        w_scale_2 = torch.tensor(1.0, dtype=torch.float32, device=w_f32.device)
        return w_packed, w_scale, w_scale_2

    w_scale_2 = amax / (6.0 * 448.0)
    w_grp = w_f32.view(M, K // 16, 16)
    g_amax = w_grp.abs().amax(dim=-1)
    s_f32 = (g_amax / (6.0 * w_scale_2)).clamp(0.0, 448.0)
    w_scale = s_f32.to(torch.float8_e4m3fn)

    eff_scale = (w_scale.to(torch.float32) * w_scale_2).unsqueeze(-1)
    w_norm = torch.where(eff_scale > 0, w_grp / eff_scale, torch.zeros_like(w_grp))
    sign = (w_norm < 0).to(torch.uint8) << 3
    abs_val = w_norm.abs()
    # Bucketize into 0..7 using decision midpoints
    mag = torch.bucketize(abs_val, midpoints).to(torch.uint8)
    nibbles = (sign | mag).view(M, K // 2, 2)
    w_packed = nibbles[:, :, 0] | (nibbles[:, :, 1] << 4)
    return w_packed, w_scale, w_scale_2


def dequant_nvfp4_check(w_packed: torch.Tensor, w_scale: torch.Tensor, w_scale_2: torch.Tensor, lut: torch.Tensor):
    M, K2 = w_packed.shape
    K = K2 * 2
    lo = (w_packed & 0x0F).to(torch.int64)
    hi = ((w_packed >> 4) & 0x0F).to(torch.int64)
    nib = torch.stack([lo, hi], dim=-1).view(M, K // 16, 16)
    vals = lut[nib]
    eff = (w_scale.to(torch.float32) * w_scale_2).unsqueeze(-1)
    return (vals * eff).view(M, K)


def download_file_parallel(url: str, dst: Path, chunks: int = 8):
    """Fast parallel range download for a single safetensors shard."""
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers['Content-Length'])
    if dst.exists() and dst.stat().st_size == total:
        return dst
    tmp = dst.with_suffix('.part')
    with open(tmp, 'wb') as f:
        f.truncate(total)
    if total < 16 * 1024 * 1024:
        chunks = 1
    step = (total + chunks - 1) // chunks

    def fetch(i):
        s = i * step
        e = min(total - 1, s + step - 1)
        if s > e:
            return
        for attempt in range(5):
            try:
                r = urllib.request.Request(url, headers={'Range': f'bytes={s}-{e}'})
                with urllib.request.urlopen(r, timeout=120) as resp:
                    data = resp.read()
                if len(data) != (e - s + 1):
                    raise IOError(f'short read {len(data)} vs {e - s + 1}')
                with open(tmp, 'r+b') as f:
                    f.seek(s)
                    f.write(data)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=chunks) as ex:
        list(ex.map(fetch, range(chunks)))
    tmp.replace(dst)
    return dst


def process_shard(shard_name: str, shard_path: Path, out_dir: Path, device: torch.device,
                  midpoints: torch.Tensor, verify: bool = False):
    tensors = load_file(str(shard_path), device='cpu')
    # Pair up weight + weight_scale for FP8 tensors
    scales = {}
    weights = {}
    for k, v in tensors.items():
        if k.startswith('vit.'):
            continue
        if k.endswith('.weight_scale'):
            scales[k[:-len('.weight_scale')]] = v
        else:
            weights[k] = v

    rank_buckets = {r: {} for r in range(4)}
    common_bucket = {}
    cos_list = []

    for k, v in weights.items():
        base = k[:-len('.weight')] if k.endswith('.weight') else None
        is_fp8 = (v.dtype == torch.float8_e4m3fn) and (base in scales)

        m_exp = EXPERT_RE.match(k)
        if m_exp:
            layer_i = int(m_exp.group(1))
            exp_i = int(m_exp.group(2))
            rank = exp_i // 64
            w_f32 = dequant_block_fp8(v.to(device), scales[base].to(device)) if is_fp8 else v.to(device, dtype=torch.float32)
            wp, ws, ws2 = quantize_nvfp4(w_f32, midpoints)
            if verify and len(cos_list) < 4:
                lut = E2M1_LUT.to(device)
                rec = dequant_nvfp4_check(wp, ws, ws2, lut)
                cos = torch.nn.functional.cosine_similarity(w_f32.flatten(), rec.flatten(), dim=0).item()
                cos_list.append((k, cos))
            rank_buckets[rank][f'{base}.weight'] = wp.cpu()
            rank_buckets[rank][f'{base}.weight_scale'] = ws.cpu()
            rank_buckets[rank][f'{base}.weight_scale_2'] = ws2.cpu()
            continue

        m_sd = SHARED_OR_DENSE_RE.match(k)
        if m_sd:
            w_f32 = dequant_block_fp8(v.to(device), scales[base].to(device)) if is_fp8 else v.to(device, dtype=torch.float32)
            wp, ws, ws2 = quantize_nvfp4(w_f32, midpoints)
            if verify and len(cos_list) < 4:
                lut = E2M1_LUT.to(device)
                rec = dequant_nvfp4_check(wp, ws, ws2, lut)
                cos = torch.nn.functional.cosine_similarity(w_f32.flatten(), rec.flatten(), dim=0).item()
                cos_list.append((k, cos))
            common_bucket[f'{base}.weight'] = wp.cpu()
            common_bucket[f'{base}.weight_scale'] = ws.cpu()
            common_bucket[f'{base}.weight_scale_2'] = ws2.cpu()
            continue

        # Everything else (attention, indexer, router, norms, hc, embed, lm_head, dfly):
        # Dequantize any block-FP8 (e.g. o_proj, linear_gate, dfly_layers) to BF16 so TensorFold loads directly.
        if is_fp8:
            w_bf16 = dequant_block_fp8(v.to(device), scales[base].to(device)).to(torch.bfloat16).cpu()
            common_bucket[k] = w_bf16
        else:
            common_bucket[k] = v

    stem = shard_name.replace('.safetensors', '')
    written = {'common': {}, 'ranks': {str(r): {} for r in range(4)}}
    if common_bucket:
        fname = f'common-{stem}.safetensors'
        save_file(common_bucket, str(out_dir / 'common' / fname))
        for tk in common_bucket:
            written['common'][tk] = fname
    for r in range(4):
        if rank_buckets[r]:
            fname = f'rank{r}-{stem}.safetensors'
            save_file(rank_buckets[r], str(out_dir / f'rank{r}' / fname))
            for tk in rank_buckets[r]:
                written['ranks'][str(r)][tk] = fname
    return written, cos_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path.home() / 'hy4-770b/Hy4-NVFP4-sharded'))
    ap.add_argument('--tmp', default='/tmp/hy4_dl')
    ap.add_argument('--shards', default='', help='Comma-separated 1-based shard indices for testing, empty=all 1..245')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--dl-chunks', type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    tmp_dir = Path(args.tmp)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'common').mkdir(parents=True, exist_ok=True)
    for r in range(4):
        (out_dir / f'rank{r}').mkdir(parents=True, exist_ok=True)

    # Download metadata files into out_dir/meta
    meta_dir = out_dir / 'meta'
    meta_dir.mkdir(exist_ok=True)
    for mf in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        dst = meta_dir / mf
        if not dst.exists():
            download_file_parallel(f'{BASE_URL}/{mf}', dst, chunks=1)
    cfg_path = meta_dir / 'config.json'
    cfg = json.loads(cfg_path.read_text())
    qcfg = {'quant_method': 'modelopt', 'quant_algo': 'NVFP4', 'group_size': 16}
    cfg['quantization_config'] = qcfg
    if isinstance(cfg.get('text_config'), dict):
        cfg['text_config']['quantization_config'] = qcfg
    cfg_path.write_text(json.dumps(cfg, indent=2))

    prog_path = out_dir / 'progress.json'
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {'done': {}, 'common_map': {}, 'rank_maps': {str(r): {} for r in range(4)}}

    if args.shards:
        indices = []
        for part in args.shards.split(','):
            if '-' in part:
                a, b = part.split('-', 1)
                indices.extend(range(int(a), int(b) + 1))
            else:
                indices.append(int(part))
    else:
        indices = list(range(1, 246))

    todo = [f'model-{i:05d}-of-00245.safetensors' for i in indices if f'model-{i:05d}-of-00245.safetensors' not in prog['done']]
    print(f'Total requested: {len(indices)}, remaining: {len(todo)}', flush=True)
    if not todo:
        return

    device = torch.device('cuda')
    midpoints = E2M1_MIDPOINTS.to(device)
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as dl_pool:
        # Prefetch first 2 shards
        futures = {}
        for s in todo[:2]:
            futures[s] = dl_pool.submit(download_file_parallel, f'{BASE_URL}/{s}', tmp_dir / s, args.dl_chunks)

        for idx, s in enumerate(todo):
            t0 = time.time()
            # Submit prefetch for idx + 2
            if idx + 2 < len(todo):
                nxt = todo[idx + 2]
                futures[nxt] = dl_pool.submit(download_file_parallel, f'{BASE_URL}/{nxt}', tmp_dir / nxt, args.dl_chunks)

            shard_path = futures.pop(s).result()
            t_dl = time.time() - t0
            t1 = time.time()
            written, cos_list = process_shard(s, shard_path, out_dir, device, midpoints, verify=args.verify)
            t_q = time.time() - t1
            shard_path.unlink(missing_ok=True)

            prog['common_map'].update(written['common'])
            for r in range(4):
                prog['rank_maps'][str(r)].update(written['ranks'][str(r)])
            prog['done'][s] = True
            prog_path.write_text(json.dumps(prog))

            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta_m = (len(todo) - idx - 1) / rate / 60.0 if rate > 0 else 0.0
            print(f'[{idx+1}/{len(todo)}] {s}: wait={t_dl:.1f}s quant={t_q:.1f}s eta={eta_m:.1f}m', flush=True)
            if cos_list:
                for ck, cv in cos_list:
                    print(f'  verify {ck}: cos_sim={cv:.6f}', flush=True)

    # Write per-rank model.safetensors.index.json when all shards complete (or always update)
    for r in range(4):
        wmap = {}
        wmap.update(prog['common_map'])
        wmap.update(prog['rank_maps'][str(r)])
        idx_obj = {'metadata': {'format': 'nvfp4-rank-sharded', 'rank': r, 'world': 4}, 'weight_map': wmap}
        (out_dir / f'rank{r}' / 'model.safetensors.index.json').write_text(json.dumps(idx_obj, indent=2))
    print('Done!', flush=True)


if __name__ == '__main__':
    main()
