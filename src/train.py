"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python -m src.train --config config.yaml [--data_dir ...]

Environment variables (take precedence over CLI flags and config file):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
    TRAIN_TF_EVENTS_PATH  TensorBoard events directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import torch

from .utils import set_seed, EarlyStopping, create_logger
from .dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from .model import PCVRHyFormer
from .trainer import PCVRHyFormerRankingTrainer



_DTYPE_MAP = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args_and_config() -> Dict[str, Any]:
    """Parse --config + minimal CLI overrides, merge into single config dict.

    Precedence (higher wins):  env vars > CLI args > config file
    """
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file')

    # CLI overrides (all optional; values in config.yaml are the defaults).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')
    parser.add_argument('--valid_data_dir', type=str, default=None,
                        help='Separate validation data directory')
    parser.add_argument('--ns_groups_json', type=str, default=None,
                        help='Path to NS-groups JSON')
    parser.add_argument('--device', type=str, default=None,
                        help='Training device, e.g. cuda or cpu')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of DataLoader workers')

    cli_args = parser.parse_args()

    # 1. Load yaml as base config.
    with open(cli_args.config) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    logging.info(f"Loaded config from {cli_args.config}")

    # 2. CLI overrides yaml (only non-None values).
    for key, val in vars(cli_args).items():
        if key != 'config' and val is not None:
            cfg[key] = val

    # 3. Environment variables take precedence.
    if 'TRAIN_DATA_PATH' in os.environ:
        cfg['data_dir'] = os.environ['TRAIN_DATA_PATH']
    if 'TRAIN_CKPT_PATH' in os.environ:
        cfg['ckpt_dir'] = os.environ['TRAIN_CKPT_PATH']
    if 'TRAIN_LOG_PATH' in os.environ:
        cfg['log_dir'] = os.environ['TRAIN_LOG_PATH']
    cfg['tf_events_dir'] = os.environ.get(
        'TRAIN_TF_EVENTS_PATH',
        os.path.join(cfg.get('log_dir', 'output'), 'tf_events'))

    # Device: if not set anywhere, auto-detect.
    if 'device' not in cfg or cfg['device'] is None:
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    return cfg


def main() -> None:
    cfg = parse_args_and_config()

    # Create output directories.
    Path(cfg['ckpt_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfg['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfg['tf_events_dir']).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(cfg['seed'])
    create_logger(os.path.join(cfg['log_dir'], 'train.log'))
    logging.info(f"Config: {cfg}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(cfg['tf_events_dir'])

    # ---- Data loading ----
    if cfg.get('schema_path'):
        schema_path = cfg['schema_path']
    else:
        schema_path = os.path.join(cfg['data_dir'], 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    sml = cfg.get('seq_max_lens', '')
    if sml:
        for pair in sml.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=cfg['data_dir'],
        schema_path=schema_path,
        batch_size=cfg['batch_size'],
        valid_ratio=cfg.get('valid_ratio', 0.1),
        train_ratio=cfg.get('train_ratio', 1.0),
        num_workers=cfg.get('num_workers', 16),
        buffer_batches=cfg.get('buffer_batches', 20),
        seed=cfg['seed'],
        seq_max_lens=seq_max_lens,
        valid_data_dir=cfg.get('valid_data_dir'),
        add_seq_time_attrs=cfg.get('add_seq_time_attrs', True),
        seq_mask_ratio=cfg.get('seq_mask_ratio', 0.0),
    )

    # ---- NS groups ----
    ns_groups_json = cfg.get('ns_groups_json', '')
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Hash embedding config: convert fid → fid_idx for item features ----
    raw_hash = cfg.get('hash_embedding', {})
    item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
    item_hash_config = {}
    for fid, hcfg in raw_hash.items():
        if fid not in item_fid_to_idx:
            logging.warning(f"hash_embedding fid {fid} not found in item_int_schema, skipping")
            continue
        item_hash_config[item_fid_to_idx[fid]] = hcfg
    if item_hash_config:
        logging.info(f"Item hash config: {item_hash_config}")

    # ---- Hash embedding config for user features ----
    user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
    user_hash_config = {}
    for fid, hcfg in raw_hash.items():
        if fid not in user_fid_to_idx:
            continue
        user_hash_config[user_fid_to_idx[fid]] = hcfg
    if user_hash_config:
        logging.info(f"User hash config: {user_hash_config}")

    # ---- Seq hash embedding config: convert fid → sideinfo position index ----
    raw_seq_hash = cfg.get('seq_hash_embedding', {})
    seq_hash_config = {}
    for domain, fids_cfg in raw_seq_hash.items():
        if domain not in pcvr_dataset.seq_domains:
            logging.warning(f"seq_hash_embedding: unknown domain {domain}, skipping")
            continue
        sideinfo = pcvr_dataset.sideinfo_fids.get(domain, [])
        domain_cfg = {}
        for fid, hcfg in fids_cfg.items():
            if fid not in sideinfo:
                logging.warning(f"seq_hash_embedding: fid {fid} not in {domain} sideinfo, skipping")
                continue
            pos = sideinfo.index(fid)
            domain_cfg[pos] = hcfg
        if domain_cfg:
            logging.info(f"Seq hash config for {domain}: {domain_cfg}")
            seq_hash_config[domain] = domain_cfg
    if seq_hash_config:
        logging.info(f"Seq hash config: {seq_hash_config}")

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)
    paired_feature_specs = build_feature_specs(
        pcvr_dataset.paired_int_schema, pcvr_dataset.paired_int_vocab_sizes) if hasattr(pcvr_dataset, 'paired_int_schema') else []
    paired_fids = [fid for fid, _, _ in pcvr_dataset.paired_int_schema.entries] if hasattr(pcvr_dataset, 'paired_int_schema') else []

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "item_hash_config": item_hash_config,
        "user_hash_config": user_hash_config,
        "paired_feature_specs": paired_feature_specs,
        "paired_fids": paired_fids,
        "seq_hash_config": seq_hash_config,
        "d_model": cfg['d_model'],
        "emb_dim": cfg['emb_dim'],
        "num_queries": cfg['num_queries'],
        "num_hyformer_blocks": cfg['num_hyformer_blocks'],
        "num_heads": cfg['num_heads'],
        "seq_encoder_type": cfg['seq_encoder_type'],
        "hidden_mult": cfg['hidden_mult'],
        "embed_dropout_rate": cfg['embed_dropout_rate'],
        "seq_id_dropout_rate": cfg['seq_id_dropout_rate'],
        "hidden_dropout_rate": cfg['hidden_dropout_rate'],
        "norm_type": cfg['norm_type'],
        "seq_top_k": cfg['seq_top_k'],
        "seq_causal": cfg['seq_causal'],
        "action_num": cfg['action_num'],
        "num_time_buckets": NUM_TIME_BUCKETS if cfg.get('use_time_buckets', True) else 0,
        "rank_mixer_mode": cfg['rank_mixer_mode'],
        "ffn_name": cfg['ffn_name'],
        "ffn_config": cfg['ffn_config'],
        "mixer_type": cfg.get('mixer_type', 'rank'),
        "use_rope": cfg['use_rope'],
        "rope_base": cfg['rope_base'],
        "emb_skip_threshold": cfg['emb_skip_threshold'],
        "seq_id_threshold": cfg['seq_id_threshold'],
        "ns_tokenizer_type": cfg['ns_tokenizer_type'],
        "user_ns_tokens": cfg['user_ns_tokens'],
        "item_ns_tokens": cfg['item_ns_tokens'],
        "dense_dtype": _DTYPE_MAP[cfg['dense_dtype']],
        "sparse_dtype": _DTYPE_MAP[cfg['sparse_dtype']],
        "fourier_seq": cfg['fourier_seq'],
        "fourier_ns": cfg['fourier_ns'],
        "use_row_time_ns": cfg['use_row_time_ns'],
        "use_domain_emb": cfg.get('use_domain_emb', False),
        "seq_proj_type": cfg['seq_proj_type'],
        "seq_ffn_name": cfg['seq_ffn_name'],
        "seq_ffn_config": cfg['seq_ffn_config'],
    }

    logging.info(f"Dtype config: dense={cfg['dense_dtype']}, sparse={cfg['sparse_dtype']}")

    model = PCVRHyFormer(**model_args).to(cfg['device'])

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = cfg['num_queries'] * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={cfg['d_model']}, rank_mixer_mode={cfg['rank_mixer_mode']}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Vocab info ----
    def _print_vocab_info():
        lines = []
        for name, tokenizer, schema in [
            ("User", model.user_ns_tokenizer, pcvr_dataset.user_int_schema),
            ("Item", model.item_ns_tokenizer, pcvr_dataset.item_int_schema),
        ]:
            lines.append(f"\n  [{name} features]")
            for fid_idx, (vs, offset, length) in enumerate(tokenizer.feature_specs):
                fid = schema.entries[fid_idx][0]
                eidx = tokenizer._emb_index[fid_idx]
                if fid_idx in tokenizer._hash_multi:
                    hcfg = tokenizer._hash_multi[fid_idx]
                    lines.append(f"    fid={fid}: vocab={vs:>9,}  HASH(H={hcfg['H']}, k={hcfg['k']})")
                elif eidx == -1:
                    thr = cfg.get('emb_skip_threshold', '?')
                    lines.append(f"    fid={fid}: vocab={vs:>9,}  SKIP(>{thr})")
                else:
                    lines.append(f"    fid={fid}: vocab={vs:>9,}  embed")
        lines.append("\n  [Seq features]")
        for domain in model.seq_domains:
            vs_list = pcvr_dataset.seq_domain_vocab_sizes[domain]
            sideinfo = pcvr_dataset.sideinfo_fids.get(domain, [])
            lines.append(f"    {domain}:")
            for i, (fid, vs) in enumerate(zip(sideinfo, vs_list)):
                eidx = model._seq_emb_index[domain][i]
                is_hash = model._seq_is_hash.get(domain, [False] * len(vs_list))[i] if i < len(model._seq_is_hash.get(domain, [])) else False
                if is_hash:
                    dhc = model.seq_hash_config.get(domain, {}).get(i, {})
                    lines.append(f"      fid={fid}: vocab={vs:>9,}  HASH(H={dhc.get('H', '?')}, k={dhc.get('k', '?')})")
                elif eidx == -1:
                    lines.append(f"      fid={fid}: vocab={vs:>9,}  SKIP(>1M)")
                else:
                    lines.append(f"      fid={fid}: vocab={vs:>9,}  embed")
        return '\n'.join(lines)
    logging.info(f"Vocab info:\n{_print_vocab_info()}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(cfg['ckpt_dir'], "placeholder", "model.pt"),
        patience=cfg['patience'],
        label='model',
    )

    ckpt_params = {
        "layer": cfg['num_hyformer_blocks'],
        "head": cfg['num_heads'],
        "hidden": cfg['d_model'],
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=cfg['lr'],
        num_epochs=cfg['num_epochs'],
        device=cfg['device'],
        save_dir=cfg['ckpt_dir'],
        early_stopping=early_stopping,
        loss_type=cfg['loss_type'],
        focal_alpha=cfg['focal_alpha'],
        focal_gamma=cfg['focal_gamma'],
        focal_weight=cfg.get('focal_weight', 0.0),
        focal_start_epoch=cfg.get('focal_start_epoch', 0),
        sparse_lr=cfg['sparse_lr'],
        sparse_weight_decay=cfg['sparse_weight_decay'],
        reinit_sparse_after_epoch=cfg['reinit_sparse_after_epoch'],
        reinit_cardinality_threshold=cfg['reinit_cardinality_threshold'],
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=ns_groups_json if ns_groups_json and os.path.exists(ns_groups_json) else None,
        eval_every_n_steps=cfg.get('eval_every_n_steps', 0),
        train_config=cfg,
        use_amp=cfg['use_amp'],
        log_step=cfg['log_step'],
        accumulation_steps=cfg['accumulation_steps'],
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
