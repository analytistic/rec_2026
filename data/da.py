import re
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

SEQ_TIMESTAMP_FEATS = {
    'domain_a': {39},
    'domain_b': {67},
    'domain_c': {27},
    'domain_d': {26},
}

BASE_COLS = {'user_id', 'item_id', 'label_type', 'label_time', 'timestamp'}


def _parse_columns(data):
    """
    Parse flat column names into groups and determine overlapping fids.
      user_int_cols, user_dense_cols, item_int_cols, seq_cols, mixed_user_fids
    """
    user_int = []
    user_dense = []
    item_int = []
    seq = defaultdict(list)

    for col in data.column_names:
        if col in BASE_COLS:
            continue
        if col.startswith('user_int_feats_'):
            user_int.append((col, int(col.split('_')[-1])))
        elif col.startswith('user_dense_feats_'):
            user_dense.append((col, int(col.split('_')[-1])))
        elif col.startswith('item_int_feats_'):
            item_int.append((col, int(col.split('_')[-1])))
        else:
            m = re.match(r'(\w+)_seq_(\d+)', col)
            if m:
                seq[m.group(1)].append((col, int(m.group(2))))

    user_int_fids = {fid for _, fid in user_int}
    user_dense_fids = {fid for _, fid in user_dense}
    mixed_user_fids = user_int_fids & user_dense_fids

    return user_int, user_dense, item_int, dict(seq), mixed_user_fids


def prepare_data(data):
    """
    Prepare the flattened dataset for training.
    return:
        itemid_info: {itemid_dict, map_range}
        userid_info: {userid_dict, map_range}
        feat_info: {seq_feature_dict, item_feature_dict, user_feature_dict}

    item_feature_dict / user_feature_dict (by type):
        int_value:
            type, range: [min, max], values: {value: {map, count, prob}},
            count, prob, coverage, map_range
        int_array:
            type, range: [min, max], values: {value: {map, count, prob}},
            count, prob, coverage, map_range, max_len
        float_array:
            type, array_len, count, prob, coverage

    seq_feature_dict:
        timestamp=True: timestamp, count, prob, coverage
        timestamp=False: range, array_id: {id: {map, count, prob}},
            count, prob, coverage, map_range
    """
    itemid_dict = {}
    userid_dict = {}
    user_feature_dict = {}
    item_feature_dict = {}
    seq_feature_dict = {}
    mixed_int_counts = defaultdict(lambda: defaultdict(int))
    mixed_float_sums = defaultdict(lambda: defaultdict(
        lambda: {'min': float('inf'), 'max': float('-inf'), 'sum': 0.0, 'count': 0}
    ))

    user_int_cols, user_dense_cols, item_int_cols, seq_cols, mixed_user_fids = _parse_columns(data)

    # Build fid → column name lookups for user features
    user_int_map = {fid: col for col, fid in user_int_cols}
    user_dense_map = {fid: col for col, fid in user_dense_cols}
    pure_user_int = [(c, f) for c, f in user_int_cols if f not in mixed_user_fids]
    pure_user_dense = [(c, f) for c, f in user_dense_cols if f not in mixed_user_fids]

    data = data.sort('item_id')
    idx = 0

    for idx, record in tqdm(enumerate(data), total=len(data)):
        item_id = record['item_id']
        user_id = record['user_id']

        # Item ID dict
        if item_id not in itemid_dict:
            itemid_dict[item_id] = {'count': 0, 'map': len(itemid_dict) + 1}
        itemid_dict[item_id]['count'] += 1

        # User ID dict
        if user_id not in userid_dict:
            userid_dict[user_id] = {'count': 0, 'map': len(userid_dict) + 1}
        userid_dict[user_id]['count'] += 1

        # --- Item features (int_value or int_array) ---
        for col, fid in item_int_cols:
            val = record[col]
            if val is None:
                continue

            ftype = 'int_array' if isinstance(val, list) else 'int_value'

            if fid not in item_feature_dict:
                entry = {
                    'type': ftype,
                    'range': [float('inf'), float('-inf')],
                    'values': {},
                    'count': 0,
                    'users_with_feature': set(),
                    'map_range': 0,
                }
                if ftype == 'int_array':
                    entry['max_len'] = 0
                item_feature_dict[fid] = entry

            d = item_feature_dict[fid]
            d['count'] += 1
            d['users_with_feature'].add(item_id)

            values = [int(val)] if ftype == 'int_value' else val
            if ftype == 'int_array' and len(val) > d['max_len']:
                d['max_len'] = len(val)

            for v in values:
                if v < d['range'][0]:
                    d['range'][0] = v
                if v > d['range'][1]:
                    d['range'][1] = v
                if v not in d['values']:
                    d['values'][v] = {'count': 0, 'map': d['map_range'] + 1}
                    d['map_range'] += 1
                d['values'][v]['count'] += 1

        # --- User features: mixed int_array + float_array (same fid) ---
        for fid in mixed_user_fids:
            int_val = record[user_int_map[fid]]
            dense_val = record[user_dense_map[fid]]
            if int_val is None and dense_val is None:
                continue

            if fid not in user_feature_dict:
                user_feature_dict[fid] = {
                    'type': 'int_array_and_float_array',
                    'array_len': None,
                    'max_len': 0,
                    'count': 0,
                    'users_with_feature': set(),
                    'map_range': 0,
                }

            d = user_feature_dict[fid]
            d['count'] += 1
            d['users_with_feature'].add(user_id)

            if dense_val is not None:
                arr_len = len(dense_val)
                if d['array_len'] is None:
                    d['array_len'] = arr_len
                elif d['array_len'] >= 0 and d['array_len'] != arr_len:
                    d['array_len'] = -1

            if int_val is not None:
                if len(int_val) > d['max_len']:
                    d['max_len'] = len(int_val)
                for i, iv in enumerate(int_val):
                    mixed_int_counts[fid][iv] += 1
                    if dense_val is not None and i < len(dense_val):
                        fv = dense_val[i]
                        stats = mixed_float_sums[fid][iv]
                        if fv < stats['min']: stats['min'] = fv
                        if fv > stats['max']: stats['max'] = fv
                        stats['sum'] += fv
                        stats['count'] += 1

        # --- User int features (pure int_value or int_array) ---
        for col, fid in pure_user_int:
            val = record[col]
            if val is None:
                continue

            ftype = 'int_array' if isinstance(val, list) else 'int_value'

            if fid not in user_feature_dict:
                entry = {
                    'type': ftype,
                    'count': 0,
                    'users_with_feature': set(),
                    'range': [float('inf'), float('-inf')],
                    'values': {},
                    'map_range': 0,
                }
                if ftype == 'int_array':
                    entry['max_len'] = 0
                user_feature_dict[fid] = entry

            d = user_feature_dict[fid]
            d['count'] += 1
            d['users_with_feature'].add(user_id)

            values = [int(val)] if ftype == 'int_value' else val
            if ftype == 'int_array' and len(val) > d['max_len']:
                d['max_len'] = len(val)

            for v in values:
                if v < d['range'][0]:
                    d['range'][0] = v
                if v > d['range'][1]:
                    d['range'][1] = v
                if v not in d['values']:
                    d['values'][v] = {'count': 0, 'map': d['map_range'] + 1}
                    d['map_range'] += 1
                d['values'][v]['count'] += 1

        # --- User dense features (pure float_array) ---
        for col, fid in pure_user_dense:
            val = record[col]
            if val is None:
                continue

            if fid not in user_feature_dict:
                user_feature_dict[fid] = {
                    'type': 'float_array',
                    'array_len': None,
                    'count': 0,
                    'users_with_feature': set(),
                }

            d = user_feature_dict[fid]
            d['count'] += 1
            d['users_with_feature'].add(user_id)

            arr_len = len(val)
            if d['array_len'] is None:
                d['array_len'] = arr_len
            elif d['array_len'] >= 0 and d['array_len'] != arr_len:
                d['array_len'] = -1

        # --- Seq features ---
        for domain, cols in seq_cols.items():
            if domain not in seq_feature_dict:
                seq_feature_dict[domain] = {}

            for col, fid in cols:
                val = record[col]
                if val is None:
                    continue

                is_ts = fid in SEQ_TIMESTAMP_FEATS.get(domain, set())

                if fid not in seq_feature_dict[domain]:
                    entry = {
                        'timestamp': is_ts,
                        'count': 0,
                        'users_with_feature': set(),
                    }
                    if not is_ts:
                        entry['range'] = [float('inf'), float('-inf')]
                        entry['array_id'] = {}
                        entry['map_range'] = 0
                    seq_feature_dict[domain][fid] = entry

                d = seq_feature_dict[domain][fid]
                d['count'] += 1
                d['users_with_feature'].add(user_id)

                if is_ts:
                    continue

                for v in val:
                    if v < d['range'][0]:
                        d['range'][0] = v
                    if v > d['range'][1]:
                        d['range'][1] = v
                    if v not in d['array_id']:
                        d['array_id'][v] = {'count': 0, 'map': d['map_range'] + 1}
                        d['map_range'] += 1
                    d['array_id'][v]['count'] += 1

    total_records = idx + 1
    total_users = len(userid_dict)
    total_items = len(itemid_dict)

    for _, d in itemid_dict.items():
        d['prob'] = d['count'] / total_records
    itemid_map_range = len(itemid_dict)

    for _, d in userid_dict.items():
        d['prob'] = d['count'] / total_records
    userid_map_range = len(userid_dict)

    for fid, d in item_feature_dict.items():
        d['prob'] = d['count'] / total_records
        d['coverage'] = len(d['users_with_feature']) / total_items
        del d['users_with_feature']
        if d['range'][0] == float('inf'):
            d.pop('range', None)
        for v, vd in d['values'].items():
            vd['prob'] = vd['count'] / (d['count'] or 1)

    for fid, d in user_feature_dict.items():
        d['prob'] = d['count'] / total_records
        d['coverage'] = len(d['users_with_feature']) / total_users
        del d['users_with_feature']
        if d['type'] == 'float_array':
            if d.get('array_len') == -1:
                d['array_len'] = None
        elif d['type'] == 'int_array_and_float_array':
            if d.get('array_len') == -1:
                d['array_len'] = None
            int_counts = mixed_int_counts.get(fid, {})
            total_int = sum(int_counts.values())
            values = {}
            for iv, cnt in int_counts.items():
                entry = {'count': cnt, 'prob': cnt / (total_int or 1), 'map': d['map_range'] + len(values) + 1}
                fs = mixed_float_sums[fid].get(iv)
                if fs and fs['count'] > 0:
                    entry['float_stats'] = {
                        'min': fs['min'],
                        'max': fs['max'],
                        'mean': fs['sum'] / fs['count'],
                    }
                values[iv] = entry
            d['values'] = values
            d['map_range'] += len(values)
        else:
            if d['range'][0] == float('inf'):
                d.pop('range', None)
            for v, vd in d['values'].items():
                vd['prob'] = vd['count'] / (d['count'] or 1)

    for seq_name, sfs in seq_feature_dict.items():
        for sfid, d in sfs.items():
            d['prob'] = d['count'] / total_records
            d['coverage'] = len(d['users_with_feature']) / total_users
            del d['users_with_feature']
            if not d['timestamp']:
                if d['range'][0] == float('inf'):
                    d.pop('range', None)
                total_elements = sum(vd['count'] for vd in d['array_id'].values())
                for v, vd in d['array_id'].items():
                    vd['prob'] = vd['count'] / (total_elements or 1)

    feat_info = {
        'seq_feature_dict': seq_feature_dict,
        'item_feature_dict': item_feature_dict,
        'user_feature_dict': user_feature_dict,
    }
    itemid_info = {
        'itemid_dict': itemid_dict,
        'map_range': itemid_map_range,
    }
    userid_info = {
        'userid_dict': userid_dict,
        'map_range': userid_map_range,
    }
    return itemid_info, userid_info, feat_info


if __name__ == "__main__":
    import json
    data_files = 'data/demo_1000.parquet'
    data = load_dataset("parquet", data_files=data_files, split='train')
    itemid_info, userid_info, feat_info = prepare_data(data)
    with open('data/itemid_info.json', 'w', encoding='utf-8') as f:
        json.dump(itemid_info, f, ensure_ascii=False, indent=2)
    with open('data/userid_info.json', 'w', encoding='utf-8') as f:
        json.dump(userid_info, f, ensure_ascii=False, indent=2)
    with open('data/feat_info.json', 'w', encoding='utf-8') as f:
        json.dump(feat_info, f, ensure_ascii=False, indent=2)
