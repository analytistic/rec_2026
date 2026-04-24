from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

SEQ_TIMESTAMP_FEATS = {
    'action_seq': {28},
    'item_seq': {29},
    'content_seq': {41},
}


def prepare_data(data):
    """
    Prepare the dataset for training.
    return:
        itemid_info: {itemid_dict, map_range}
            itemid_dict: {item_id: {map, count, prob}}
        userid_info: {userid_dict, map_range}
            userid_dict: {user_id: {map, count, prob}}
        feat_info: {seq_feature_dict, item_feature_dict, user_feature_dict}

        item_feature_dict / user_feature_dict (by type):
            int_value / int_array:
                type, range: [min, max], values: {value: {map, count, prob}},
                count, prob, coverage, map_range
            float_array:
                type, array_len (固定长度 or None), count, prob, coverage
            int_array_and_float_array:
                type, array_len (固定长度 or -1→None), count, prob, coverage, map_range,
                values: {int_value: {map, count, prob, float_stats: {min, max, mean}}}

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
    mixed_float_sums = defaultdict(lambda: defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'sum': 0.0, 'count': 0}))
    data = data.sort('item_id')
    idx = 0

    for idx, record in tqdm(enumerate(data), total=len(data)):
        # Item ID dict
        item_id = record['item_id']
        if item_id not in itemid_dict:
            itemid_dict[item_id] = {
                'count': 0,
                'map': len(itemid_dict) + 1
            }
        itemid_dict[item_id]['count'] += 1

        # User ID dict
        user_id = record['user_id']
        if user_id not in userid_dict:
            userid_dict[user_id] = {
                'count': 0,
                'map': len(userid_dict) + 1
            }
        userid_dict[user_id]['count'] += 1

        # Item feature dict
        for f in record.get('item_feature') or []:
            fid = f['feature_id']
            if fid not in item_feature_dict:
                item_feature_dict[fid] = {
                    'type': f['feature_value_type'],
                    'range': [float('inf'), float('-inf')],
                    'values': {},
                    'count': 0,
                    'users_with_feature': set(),
                    'map_range': 0,
                }

            ftype = f['feature_value_type']
            item_feature_dict[fid]['count'] += 1
            item_feature_dict[fid]['users_with_feature'].add(item_id)

            # handle int_value and int_array
            values_to_add = []
            if ftype == 'int_value' and f.get('int_value') is not None:
                values_to_add.append(int(f['int_value']))
            elif ftype == 'int_array' and f.get('int_array') is not None:
                values_to_add.extend(f['int_array'])

            for v in values_to_add:
                if v < item_feature_dict[fid]['range'][0]: item_feature_dict[fid]['range'][0] = v
                if v > item_feature_dict[fid]['range'][1]: item_feature_dict[fid]['range'][1] = v

                if v not in item_feature_dict[fid]['values']:
                    item_feature_dict[fid]['values'][v] = {'count': 0, 'map': item_feature_dict[fid]['map_range'] + 1}
                    item_feature_dict[fid]['map_range'] += 1
                item_feature_dict[fid]['values'][v]['count'] += 1

        # User feature dict
        for f in record.get('user_feature') or []:
            fid = f['feature_id']
            if fid not in user_feature_dict:
                entry = {
                    'type': f['feature_value_type'],
                    'count': 0,
                    'users_with_feature': set(),
                }
                ftype = f['feature_value_type']
                if ftype in ('int_value', 'int_array'):
                    entry['range'] = [float('inf'), float('-inf')]
                    entry['values'] = {}
                    entry['map_range'] = 0
                elif ftype in ('float_array', 'int_array_and_float_array'):
                    entry['array_len'] = None
                    entry['map_range'] = 0
                user_feature_dict[fid] = entry

            ftype = f['feature_value_type']
            user_feature_dict[fid]['count'] += 1
            user_feature_dict[fid]['users_with_feature'].add(user_id)

            values_to_add = []
            if ftype == 'int_value' and f.get('int_value') is not None:
                values_to_add.append(int(f['int_value']))
            elif ftype == 'int_array' and f.get('int_array') is not None:
                values_to_add.extend(f['int_array'])
            elif ftype == 'float_array' and f.get('float_array') is not None:
                arr_len = len(f['float_array'])
                if user_feature_dict[fid]['array_len'] is None:
                    user_feature_dict[fid]['array_len'] = arr_len
                elif user_feature_dict[fid]['array_len'] >= 0 and user_feature_dict[fid]['array_len'] != arr_len:
                    user_feature_dict[fid]['array_len'] = -1
            elif ftype == 'int_array_and_float_array':
                ia = f.get('int_array') or []
                fa = f.get('float_array') or []
                arr_len = len(ia)
                if user_feature_dict[fid]['array_len'] is None:
                    user_feature_dict[fid]['array_len'] = arr_len
                elif user_feature_dict[fid]['array_len'] >= 0 and user_feature_dict[fid]['array_len'] != arr_len:
                    user_feature_dict[fid]['array_len'] = -1
                for i, iv in enumerate(ia):
                    mixed_int_counts[fid][iv] += 1
                    if i < len(fa):
                        stats = mixed_float_sums[fid][iv]
                        fv = fa[i]
                        if fv < stats['min']: stats['min'] = fv
                        if fv > stats['max']: stats['max'] = fv
                        stats['sum'] += fv
                        stats['count'] += 1

            for v in values_to_add:
                if v < user_feature_dict[fid]['range'][0]: user_feature_dict[fid]['range'][0] = v
                if v > user_feature_dict[fid]['range'][1]: user_feature_dict[fid]['range'][1] = v

                if v not in user_feature_dict[fid]['values']:
                    user_feature_dict[fid]['values'][v] = {'count': 0, 'map': user_feature_dict[fid]['map_range'] + 1}
                    user_feature_dict[fid]['map_range'] += 1
                user_feature_dict[fid]['values'][v]['count'] += 1

        # Seq feature dict
        for seq_name, seq_list in (record.get('seq_feature') or {}).items():
            if seq_name not in seq_feature_dict:
                seq_feature_dict[seq_name] = {}
                
            for sf in seq_list or []:
                sfid = sf['feature_id']
                is_ts = sfid in SEQ_TIMESTAMP_FEATS.get(seq_name, set())
                if sfid not in seq_feature_dict[seq_name]:
                    entry = {
                        'timestamp': is_ts,
                        'count': 0,
                        'users_with_feature': set()
                    }
                    if not is_ts:
                        entry['range'] = [float('inf'), float('-inf')]
                        entry['array_id'] = {}
                        entry['map_range'] = 0
                    seq_feature_dict[seq_name][sfid] = entry

                seq_feature_dict[seq_name][sfid]['count'] += 1
                seq_feature_dict[seq_name][sfid]['users_with_feature'].add(user_id)
                if is_ts:
                    continue
                arr = sf.get('int_array') or []
                for v in arr:
                    if v < seq_feature_dict[seq_name][sfid]['range'][0]: seq_feature_dict[seq_name][sfid]['range'][0] = v
                    if v > seq_feature_dict[seq_name][sfid]['range'][1]: seq_feature_dict[seq_name][sfid]['range'][1] = v

                    if v not in seq_feature_dict[seq_name][sfid]['array_id']:
                        seq_feature_dict[seq_name][sfid]['array_id'][v] = {'count': 0, 'map': seq_feature_dict[seq_name][sfid]['map_range'] + 1}
                        seq_feature_dict[seq_name][sfid]['map_range'] += 1
                    seq_feature_dict[seq_name][sfid]['array_id'][v]['count'] += 1

    total_records = idx + 1
    total_users = len(userid_dict)
    total_items = len(itemid_dict)

    # Post-process: probabilities and cleanup
    for _, d in itemid_dict.items(): d['prob'] = d['count'] / total_records
    itemid_map_range = len(itemid_dict)

    for _, d in userid_dict.items(): d['prob'] = d['count'] / total_records
    userid_map_range = len(userid_dict)

    for fid, d in item_feature_dict.items():
        d['prob'] = d['count'] / total_records
        d['coverage'] = len(d['users_with_feature']) / total_items
        del d['users_with_feature']
        if d['range'][0] == float('inf'): d.pop('range', None)
        for v, vd in d['values'].items(): vd['prob'] = vd['count'] / (d['count'] or 1)

    for fid, d in user_feature_dict.items():
        d['prob'] = d['count'] / total_records
        d['coverage'] = len(d['users_with_feature']) / total_users
        del d['users_with_feature']
        if d.get('array_len') == -1:
            d['array_len'] = None
        if d['type'] == 'int_array_and_float_array':
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
        elif d['type'] in ('int_value', 'int_array'):
            if d['range'][0] == float('inf'): d.pop('range', None)
            for v, vd in d['values'].items(): vd['prob'] = vd['count'] / (d['count'] or 1)

    for seq_name, sfs in seq_feature_dict.items():
        for sfid, d in sfs.items():
            d['prob'] = d['count'] / total_records
            d['coverage'] = len(d['users_with_feature']) / total_users
            del d['users_with_feature']
            if not d['timestamp']:
                if d['range'][0] == float('inf'): d.pop('range', None)
                total_elements = sum(vd['count'] for vd in d['array_id'].values())
                for v, vd in d['array_id'].items(): vd['prob'] = vd['count'] / (total_elements or 1)

    feat_info = {
        'seq_feature_dict': seq_feature_dict,
        'item_feature_dict': item_feature_dict,
        'user_feature_dict': user_feature_dict,
    }
    itemid_info = {
        'itemid_dict': itemid_dict,
        'map_range': itemid_map_range
    }
    userid_info = {
        'userid_dict': userid_dict,
        'map_range': userid_map_range
    }
    return itemid_info, userid_info, feat_info


if __name__ == "__main__":
    import json
    data_files = 'data/sample_data.parquet'
    data  = load_dataset("parquet", data_files=data_files, split='train')
    itemid_info, userid_info, feat_info = prepare_data(data)
    with open('data/itemid_info.json', 'w', encoding='utf-8') as f:
        json.dump(itemid_info, f, ensure_ascii=False, indent=2)
    with open('data/userid_info.json', 'w', encoding='utf-8') as f:
        json.dump(userid_info, f, ensure_ascii=False, indent=2)
    with open('data/feat_info.json', 'w', encoding='utf-8') as f:
        json.dump(feat_info, f, ensure_ascii=False, indent=2)