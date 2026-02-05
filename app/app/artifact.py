import numpy as np
from .baseline import mad

def detect_artifact_blocks_strict(t, y,
                                 slope_pos=5, slope_neg=8,
                                 amp_pos=3, amp_neg=12,
                                 min_len=10, merge_gap=2):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = float(np.median(np.diff(t)))
    d  = np.diff(y) / (dt + 1e-12)
    z  = (d - np.median(d)) / (mad(d) + 1e-9)
    m  = np.median(y)
    s  = mad(y) + 1e-12

    bad_pos = np.where((z > slope_pos) | (y[1:] > m + amp_pos * s))[0]
    bad_neg = np.where((z < -slope_neg) | (y[:-1] < m - amp_neg * s))[0]

    def to_blocks(idxs):
        if len(idxs) == 0:
            return []
        out = []
        s0 = idxs[0]
        p = idxs[0]
        for i in idxs[1:]:
            if i == p + 1:
                p = i
            else:
                if (p - s0 + 1) >= min_len:
                    out.append((t[s0], t[p]))
                s0 = i
                p = i
        if (p - s0 + 1) >= min_len:
            out.append((t[s0], t[p]))
        return out

    blocks = to_blocks(np.sort(np.concatenate([bad_pos, bad_neg])))
    merged = []
    for s_b, e_b in blocks:
        if not merged:
            merged.append([s_b, e_b])
        elif s_b - merged[-1][1] <= merge_gap * dt:
            merged[-1][1] = e_b
        else:
            merged.append([s_b, e_b])
    return [(a, b) for a, b in merged]

def extend_artifact_until_calm_light(t, y, blocks,
                                    calm_window=1.0,
                                    calm_sigma_k=1.5,
                                    mean_k=2.0,
                                    max_extend=8.0,
                                    post_silence=0.5,
                                    extend_before=0.2,
                                    extend_after=0.3):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = float(np.median(np.diff(t)))
    win_n = int(round(calm_window / (dt + 1e-12)))
    win_n = max(5, win_n)
    global_mad = mad(y) + 1e-12
    global_med = np.median(y)
    extended = []
    for s_b, e_b in blocks:
        end_idx = int(np.searchsorted(t, e_b))
        max_idx = int(np.searchsorted(t, e_b + max_extend))
        while end_idx + win_n < len(y) and end_idx < max_idx:
            local = y[end_idx:end_idx + win_n]
            local_mad = mad(local)
            local_mean = float(np.mean(local))
            if (local_mad <= calm_sigma_k * global_mad) and (abs(local_mean - global_med) < mean_k * global_mad):
                break
            end_idx += max(1, win_n // 2)
        s_ext = max(t[0], s_b - extend_before)
        e_ext = min(t[-1], t[min(end_idx, len(t) - 1)] + post_silence + extend_after)
        extended.append((s_ext, e_ext))
    return extended

def merge_blocks(all_blocks, merge_gap=0.5):
    merged = []
    all_list = sorted([b for blocks in all_blocks for b in blocks], key=lambda x: x[0])
    for s_b, e_b in all_list:
        if not merged:
            merged.append([s_b, e_b])
        elif s_b <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], e_b)
        else:
            merged.append([s_b, e_b])
    return [(s_b, e_b) for s_b, e_b in merged]

def apply_blocks(t, y, blocks):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.ones_like(y, bool)
    for s_b, e_b in blocks:
        mask[(t >= s_b) & (t <= e_b)] = False
    return t[mask], y[mask]
