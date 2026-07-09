from __future__ import annotations
import re, zipfile, time
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np

import daily_ladder_engine as eng


# ---- embedded support_debug module (single-file Streamlit Cloud safety) ----
from pathlib import Path
import pandas as pd
import numpy as np

COMPONENT_COLS = [
    'score_stream_core_usable',
    'score_seed_trait_usable',
    'score_stream_seed_trait_usable',
    'score_cadence',
    'score_member_role',
    'score_exact_stream_core_member',
]
MAJOR_COMPONENTS = [
    'score_stream_core_usable',
    'score_seed_trait_usable',
    'score_stream_seed_trait_usable',
    'score_cadence',
    'score_exact_stream_core_member',
]

KEY_TRAIT_COLS = ['seed_parity','seed_highlow','seed_structure','seed_parity_count','seed_sum_bucket']

def _num(s):
    return pd.to_numeric(s, errors='coerce').fillna(0)

def profile_inventory(profile_dir) -> pd.DataFrame:
    rows=[]
    p=Path(profile_dir)
    search_dirs=[]
    for d in [p, Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent/'profiles']:
        try:
            rd=d.resolve()
        except Exception:
            rd=d
        if d.exists() and rd not in search_dirs:
            search_dirs.append(rd)
    files=[]
    seen=set()
    for d in search_dirs:
        for fp in sorted(Path(d).glob('*.csv')):
            if fp.name.startswith('SUPPORT_') or fp.name.startswith('0') or fp.name.startswith('STEP2_'):
                continue
            key=fp.name.lower()
            if key not in seen:
                seen.add(key); files.append(fp)
    for fp in files:
        try:
            df=pd.read_csv(fp, dtype=str)
            cols=list(df.columns)
            score_cols=[c for c in cols if any(tok in c.lower() for tok in ['score','support','confidence','hit','lift','precision','count'])]
            key_cols=[c for c in cols if c in ['StreamKey','core_str','member_str','target_core','candidate_member','trait_name','trait_value','SameCoreGapBucket','stream','PLAY_DATE','HISTORY_THROUGH']]
            rows.append({
                'file':fp.name,
                'source_dir':str(fp.parent),
                'rows':len(df),
                'columns':len(cols),
                'key_columns':' | '.join(key_cols),
                'score_support_columns':' | '.join(score_cols[:30]),
                'has_StreamKey':'StreamKey' in cols,
                'has_core_str':'core_str' in cols,
                'has_member_str':'member_str' in cols,
                'has_target_core':'target_core' in cols,
                'has_support_column':'support' in cols or 'total_support' in cols,
            })
        except Exception as e:
            rows.append({'file':fp.name,'rows':np.nan,'columns':np.nan,'error':str(e)})
    return pd.DataFrame(rows)

def support_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows=[]
    d=df.copy()
    for col in COMPONENT_COLS + ['major_support_count','all_support_count']:
        if col in d.columns:
            s=_num(d[col])
            rows.append({
                'table':label,'field':col,'present':True,'rows':len(d),
                'nonzero_rows':int(s.gt(0).sum()),
                'pct_nonzero':round(float(s.gt(0).mean()*100),2) if len(s) else 0,
                'min':float(s.min()) if len(s) else np.nan,
                'mean':float(s.mean()) if len(s) else np.nan,
                'max':float(s.max()) if len(s) else np.nan,
                'distinct_values':int(s.nunique()) if len(s) else 0,
            })
        else:
            rows.append({'table':label,'field':col,'present':False,'rows':len(d),'nonzero_rows':0,'pct_nonzero':0})
    return pd.DataFrame(rows)

def recompute_support_counts(df: pd.DataFrame) -> pd.DataFrame:
    d=df.copy()
    for col in COMPONENT_COLS:
        if col not in d.columns:
            d[col]=0
    d['debug_recomputed_major_support_count']=sum(_num(d[c]).gt(0).astype(int) for c in MAJOR_COMPONENTS)
    d['debug_recomputed_all_support_count']=sum(_num(d[c]).gt(0).astype(int) for c in COMPONENT_COLS)
    d['support_count_mismatch_major']=False
    d['support_count_mismatch_all']=False
    if 'major_support_count' in d.columns:
        d['support_count_mismatch_major']=_num(d['major_support_count']).astype(int).ne(d['debug_recomputed_major_support_count'].astype(int))
    if 'all_support_count' in d.columns:
        d['support_count_mismatch_all']=_num(d['all_support_count']).astype(int).ne(d['debug_recomputed_all_support_count'].astype(int))
    return d

def join_audit_rows(step2: pd.DataFrame, limit: int|None=None) -> pd.DataFrame:
    d=recompute_support_counts(step2)
    rows=[]
    take=d if limit is None else d.head(limit)
    for _,r in take.iterrows():
        trait_hits=[]
        for c in KEY_TRAIT_COLS:
            if c in r.index and pd.notna(r.get(c,'')) and str(r.get(c,''))!='':
                trait_hits.append(f'{c}={r.get(c)}')
        rec={
            'play_date':r.get('play_date',''),
            'stream':r.get('stream',''),
            'seed':r.get('seed',''),
            'core':str(r.get('core','')).zfill(3),
            'member':str(r.get('member','')).zfill(4),
            'same_core_gap_bucket':r.get('same_core_gap_bucket',''),
            'seed_trait_values':' | '.join(trait_hits),
            'major_support_count_saved':r.get('major_support_count',np.nan),
            'all_support_count_saved':r.get('all_support_count',np.nan),
            'major_support_count_recomputed':r.get('debug_recomputed_major_support_count',np.nan),
            'all_support_count_recomputed':r.get('debug_recomputed_all_support_count',np.nan),
            'mismatch_major':r.get('support_count_mismatch_major',False),
            'mismatch_all':r.get('support_count_mismatch_all',False),
        }
        for c in COMPONENT_COLS:
            rec[c]=r.get(c,0)
            rec[c+'_fired']=float(pd.to_numeric(pd.Series([r.get(c,0)]), errors='coerce').fillna(0).iloc[0])>0
        rows.append(rec)
    return pd.DataFrame(rows)

def component_summary_by_core_stream(step2: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    d=recompute_support_counts(step2)
    for c in COMPONENT_COLS + ['major_support_count','all_support_count','debug_recomputed_major_support_count','debug_recomputed_all_support_count']:
        if c in d.columns: d[c]=_num(d[c])
    agg={}
    for c in COMPONENT_COLS:
        agg[c+'_fired_rows']=(c, lambda x: int(_num(x).gt(0).sum()))
        agg[c+'_max']=(c,'max')
    agg.update({
        'rows':('member','size'),
        'saved_major_max':('major_support_count','max'),
        'recomputed_major_max':('debug_recomputed_major_support_count','max'),
        'saved_all_max':('all_support_count','max'),
        'recomputed_all_max':('debug_recomputed_all_support_count','max'),
        'mismatch_major_rows':('support_count_mismatch_major', lambda x: int(pd.Series(x).fillna(False).sum())),
    })
    by_core=d.groupby(['play_date','core'], as_index=False).agg(**agg) if {'play_date','core'}.issubset(d.columns) else pd.DataFrame()
    by_stream=d.groupby(['play_date','stream'], as_index=False).agg(**agg) if {'play_date','stream'}.issubset(d.columns) else pd.DataFrame()
    return by_core, by_stream

def diagnosis(profile_dir, full: pd.DataFrame, step2_before: pd.DataFrame|None, step2_after: pd.DataFrame) -> pd.DataFrame:
    inv=profile_inventory(profile_dir)
    full_s=support_summary(full,'full_step1')
    step2_s=support_summary(step2_after,'step2_after_transition')
    diag=[]
    profile_files=len(inv)
    profile_rows=int(pd.to_numeric(inv.get('rows', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not inv.empty else 0
    full_major_nonzero=int(full_s.loc[full_s.field.eq('major_support_count'),'nonzero_rows'].iloc[0]) if not full_s.empty and full_s.field.eq('major_support_count').any() else 0
    step2_major_nonzero=int(step2_s.loc[step2_s.field.eq('major_support_count'),'nonzero_rows'].iloc[0]) if not step2_s.empty and step2_s.field.eq('major_support_count').any() else 0
    full_component_nonzero=int(full_s[full_s.field.isin(COMPONENT_COLS)]['nonzero_rows'].sum()) if not full_s.empty else 0
    step2_component_nonzero=int(step2_s[step2_s.field.isin(COMPONENT_COLS)]['nonzero_rows'].sum()) if not step2_s.empty else 0
    diag.append({'check':'profile_files_loaded','value':profile_files,'status':'OK' if profile_files>0 else 'FAIL','note':'Number of profile CSVs found in profiles/ OR repo root fallback.'})
    diag.append({'check':'profile_total_rows','value':profile_rows,'status':'OK' if profile_rows>0 else 'FAIL','note':'Total profile/rule rows available.'})
    diag.append({'check':'full_step1_major_support_nonzero_rows','value':full_major_nonzero,'status':'OK' if full_major_nonzero>0 else 'WARN','note':'If zero, support joins failed before Step 2.'})
    diag.append({'check':'step2_major_support_nonzero_rows','value':step2_major_nonzero,'status':'OK' if step2_major_nonzero>0 else 'FAIL','note':'If zero but Step1 nonzero, support was lost during Step2 handoff/scope/export.'})
    diag.append({'check':'full_step1_component_nonzero_total','value':full_component_nonzero,'status':'OK' if full_component_nonzero>0 else 'WARN','note':'Nonzero score components in Step1.'})
    diag.append({'check':'step2_component_nonzero_total','value':step2_component_nonzero,'status':'OK' if step2_component_nonzero>0 else 'FAIL','note':'Nonzero score components still present in Step2.'})
    # recompute mismatch
    if step2_after is not None and not step2_after.empty:
        rec=recompute_support_counts(step2_after)
        mism_major=int(rec['support_count_mismatch_major'].sum()) if 'support_count_mismatch_major' in rec else 0
        mism_all=int(rec['support_count_mismatch_all'].sum()) if 'support_count_mismatch_all' in rec else 0
        diag.append({'check':'major_support_saved_vs_recomputed_mismatch_rows','value':mism_major,'status':'FAIL' if mism_major>0 else 'OK','note':'Mismatch means saved major_support_count is wrong even though component scores are present.'})
        diag.append({'check':'all_support_saved_vs_recomputed_mismatch_rows','value':mism_all,'status':'FAIL' if mism_all>0 else 'OK','note':'Mismatch means saved all_support_count is wrong even though component scores are present.'})
    return pd.DataFrame(diag)

def write_support_debug_outputs(out_dir, profile_dir, play_date, full, step2_before, step2_after):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    suffix=str(play_date)
    inv=profile_inventory(profile_dir)
    inv.to_csv(out/'SUPPORT_00_PROFILE_INVENTORY.csv', index=False)
    sumdf=pd.concat([
        support_summary(full,'full_step1'),
        support_summary(step2_before,'step2_before_transition') if step2_before is not None else pd.DataFrame(),
        support_summary(step2_after,'step2_after_transition')
    ], ignore_index=True)
    sumdf.to_csv(out/f'SUPPORT_01_SIGNAL_SUMMARY_{suffix}.csv', index=False)
    rec=recompute_support_counts(step2_after)
    audit_cols=['play_date','stream','seed','core','member'] + COMPONENT_COLS + ['major_support_count','all_support_count','debug_recomputed_major_support_count','debug_recomputed_all_support_count','support_count_mismatch_major','support_count_mismatch_all']
    rec[[c for c in audit_cols if c in rec.columns]].to_csv(out/f'SUPPORT_02_RECOMPUTED_COUNTS_{suffix}.csv', index=False)
    join_audit_rows(step2_after).to_csv(out/f'SUPPORT_03_JOIN_AUDIT_ROWS_{suffix}.csv', index=False)
    by_core, by_stream=component_summary_by_core_stream(step2_after)
    by_core.to_csv(out/f'SUPPORT_04_BY_CORE_{suffix}.csv', index=False)
    by_stream.to_csv(out/f'SUPPORT_05_BY_STREAM_{suffix}.csv', index=False)
    diagnosis(profile_dir, full, step2_before, step2_after).to_csv(out/f'SUPPORT_99_DIAGNOSIS_{suffix}.csv', index=False)

class _EmbeddedSupportDebug:
    write_support_debug_outputs = staticmethod(write_support_debug_outputs)

sd = _EmbeddedSupportDebug()
# ---- end embedded support_debug ----

BUILD_ID = "MEMBER_LOCATION_AUDITOR_V4_3_SUPPORT_DISPLAY_FIX"
BUILD_LABEL = "Member Location Auditor V4.3 — support display fix + support-count debug + lockdown audit"
WATCHED8 = set(getattr(eng, 'WATCHED8', {'027','067','138','145','389','457','567','679'}))

BUCKET_BASIS_OPTIONS = [
    'final_x15_positive',
    'major_ge3',
    'major_ge4',
    'good_transition_ge1',
    'good_ge1_no_bad',
    'major_ge3_and_good_ge1',
    'major_ge4_and_good_ge1',
    'profile_score_positive',
    'all_step2_rows_cartesian_reference',
]
STEP2_SCOPE_OPTIONS = ['watched8_all_members','watched8_positive_support','full120_all_members','legacy_q2_balanced']

CORE_FILTERS = [
    'CORE_all','CORE_above_mean','CORE_at_mean','CORE_below_mean','CORE_at_or_above_mean','CORE_at_or_below_mean',
    'CORE_above_median','CORE_at_median','CORE_below_median','CORE_at_or_above_median','CORE_at_or_below_median','CORE_is_max','CORE_is_min'
]
STREAM_FILTERS = [
    'STREAM_all','STREAM_above_mean','STREAM_at_mean','STREAM_below_mean','STREAM_at_or_above_mean','STREAM_at_or_below_mean',
    'STREAM_above_median','STREAM_at_median','STREAM_below_median','STREAM_at_or_above_median','STREAM_at_or_below_median','STREAM_is_max','STREAM_is_min'
]
QUAL_FILTERS = [
    'all_rows','safe_no_bad','aggressive_good1_no_bad','major_ge4','major_ge4_and_aggressive','major_ge3_and_good_ge1', 'major_ge4_and_good_ge1'
]


TRANSITION_FILTERS = [
    'TRANS_all',
    'TRANS_score_ge_0', 'TRANS_score_ge_0_25', 'TRANS_score_ge_0_5', 'TRANS_score_ge_1',
    'GOOD_ge1', 'GOOD_ge2', 'GOOD_ge3', 'GOOD_ge4',
    'BAD_eq0', 'BAD_le1', 'BAD_le2',
    'GOOD_ge1_BAD_le1', 'GOOD_ge1_BAD_le2', 'GOOD_ge2_BAD_le2',
    'X15_ge_0', 'X15_ge_10', 'X15_ge_15', 'X15_ge_20',
]

# Compact, targeted filters for the immediate playable-system lockdown search.
LOCKDOWN_CORE_FILTERS = [
    'CORE_all', 'CORE_above_mean', 'CORE_at_or_above_mean', 'CORE_at_or_above_median',
    'CORE_is_max', 'CORE_at_or_above_median_OR_is_max'
]
LOCKDOWN_STREAM_FILTERS = [
    'STREAM_all', 'STREAM_below_mean', 'STREAM_at_or_below_mean', 'STREAM_below_median',
    'STREAM_at_or_below_median', 'STREAM_at_or_above_median', 'STREAM_is_max'
]
LOCKDOWN_QUAL_FILTERS = [
    'all_rows', 'major_ge4', 'major_ge3_and_good_ge1', 'major_ge4_and_good_ge1',
    'major_ge4_and_aggressive'
]
LOCKDOWN_TRANSITION_FILTERS = [
    'TRANS_all', 'TRANS_score_ge_0_25', 'TRANS_score_ge_0_5', 'TRANS_score_ge_1',
    'GOOD_ge1', 'GOOD_ge2', 'GOOD_ge3', 'GOOD_ge1_BAD_le2',
    'X15_ge_10', 'X15_ge_15'
]


def normalize_date(x):
    return pd.to_datetime(x).strftime('%Y-%m-%d')


def display_stream(state, game):
    state = '' if pd.isna(state) else str(state).strip()
    game = '' if pd.isna(game) else str(game).strip()
    if state and game: return f"{state} | {game}"
    return state or game


def read_winners_any(path_or_file) -> pd.DataFrame:
    """Read winner targets from TXT/CSV. Returns play_date, state, game, stream, result, member, core."""
    if path_or_file is None:
        return pd.DataFrame(columns=['play_date','state','game','stream','result','member','core','is_watched8_core'])
    # Streamlit uploads have .name and .read; local paths are strings/Path.
    if hasattr(path_or_file, 'read'):
        name = getattr(path_or_file, 'name', 'uploaded_winners')
        data = path_or_file.read()
        if isinstance(data, bytes): text = data.decode('utf-8', errors='ignore')
        else: text = str(data)
        suffix = Path(name).suffix.lower()
    else:
        p = Path(path_or_file)
        text = p.read_text(encoding='utf-8', errors='ignore')
        suffix = p.suffix.lower()

    if suffix == '.csv':
        from io import StringIO
        df = pd.read_csv(StringIO(text), dtype=str)
        cols = {c.lower().strip(): c for c in df.columns}
        date_col = cols.get('date') or cols.get('draw_date') or cols.get('play_date')
        state_col = cols.get('state')
        game_col = cols.get('game')
        result_col = cols.get('result') or cols.get('result4') or cols.get('base4') or cols.get('winner')
        stream_col = cols.get('stream') or cols.get('streamkey') or cols.get('stream_name')
        rows=[]
        for _,r in df.iterrows():
            result = r.get(result_col, '') if result_col else ''
            m = re.search(r'(\d)[-\s]?(\d)[-\s]?(\d)[-\s]?(\d)', str(result))
            if not m: continue
            base4 = ''.join(m.groups())
            state = r.get(state_col,'') if state_col else ''
            game = r.get(game_col,'') if game_col else ''
            stream = r.get(stream_col,'') if stream_col else display_stream(state, game)
            date = r.get(date_col, '') if date_col else ''
            try: date = normalize_date(date)
            except Exception: date = ''
            core = eng.core_from_result(base4); member = eng.boxed_member(base4)
            rows.append({'play_date':date,'state':state,'game':game,'stream':stream,'result':base4,'member':member,'core':core,'is_watched8_core':core in WATCHED8})
        return pd.DataFrame(rows)

    rows=[]
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        parts=line.split('\t')
        if len(parts) >= 4:
            date_raw,state,game,result = parts[0],parts[1],parts[2],parts[3]
        else:
            # Loose fallback: date, state, game words, first 4-digit/dashed result.
            mres = re.search(r'(\d)\s*[- ]\s*(\d)\s*[- ]\s*(\d)\s*[- ]\s*(\d)', line)
            if not mres: continue
            result = mres.group(0)
            date_raw=''; state=''; game=''
        m = re.search(r'(\d)\s*[- ]\s*(\d)\s*[- ]\s*(\d)\s*[- ]\s*(\d)', str(result))
        if not m: continue
        base4=''.join(m.groups())
        try: date=normalize_date(date_raw)
        except Exception: date=''
        stream=display_stream(state, game)
        core=eng.core_from_result(base4); member=eng.boxed_member(base4)
        rows.append({'play_date':date,'state':state,'game':game,'stream':stream,'result':base4,'member':member,'core':core,'is_watched8_core':core in WATCHED8})
    return pd.DataFrame(rows)


def winners_from_history(hist: pd.DataFrame, play_date: str, watched_only=True) -> pd.DataFrame:
    date = normalize_date(play_date)
    d = hist[hist['draw_date'].eq(date)].copy()
    if d.empty:
        return pd.DataFrame(columns=['play_date','state','game','stream','result','member','core','is_watched8_core'])
    out = pd.DataFrame({
        'play_date': d['draw_date'].astype(str),
        'state': d.get('state',''),
        'game': d.get('game',''),
        'stream': d['stream'].astype(str),
        'result': d['base4'].astype(str).str.zfill(4),
        'member': d['member'].astype(str).str.zfill(4),
        'core': d['core'].astype(str).str.zfill(3),
    })
    out['is_watched8_core'] = out['core'].isin(WATCHED8)
    if watched_only:
        out = out[out['is_watched8_core']].copy()
    return out.reset_index(drop=True)


def choose_basis(step2: pd.DataFrame, basis: str) -> pd.DataFrame:
    d = step2.copy()
    num = lambda c: pd.to_numeric(d.get(c, 0), errors='coerce').fillna(0)
    if basis == 'all_step2_rows_cartesian_reference':
        return d
    if basis == 'final_x15_positive':
        return d[num('final_plus_transition_x15').gt(0)].copy()
    if basis == 'major_ge3':
        return d[num('major_support_count').ge(3)].copy()
    if basis == 'major_ge4':
        return d[num('major_support_count').ge(4)].copy()
    if basis == 'good_transition_ge1':
        return d[num('good_transition_count').ge(1)].copy()
    if basis == 'good_ge1_no_bad':
        return d[num('good_transition_count').ge(1) & num('bad_transition_count').le(0)].copy()
    if basis == 'major_ge3_and_good_ge1':
        return d[num('major_support_count').ge(3) & num('good_transition_count').ge(1)].copy()
    if basis == 'major_ge4_and_good_ge1':
        return d[num('major_support_count').ge(4) & num('good_transition_count').ge(1)].copy()
    if basis == 'profile_score_positive':
        return d[num('profile_final_member_score').gt(0)].copy()
    raise ValueError(f'Unknown bucket basis: {basis}')


def add_rank_columns(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    sort_cols = ['final_plus_transition_x15','profile_final_member_score','major_support_count','all_support_count','stream_gate_rank','stream','core','member']
    for c in sort_cols:
        if c not in out.columns: out[c] = 0 if c not in ['stream','core','member'] else ''
    out = out.sort_values(sort_cols, ascending=[False,False,False,False,True,True,True,True]).copy()
    out['rank_unified_best_to_worst'] = range(1, len(out)+1)
    return out


def build_day_step2(hist: pd.DataFrame, prof: dict, play_date: str, history_through: str, step2_scope='watched8_all_members', gate_top=50, bucket_basis='final_x15_positive'):
    events = eng.derive_seed_events(hist, history_through, play_date)
    full = eng.build_full_enumeration(hist, events, prof)
    gate = eng.build_stream_gate(full, hist=hist, history_through=history_through, mode='baseline_n47')
    step2base = eng.build_step2_candidate_base(full, gate, use_top=int(gate_top), step2_scope=step2_scope)
    step2 = eng.apply_step2_transition(step2base, hist, history_through, x=15)
    basis_df = choose_basis(step2, bucket_basis)
    bucketed, core_counts, stream_counts = eng.add_rowcount_buckets(step2, basis_df=basis_df)
    bucketed = add_rank_columns(bucketed)
    return {'events': events, 'full': full, 'gate': gate, 'step2base': step2base, 'step2': bucketed, 'basis': basis_df, 'core_counts': core_counts, 'stream_counts': stream_counts}


def match_winners_to_step2(step2: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    if winners is None or winners.empty:
        return pd.DataFrame()
    d = step2.copy()
    for c in ['play_date','stream','core','member']:
        if c not in d.columns: d[c]=''
    d['play_date'] = d['play_date'].astype(str)
    d['stream_norm'] = d['stream'].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    d['core'] = d['core'].astype(str).str.zfill(3)
    d['member'] = d['member'].astype(str).str.zfill(4)
    rows=[]
    for _,w in winners.iterrows():
        stream_norm = str(w.get('stream','')).lower().strip()
        stream_norm = re.sub(r'\s+', ' ', stream_norm)
        core = str(w.get('core','')).zfill(3)
        member = str(w.get('member','')).zfill(4)
        play_date = str(w.get('play_date',''))
        m = d[(d['play_date'].astype(str).eq(play_date)) & (d['stream_norm'].eq(stream_norm)) & (d['core'].eq(core)) & (d['member'].eq(member))].copy()
        if m.empty:
            # fallback by stream+member only if core mismatch due parsing should not happen
            m = d[(d['play_date'].astype(str).eq(play_date)) & (d['stream_norm'].eq(stream_norm)) & (d['member'].eq(member))].copy()
        if m.empty:
            rows.append({
                'play_date': play_date, 'winner_stream': w.get('stream',''), 'winner_result': w.get('result',''),
                'winner_member': member, 'winner_core': core, 'in_step2': False, 'rank_unified_best_to_worst': np.nan,
                'core_mean_bucket':'MISSING','core_median_bucket':'MISSING','stream_mean_bucket':'MISSING','stream_median_bucket':'MISSING'
            })
        else:
            r = m.iloc[0].to_dict()
            def bucket(prefix, system):
                if bool(r.get(f'{prefix}_above_{system}', False)): return f'above_{system}'
                if bool(r.get(f'{prefix}_at_{system}', False)): return f'at_{system}'
                if bool(r.get(f'{prefix}_below_{system}', False)): return f'below_{system}'
                return 'unknown'
            rows.append({
                'play_date': play_date, 'winner_stream': w.get('stream',''), 'winner_result': w.get('result',''),
                'winner_member': member, 'winner_core': core, 'in_step2': True,
                'rank_unified_best_to_worst': r.get('rank_unified_best_to_worst'),
                'step2_x15_rank': r.get('step2_x15_rank'),
                'seed': r.get('seed'), 'score_final_x15': r.get('final_plus_transition_x15'),
                'profile_final_member_score': r.get('profile_final_member_score'), 'transition_compat_score': r.get('transition_compat_score'),
                'major_support_count': r.get('major_support_count'), 'all_support_count': r.get('all_support_count'),
                'good_transition_count': r.get('good_transition_count'), 'bad_transition_count': r.get('bad_transition_count'),
                'core_row_count': r.get('core_row_count'), 'mean_core_row_count': r.get('mean_core_row_count'), 'median_core_row_count': r.get('median_core_row_count'),
                'stream_row_count': r.get('stream_row_count'), 'mean_stream_row_count': r.get('mean_stream_row_count'), 'median_stream_row_count': r.get('median_stream_row_count'),
                'core_mean_bucket': bucket('core','mean'), 'core_median_bucket': bucket('core','median'),
                'stream_mean_bucket': bucket('stream','mean'), 'stream_median_bucket': bucket('stream','median'),
                'core_is_max': r.get('core_is_max'), 'core_is_min': r.get('core_is_min'),
                'stream_is_max': r.get('stream_is_max'), 'stream_is_min': r.get('stream_is_min'),
            })
    return pd.DataFrame(rows)


def filter_mask(df: pd.DataFrame, qual='all_rows', core_filter='CORE_all', stream_filter='STREAM_all', transition_filter='TRANS_all'):
    d=df
    num=lambda c: pd.to_numeric(d.get(c,0), errors='coerce').fillna(0)
    mask=pd.Series(True, index=d.index)
    if qual == 'safe_no_bad': mask &= num('bad_transition_count').le(0)
    elif qual == 'aggressive_good1_no_bad': mask &= num('good_transition_count').ge(1) & num('bad_transition_count').le(0)
    elif qual == 'major_ge4': mask &= num('major_support_count').ge(4)
    elif qual == 'major_ge4_and_good_ge1': mask &= num('major_support_count').ge(4) & num('good_transition_count').ge(1)
    elif qual == 'major_ge4_and_aggressive': mask &= num('major_support_count').ge(4) & num('good_transition_count').ge(1) & num('bad_transition_count').le(0)
    elif qual == 'major_ge3_and_good_ge1': mask &= num('major_support_count').ge(3) & num('good_transition_count').ge(1)
    elif qual != 'all_rows': raise ValueError(f'Unknown qualification: {qual}')

    def core_or_stream_mask(prefix, f):
        if f.endswith('_all'):
            return pd.Series(True, index=d.index)
        pf = prefix.lower()
        if '_OR_' in f:
            left,right = f.split('_OR_',1)
            return core_or_stream_mask(prefix, left) | core_or_stream_mask(prefix, prefix+'_'+right)
        s=f.replace(prefix+'_','').lower()
        if s == 'is_max': col = pf+'_is_max'
        elif s == 'is_min': col = pf+'_is_min'
        else: col = pf+'_'+s
        if col in d.columns:
            return d[col].fillna(False).astype(bool)
        return pd.Series(False, index=d.index)

    mask &= core_or_stream_mask('CORE', core_filter)
    mask &= core_or_stream_mask('STREAM', stream_filter)

    tf = transition_filter
    if tf == 'TRANS_all': pass
    elif tf == 'TRANS_score_ge_0': mask &= num('transition_compat_score').ge(0)
    elif tf == 'TRANS_score_ge_0_25': mask &= num('transition_compat_score').ge(0.25)
    elif tf == 'TRANS_score_ge_0_5': mask &= num('transition_compat_score').ge(0.5)
    elif tf == 'TRANS_score_ge_1': mask &= num('transition_compat_score').ge(1)
    elif tf == 'GOOD_ge1': mask &= num('good_transition_count').ge(1)
    elif tf == 'GOOD_ge2': mask &= num('good_transition_count').ge(2)
    elif tf == 'GOOD_ge3': mask &= num('good_transition_count').ge(3)
    elif tf == 'GOOD_ge4': mask &= num('good_transition_count').ge(4)
    elif tf == 'BAD_eq0': mask &= num('bad_transition_count').le(0)
    elif tf == 'BAD_le1': mask &= num('bad_transition_count').le(1)
    elif tf == 'BAD_le2': mask &= num('bad_transition_count').le(2)
    elif tf == 'GOOD_ge1_BAD_le1': mask &= num('good_transition_count').ge(1) & num('bad_transition_count').le(1)
    elif tf == 'GOOD_ge1_BAD_le2': mask &= num('good_transition_count').ge(1) & num('bad_transition_count').le(2)
    elif tf == 'GOOD_ge2_BAD_le2': mask &= num('good_transition_count').ge(2) & num('bad_transition_count').le(2)
    elif tf == 'X15_ge_0': mask &= num('final_plus_transition_x15').ge(0)
    elif tf == 'X15_ge_10': mask &= num('final_plus_transition_x15').ge(10)
    elif tf == 'X15_ge_15': mask &= num('final_plus_transition_x15').ge(15)
    elif tf == 'X15_ge_20': mask &= num('final_plus_transition_x15').ge(20)
    else: raise ValueError(f'Unknown transition filter: {transition_filter}')
    return mask

def whatif_matrix(step2: pd.DataFrame, winners_loc: pd.DataFrame, quals=None, core_filters=None, stream_filters=None, transition_filters=None, topn_list=(20,30,40,47,50,75,100), play_cap=50, lockdown_mode=True) -> pd.DataFrame:
    """Fast Step 3 what-if matrix with transition/score filters and Top-N rank quality."""
    quals = quals or (LOCKDOWN_QUAL_FILTERS if lockdown_mode else QUAL_FILTERS)
    core_filters = core_filters or (LOCKDOWN_CORE_FILTERS if lockdown_mode else CORE_FILTERS)
    stream_filters = stream_filters or (LOCKDOWN_STREAM_FILTERS if lockdown_mode else STREAM_FILTERS)
    transition_filters = transition_filters or (LOCKDOWN_TRANSITION_FILTERS if lockdown_mode else TRANSITION_FILTERS)
    topn_list = tuple(sorted(set(int(x) for x in topn_list if int(x) > 0)))
    rows=[]
    winner_keys=set()
    if winners_loc is not None and not winners_loc.empty:
        for _,w in winners_loc[winners_loc.get('in_step2', False)==True].iterrows():
            winner_keys.add((str(w['play_date']), str(w['winner_stream']).lower().strip(), str(w['winner_core']).zfill(3), str(w['winner_member']).zfill(4)))
    d=step2.copy()
    if d.empty:
        return pd.DataFrame()
    # Sort once. Any Step 3 filtered list uses the same score/rank order, so relative order is preserved.
    d=add_rank_columns(d)
    d['stream_norm']=d['stream'].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    d['core']=d['core'].astype(str).str.zfill(3)
    d['member']=d['member'].astype(str).str.zfill(4)
    d['play_date']=d['play_date'].astype(str)
    d['_key'] = list(zip(d['play_date'].astype(str), d['stream_norm'].astype(str), d['core'].astype(str), d['member'].astype(str)))

    # Precompute masks for speed.
    qual_masks={q: filter_mask(d, q, 'CORE_all', 'STREAM_all', 'TRANS_all') for q in quals}
    core_masks={cf: filter_mask(d, 'all_rows', cf, 'STREAM_all', 'TRANS_all') for cf in core_filters}
    stream_masks={sf: filter_mask(d, 'all_rows', 'CORE_all', sf, 'TRANS_all') for sf in stream_filters}
    trans_masks={tf: filter_mask(d, 'all_rows', 'CORE_all', 'STREAM_all', tf) for tf in transition_filters}

    keys_array=d['_key'].tolist()
    for q in quals:
        qm=qual_masks[q]
        for cf in core_filters:
            qcm = qm & core_masks[cf]
            if not qcm.any():
                continue
            for sf in stream_filters:
                qcms = qcm & stream_masks[sf]
                if not qcms.any():
                    continue
                for tf in transition_filters:
                    mask = qcms & trans_masks[tf]
                    idx=np.flatnonzero(mask.to_numpy())
                    if len(idx)==0:
                        keep_keys=[]
                    else:
                        keep_keys=[keys_array[i] for i in idx]
                    kept_set=set(keep_keys)
                    found = sorted(winner_keys & kept_set)
                    ranks=[]
                    if found:
                        rank_lookup={k: r+1 for r,k in enumerate(keep_keys)}
                        ranks=[rank_lookup[k] for k in found if k in rank_lookup]
                    row={
                        'qualification':q,'core_filter':cf,'stream_filter':sf,'transition_filter':tf,
                        'rows_kept':int(len(idx)),
                        'under_or_equal_play_cap': int(len(idx)) <= int(play_cap),
                        'winner_targets_in_step2':len(winner_keys),
                        'winner_kept':len(found),
                        'winner_missed_after_filter': max(len(winner_keys)-len(found),0),
                        'best_winner_rank': min(ranks) if ranks else np.nan,
                        'worst_winner_rank': max(ranks) if ranks else np.nan,
                        'avg_winner_rank': float(np.mean(ranks)) if ranks else np.nan,
                    }
                    for topn in topn_list:
                        row[f'winner_top{topn}']=sum(1 for r in ranks if r <= int(topn))
                    rows.append(row)
    out=pd.DataFrame(rows)
    if not out.empty:
        out['capture_rate_kept'] = np.where(out['winner_targets_in_step2'].gt(0), out['winner_kept']/out['winner_targets_in_step2'], np.nan)
        for topn in topn_list:
            c=f'winner_top{topn}'
            out[f'capture_rate_top{topn}'] = np.where(out['winner_targets_in_step2'].gt(0), out[c]/out['winner_targets_in_step2'], np.nan)
        sort_cols=[]; ascending=[]
        if 'winner_top50' in out.columns:
            sort_cols.append('winner_top50'); ascending.append(False)
        sort_cols += ['under_or_equal_play_cap','winner_kept','rows_kept','avg_winner_rank']
        ascending += [False,False,True,True]
        out=out.sort_values(sort_cols, ascending=ascending)
    return out

def summarize_topn_quality(what_all: pd.DataFrame, play_cap=50) -> pd.DataFrame:
    if what_all is None or what_all.empty:
        return pd.DataFrame()
    top_cols=[c for c in what_all.columns if c.startswith('winner_top')]
    agg={
        'dates_tested':('play_date','nunique'),
        'total_rows_kept':('rows_kept','sum'),
        'avg_rows_kept':('rows_kept','mean'),
        'max_rows_kept':('rows_kept','max'),
        'winner_targets':('winner_targets_in_step2','sum'),
        'winners_kept':('winner_kept','sum'),
        'avg_winner_rank_mean':('avg_winner_rank','mean'),
        'worst_winner_rank_max':('worst_winner_rank','max'),
    }
    for c in top_cols:
        agg[c]=(c,'sum')
    combo=what_all.groupby(['qualification','core_filter','stream_filter','transition_filter'], as_index=False).agg(**agg)
    combo['avg_rows_under_or_equal_play_cap'] = combo['avg_rows_kept'].le(int(play_cap))
    combo['max_rows_under_or_equal_play_cap'] = combo['max_rows_kept'].le(int(play_cap))
    combo['capture_rate_kept']=np.where(combo['winner_targets'].gt(0), combo['winners_kept']/combo['winner_targets'], np.nan)
    if 'winner_top50' in what_all.columns:
        dayhits = what_all.assign(day_has_top50=what_all['winner_top50'].gt(0).astype(int)).groupby(['qualification','core_filter','stream_filter','transition_filter'])['day_has_top50'].sum().reset_index(name='days_with_top50_winner')
        combo = combo.merge(dayhits, on=['qualification','core_filter','stream_filter','transition_filter'], how='left')
    else:
        combo['days_with_top50_winner'] = 0
    combo['days_with_top50_winner'] = combo['days_with_top50_winner'].fillna(0).astype(int)
    combo['weekly_75pct_days_top50'] = np.where(combo['dates_tested'].gt(0), combo['days_with_top50_winner'] / combo['dates_tested'], np.nan)
    for c in top_cols:
        n=c.replace('winner_top','')
        combo[f'capture_rate_top{n}']=np.where(combo['winner_targets'].gt(0), combo[c]/combo['winner_targets'], np.nan)
    sort_cols=[]; ascending=[]
    if 'days_with_top50_winner' in combo.columns:
        sort_cols += ['days_with_top50_winner']; ascending += [False]
    if 'winner_top50' in combo.columns:
        sort_cols += ['winner_top50']; ascending += [False]
    sort_cols += ['winners_kept','avg_rows_under_or_equal_play_cap','avg_rows_kept','avg_winner_rank_mean']
    ascending += [False,False,True,True]
    return combo.sort_values(sort_cols, ascending=ascending)

def replay_audit(history_path, profile_dir, out_dir, start_date, end_date, winners_path=None, use_history_winners=True, exclude_az_md=True, step2_scope='watched8_all_members', bucket_basis='final_x15_positive', gate_top=50, max_dates=31, progress_cb=None, audit_watched_only=True, play_cap=50):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    hist0=eng.read_history(history_path)
    hist, exclusion_audit=eng.apply_exclusions(hist0, exclude_az_md=exclude_az_md)
    prof=eng.load_profiles(profile_dir)
    start=pd.to_datetime(start_date); end=pd.to_datetime(end_date)
    dates=[d.strftime('%Y-%m-%d') for d in pd.date_range(start,end,freq='D')]
    if len(dates) > int(max_dates):
        dates=dates[:int(max_dates)]
    external_winners = read_winners_any(winners_path) if winners_path else pd.DataFrame()
    if audit_watched_only and not external_winners.empty and 'is_watched8_core' in external_winners.columns:
        external_winners = external_winners[external_winners['is_watched8_core'].astype(bool)].copy()
    all_loc=[]; all_summ=[]; all_what=[]; logs=[]
    for i,play_date in enumerate(dates, start=1):
        t0=time.time()
        through=(pd.to_datetime(play_date)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if progress_cb: progress_cb(i, len(dates), play_date, 'building Step 0-2')
        try:
            parts=build_day_step2(hist, prof, play_date, through, step2_scope=step2_scope, gate_top=gate_top, bucket_basis=bucket_basis)
            step2=parts['step2']
            # V4: write support-count/join diagnostics for every replayed date.
            try:
                sd.write_support_debug_outputs(out, profile_dir, play_date, parts.get('full', pd.DataFrame()), parts.get('step2base', pd.DataFrame()), step2)
            except Exception as _support_debug_error:
                logs.append({'play_date':play_date,'stage':'support_debug','error':str(_support_debug_error)})
            # Winners: use external if present for date, otherwise use history winners if requested.
            if not external_winners.empty:
                winners=external_winners[external_winners['play_date'].astype(str).eq(play_date)].copy()
            else:
                winners=pd.DataFrame()
            if winners.empty and use_history_winners:
                winners=winners_from_history(hist, play_date, watched_only=True)
            loc=match_winners_to_step2(step2, winners) if not winners.empty else pd.DataFrame()
            if not loc.empty: all_loc.append(loc)
            wm=whatif_matrix(step2, loc, play_cap=play_cap) if not loc.empty else pd.DataFrame()
            if not wm.empty:
                wm.insert(0,'play_date',play_date); all_what.append(wm)
            summ={
                'play_date': play_date, 'history_through': through, 'status':'OK', 'seconds':round(time.time()-t0,2),
                'history_rows_after_exclusion': len(hist), 'seed_streams': parts['events']['stream'].nunique(),
                'step1_rows': len(parts['full']), 'stream_gate_rows': int(parts['gate'][f'in_stream_gate_top{gate_top}'].sum()) if f'in_stream_gate_top{gate_top}' in parts['gate'].columns else len(parts['gate']),
                'step2_rows': len(step2), 'bucket_basis_rows': len(parts['basis']), 'actual_watched_core_winners': len(winners),
                'winners_found_in_step2': int(loc['in_step2'].sum()) if not loc.empty and 'in_step2' in loc.columns else 0,
            }
            all_summ.append(summ)
            # Save one per-date compact step2 for deeper inspection but not full enumeration.
            step2_cols=[c for c in ['play_date','stream','seed','core','member','rank_unified_best_to_worst','final_plus_transition_x15','profile_final_member_score','transition_compat_score','major_support_count','all_support_count','good_transition_count','bad_transition_count','core_row_count','stream_row_count','core_above_mean','core_at_mean','core_below_mean','core_above_median','core_at_median','core_below_median','stream_above_mean','stream_at_mean','stream_below_mean','stream_above_median','stream_at_median','stream_below_median','core_is_max','core_is_min','stream_is_max','stream_is_min'] if c in step2.columns]
            step2[step2_cols].to_csv(out/f'STEP2_BUCKETED_ROWS_{play_date}.csv', index=False)
        except Exception as e:
            all_summ.append({'play_date':play_date,'history_through':through,'status':'FAILED','error':repr(e),'seconds':round(time.time()-t0,2)})
            logs.append({'play_date':play_date,'error':repr(e)})
    summary=pd.DataFrame(all_summ)
    loc_all=pd.concat(all_loc, ignore_index=True) if all_loc else pd.DataFrame()
    what_all=pd.concat(all_what, ignore_index=True) if all_what else pd.DataFrame()
    exclusion_audit.to_csv(out/'00_STEP0_EXCLUSION_AUDIT.csv', index=False)
    summary.to_csv(out/'00_AUDIT_RUN_SUMMARY.csv', index=False)
    loc_all.to_csv(out/'01_WINNER_LOCATION_IN_STEP2_BUCKETS.csv', index=False)
    what_all.to_csv(out/'02_STEP3_FILTER_WHATIF_WINNER_SURVIVAL.csv', index=False)
    pd.DataFrame(logs).to_csv(out/'99_FAILURE_LOG.csv', index=False)
    # Rollup combo performance with rank quality and Top-N counts.
    if not what_all.empty:
        combo=summarize_topn_quality(what_all, play_cap=play_cap)
        combo.to_csv(out/'03_BEST_STEP3_COMBOS_BY_CAPTURE_AND_PLAYCOUNT.csv', index=False)
        # Separate affordable subset for the actual play-cap goal.
        affordable=combo[combo['avg_rows_kept'].le(int(play_cap))].copy()
        affordable.to_csv(out/f'04_AFFORDABLE_COMBOS_AVG_ROWS_LE_{int(play_cap)}.csv', index=False)
        strict=combo[combo['max_rows_kept'].le(int(play_cap))].copy()
        strict.to_csv(out/f'05_STRICT_COMBOS_MAX_ROWS_LE_{int(play_cap)}.csv', index=False)
        target_days = int(np.ceil(0.75 * max(combo['dates_tested'].max(), 1))) if not combo.empty else 0
        lockdown = combo[(combo['max_rows_kept'].le(int(play_cap))) & (combo['days_with_top50_winner'].ge(target_days))].copy() if 'days_with_top50_winner' in combo.columns else pd.DataFrame()
        lockdown.to_csv(out/f'06_LOCKDOWN_CANDIDATES_75PCT_DAYS_TOP50_MAX_ROWS_LE_{int(play_cap)}.csv', index=False)
        near = combo[(combo['avg_rows_kept'].le(int(play_cap)*1.5)) & (combo['days_with_top50_winner'].ge(max(target_days-1,1)))].copy() if 'days_with_top50_winner' in combo.columns else pd.DataFrame()
        near.to_csv(out/f'07_NEAR_MISS_CANDIDATES_AVG_ROWS_LE_{int(play_cap)}x1_5.csv', index=False)
    # zip
    zip_path=out.parent/(f'{BUILD_ID}_{dates[0]}_TO_{dates[-1]}_OUTPUTS.zip' if dates else f'{BUILD_ID}_OUTPUTS.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(out))
    return {'summary':summary,'winner_locations':loc_all,'whatif':what_all,'zip_path':str(zip_path),'out_dir':str(out)}
