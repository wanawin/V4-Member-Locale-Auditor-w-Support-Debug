from __future__ import annotations
import tempfile, time, os, zipfile
from pathlib import Path
import pandas as pd
import streamlit as st

import audit_engine as ae

def safe_read_csv(path, **kwargs):
    try:
        if not Path(path).exists() or Path(path).stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        raise


st.set_page_config(page_title='Member Location Auditor V4.3 — Support Display Fix', layout='wide')

ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / 'profiles'
IN_DIR = ROOT / 'IN'
OUT_ROOT = ROOT / 'outputs'
OUT_ROOT.mkdir(exist_ok=True)

st.title('Member Location Auditor V4.3 — Support Debug + Lockdown Finder')
st.caption('BUILD: V4.3 SUPPORT DISPLAY FIX. Separate audit app: support-count/join diagnostics + historical Step 2 bucket locator + Step 3 what-if + Top-50 lockdown testing. This does not replace the working daily app.')
st.info('Purpose: first debug the missing support counts, then find a main playable system: at least 1 daily winner inside Top 50 on 75%+ of tested days, before straight/rescue layers.')

with st.sidebar:
    st.header('Inputs')
    history_file = st.file_uploader('History CSV/TXT', type=['csv','txt'], help='For multi-date replay, upload a history file that includes both training days and winner/play dates.')
    winners_file = st.file_uploader('Optional winners TXT/CSV', type=['txt','csv'], help='Use this when your history ends the day before the winner date. If omitted, the app uses next-day winners found inside the uploaded history.')
    use_sample = st.checkbox('Use included 06/18 sample history + 06/19 sample winners', value=(history_file is None))
    st.header('Replay window')
    start_date = st.date_input('Start play date', value=pd.to_datetime('2026-06-19').date())
    end_date = st.date_input('End play date', value=pd.to_datetime('2026-06-19').date())
    max_dates = st.slider('Maximum dates to run in this batch', 1, 45, 31)
    st.caption('Hard cap is 45 dates per batch in this first auditor build. Use 7–14 dates while testing settings; use 31 when you are ready for a longer audit.')
    st.header('Step 0–2 settings')
    exclude_az_md = st.checkbox('Exclude AZ/MD before building seeds', value=True)
    step2_scope = st.selectbox('Step 2 row scope', ae.STEP2_SCOPE_OPTIONS, index=0)
    bucket_basis = st.selectbox('Bucket-count basis', ae.BUCKET_BASIS_OPTIONS, index=0)
    gate_top = st.number_input('Stream gate count used before Step 2', min_value=1, max_value=78, value=50, step=1)
    play_cap = st.number_input('Affordable playlist cap for combo rollups', min_value=10, max_value=300, value=50, step=5)
    st.caption('The auditor still records Top20/30/40/47/50/75/100, but this cap controls the affordable-combo output files.')
    run_btn = st.button('Run lockdown audit', type='primary')

st.subheader('Correct audit path')
st.markdown('''
**For each play date**, this app rebuilds the same evidence path blind/walk-forward:

1. **Step 0** — use history only through the day before the play date; build seed streams.
2. **Step 1** — build broad candidate rows.
3. **Step 2** — apply corrected transition x15 and build average/median bucket labels.
4. **Winner locator** — normalize real winners to boxed member/core and find their Step 2 bucket positions.
5. **Step 3 what-if** — test keep/delete filter combinations without locking or changing the daily app.
6. **Step 4 rank-quality audit** — rank each surviving list best-to-worst and count winners inside Top20/30/40/47/50/75/100.

This is not a fake simulation. It still computes each date. The time saver is that it does many dates in one batch and tests many Step 3 ideas against each date automatically.
''')

st.subheader('What this app will tell you')
st.markdown('''
Output tables include:

- `01_WINNER_LOCATION_IN_STEP2_BUCKETS.csv` — where each winner fell: stream, seed, core, member, rank, score, core mean bucket, core median bucket, stream mean bucket, stream median bucket.
- `02_STEP3_FILTER_WHATIF_WINNER_SURVIVAL.csv` — whether each filter kept/deleted winners, plus winner Top-N rank counts.
- `03_BEST_STEP3_COMBOS_BY_CAPTURE_AND_PLAYCOUNT.csv` — combo rollup by capture, row count, rank quality, and days with a Top50 winner.
- `04_AFFORDABLE_COMBOS_AVG_ROWS_LE_<cap>.csv` — combos averaging within your play cap.
- `05_STRICT_COMBOS_MAX_ROWS_LE_<cap>.csv` — combos that stayed within your play cap on every tested date.
- `06_LOCKDOWN_CANDIDATES_75PCT_DAYS_TOP50_MAX_ROWS_LE_<cap>.csv` — strict candidates meeting the immediate main-system target.
- `07_NEAR_MISS_CANDIDATES_AVG_ROWS_LE_<cap>x1_5.csv` — close candidates worth reviewing if no strict lock appears.
- Per-date `STEP2_BUCKETED_ROWS_<date>.csv` — full Step 2 bucketed row table for searching.
- `SUPPORT_01_SIGNAL_SUMMARY_<date>.csv` — proves whether support scores are present before/after Step 2.
- `SUPPORT_02_RECOMPUTED_COUNTS_<date>.csv` — recomputes support counts from component score columns.
- `SUPPORT_03_JOIN_AUDIT_ROWS_<date>.csv` — row-level support/join audit: stream, seed, core, member, component scores, saved vs recomputed counts.
- `SUPPORT_99_DIAGNOSIS_<date>.csv` — clear OK/FAIL diagnosis for support loading/joins/handoff.
''')


def write_upload(upload, suffix='.csv'):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getvalue())
    tmp.close()
    return tmp.name

if run_btn:
    try:
        if use_sample and history_file is None:
            hist_path = str(IN_DIR / 'sample_history_THROUGH_2026-06-18.csv')
            win_path = str(IN_DIR / 'sample_06192026_pk4_dbl_winners.txt') if winners_file is None else write_upload(winners_file, Path(winners_file.name).suffix)
        else:
            if history_file is None:
                st.error('Upload a history file or check the sample-history box.')
                st.stop()
            hist_path = write_upload(history_file, Path(history_file.name).suffix or '.csv')
            win_path = write_upload(winners_file, Path(winners_file.name).suffix or '.txt') if winners_file is not None else None

        if pd.to_datetime(end_date) < pd.to_datetime(start_date):
            st.error('End date must be on or after start date.')
            st.stop()
        requested = len(pd.date_range(start_date, end_date, freq='D'))
        if requested > max_dates:
            st.warning(f'You requested {requested} dates. This run will process the first {max_dates} dates because of your max-date setting.')

        tag = f"{ae.BUILD_ID}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir = OUT_ROOT / tag
        progress = st.progress(0)
        status = st.empty()
        def cb(i, n, d, msg):
            progress.progress(i / max(n,1))
            status.write(f'Running {i}/{n}: {d} — {msg}')

        with st.spinner('Running blind historical replay...'):
            res = ae.replay_audit(
                history_path=hist_path,
                profile_dir=str(PROFILE_DIR),
                out_dir=str(out_dir),
                start_date=str(start_date),
                end_date=str(end_date),
                winners_path=win_path,
                use_history_winners=True,
                exclude_az_md=exclude_az_md,
                step2_scope=step2_scope,
                bucket_basis=bucket_basis,
                gate_top=int(gate_top),
                max_dates=int(max_dates),
                progress_cb=cb,
                audit_watched_only=True,
                play_cap=int(play_cap),
            )
        progress.progress(1.0)
        status.success('Audit complete.')
        st.session_state['audit_result'] = res
    except Exception as e:
        st.exception(e)

res = st.session_state.get('audit_result')
if res:
    st.header('Audit results')
    summary = res.get('summary', pd.DataFrame())
    loc = res.get('winner_locations', pd.DataFrame())
    what = res.get('whatif', pd.DataFrame())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Dates processed', len(summary) if summary is not None else 0)
    if summary is not None and not summary.empty:
        c2.metric('Step 2 rows avg', f"{summary['step2_rows'].dropna().mean():.0f}" if 'step2_rows' in summary else 'n/a')
        c3.metric('Winner targets', int(summary.get('actual_watched_core_winners', pd.Series(dtype=float)).fillna(0).sum()) if 'actual_watched_core_winners' in summary else 0)
        c4.metric('Winners found in Step 2', int(summary.get('winners_found_in_step2', pd.Series(dtype=float)).fillna(0).sum()) if 'winners_found_in_step2' in summary else 0)

    tabs = st.tabs(['Run summary','Support debug','Winner locations','Step 3 what-if + transition + Top-N','Best lockdown combo rollup','Affordable/Lockdown <= cap','Downloads'])
    with tabs[0]:
        st.dataframe(summary, use_container_width=True, height=360)
    with tabs[1]:
        outdir = Path(res['out_dir'])
        diag_files = sorted(outdir.glob('SUPPORT_99_DIAGNOSIS_*.csv'))
        summary_files = sorted(outdir.glob('SUPPORT_01_SIGNAL_SUMMARY_*.csv'))
        recompute_files = sorted(outdir.glob('SUPPORT_02_RECOMPUTED_COUNTS_*.csv'))
        audit_files = sorted(outdir.glob('SUPPORT_03_JOIN_AUDIT_ROWS_*.csv'))
        inv_file = outdir / 'SUPPORT_00_PROFILE_INVENTORY.csv'
        if inv_file.exists():
            st.markdown('**Profile/rule file inventory**')
            
            inv_df = safe_read_csv(inv_file)
            if inv_df.empty:
                st.warning('Profile inventory file was empty. This usually means the deployed repo is missing profiles/ or the inventory write found no readable files.')
            else:
                st.dataframe(inv_df, use_container_width=True, height=260)
        if diag_files:
            st.markdown('**Support diagnosis by date**')
            diag_frames = [safe_read_csv(f).assign(source_file=f.name) for f in diag_files if not safe_read_csv(f).empty]
            diag_all = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
            
            if diag_all.empty:
                st.info('No support diagnosis rows were written yet.')
            else:
                st.dataframe(diag_all, use_container_width=True, height=320)
        if summary_files:
            st.markdown('**Support signal summary before/after Step 2**')
            summary_frames = [safe_read_csv(f).assign(source_file=f.name) for f in summary_files if not safe_read_csv(f).empty]
            ssum = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
            
            if ssum.empty:
                st.info('No support summary rows were written yet.')
            else:
                st.dataframe(ssum, use_container_width=True, height=360)
        if recompute_files:
            st.markdown('**Saved vs recomputed support counts — first file preview**')
            
            rdf = safe_read_csv(recompute_files[0])
            if rdf.empty:
                st.info('Recomputed support file exists but is empty.')
            else:
                st.dataframe(rdf.head(500), use_container_width=True, height=360)
        if audit_files:
            st.markdown('**Row-level support join audit — first file preview**')
            q = st.text_input('Search support audit by stream/core/member/seed', '')
            adf = safe_read_csv(audit_files[0])
            if q:
                adf = adf[adf.astype(str).agg(' '.join, axis=1).str.lower().str.contains(q.lower(), na=False)]
            
            if adf.empty:
                st.info('Support join audit file exists but is empty.')
            else:
                st.dataframe(adf.head(1000), use_container_width=True, height=420)
    with tabs[2]:
        if loc is None or loc.empty:
            st.warning('No winner-location rows were found. Check whether your history contains winner dates, or upload a winner file for those dates.')
        else:
            q = st.text_input('Search winner locations by stream/core/member/seed/result', '')
            show = loc.copy()
            if q:
                mask = show.astype(str).agg(' '.join, axis=1).str.lower().str.contains(q.lower(), na=False)
                show = show[mask]
            st.dataframe(show, use_container_width=True, height=520)
    with tabs[3]:
        if what is None or what.empty:
            st.warning('No what-if matrix rows. This usually means no winners were matched in Step 2.')
        else:
            st.dataframe(what, use_container_width=True, height=520)
    with tabs[4]:
        combo_path = Path(res['out_dir']) / '03_BEST_STEP3_COMBOS_BY_CAPTURE_AND_PLAYCOUNT.csv'
        if combo_path.exists():
            combo = safe_read_csv(combo_path)
            if combo.empty:
                st.info('Combo rollup file exists but is empty.')
            else:
                st.dataframe(combo, use_container_width=True, height=520)
        else:
            st.info('No combo rollup created.')
    with tabs[5]:
        outdir = Path(res['out_dir'])
        cap_files = (sorted(outdir.glob('04_AFFORDABLE_COMBOS_AVG_ROWS_LE_*.csv'))
                     + sorted(outdir.glob('05_STRICT_COMBOS_MAX_ROWS_LE_*.csv'))
                     + sorted(outdir.glob('06_LOCKDOWN_CANDIDATES_*.csv'))
                     + sorted(outdir.glob('07_NEAR_MISS_CANDIDATES_*.csv')))
        if cap_files:
            for fp in cap_files:
                st.markdown(f'**{fp.name}**')
                try:
                    dfp = safe_read_csv(fp)
                    if dfp.empty:
                        st.info(f'{fp.name} exists but is empty.')
                    else:
                        st.dataframe(dfp, use_container_width=True, height=300)
                except Exception as e:
                    st.warning(f'Could not preview {fp.name}: {e}')
        else:
            st.info('No affordable/strict combo files were created. Usually this means no winners were matched in Step 2.')
    with tabs[6]:
        zip_path = Path(res['zip_path'])
        if zip_path.exists():
            st.download_button('Download full audit output ZIP', zip_path.read_bytes(), file_name=zip_path.name, mime='application/zip')
        outdir = Path(res['out_dir'])
        for name in ['00_AUDIT_RUN_SUMMARY.csv','01_WINNER_LOCATION_IN_STEP2_BUCKETS.csv','02_STEP3_FILTER_WHATIF_WINNER_SURVIVAL.csv','03_BEST_STEP3_COMBOS_BY_CAPTURE_AND_PLAYCOUNT.csv'] + [p.name for p in Path(res['out_dir']).glob('04_AFFORDABLE_COMBOS_AVG_ROWS_LE_*.csv')] + [p.name for p in Path(res['out_dir']).glob('05_STRICT_COMBOS_MAX_ROWS_LE_*.csv')] + [p.name for p in Path(res['out_dir']).glob('06_LOCKDOWN_CANDIDATES_*.csv')] + [p.name for p in Path(res['out_dir']).glob('07_NEAR_MISS_CANDIDATES_*.csv')]:
            p = outdir / name
            if p.exists():
                st.download_button(f'Download {name}', p.read_bytes(), file_name=name, mime='text/csv')
