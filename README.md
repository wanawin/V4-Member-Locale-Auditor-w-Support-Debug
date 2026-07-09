# Member Location Auditor V4.3 — Support Display Fix

This build fixes the Streamlit crash caused by attempting to preview an empty CSV diagnostic file.

Use this package, not V4/V4.1/V4.2, for the support-count debug run. The app title must show **V4.3**.

## Deploy checklist
Upload/replace the full folder contents in GitHub/Streamlit:

- app.py
- audit_engine.py
- daily_ladder_engine.py
- requirements.txt
- profiles/
- IN/
- outputs/

Do not keep old app.py from V4 or V4.1 in the repo.

## First run
- Start date: 2026-06-19
- End date: 2026-06-19
- Upload history through 2026-06-18
- Upload 06/19 winners file or use included sample
- Exclude AZ/MD: checked
- Step 2 scope: watched8_all_members
- Bucket basis: final_x15_positive
- Stream gate: 50
- Play cap: 50

## Key outputs
- SUPPORT_00_PROFILE_INVENTORY.csv
- SUPPORT_01_SIGNAL_SUMMARY_<date>.csv
- SUPPORT_02_RECOMPUTED_COUNTS_<date>.csv
- SUPPORT_03_JOIN_AUDIT_ROWS_<date>.csv
- SUPPORT_99_DIAGNOSIS_<date>.csv
