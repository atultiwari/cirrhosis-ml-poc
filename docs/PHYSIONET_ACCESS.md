# PhysioNet credentialed access — checklist (Atul)

1. [ ] Create account at https://physionet.org (use institutional email if possible).
2. [ ] Complete CITI course "Data or Specimens Only Research" (via PhysioNet's CITI link); download completion report.
3. [ ] Apply for **credentialed access** in PhysioNet profile; give a reference (supervisor / senior colleague
       who can confirm research role). Upload CITI certificate.
4. [ ] Once credentialed, sign the DUA on the project pages for **MIMIC-IV** (latest v3.x) and **eICU-CRD** (v2.0).
5. [ ] Download CSVs (wget command on each project page) to `data/restricted/mimic_iv/` and `data/restricted/eicu/`
       on an encrypted disk. Do this yourself in Terminal, not via Claude.
6. [ ] Re-read the DUA section on use with online/AI services. Rule for this project: no restricted rows
       ever go to Claude or any cloud LLM; Claude writes code against the demo data; you run it.
7. [ ] Meanwhile download the open **MIMIC-IV Clinical Database Demo** and **eICU-CRD Demo** into
       `data/open/mimic_iv_demo/` and `data/open/eicu_demo/` (confirm the licence on each page).

Record dates and version numbers here:
- Credentialed on: ____
- MIMIC-IV version: ____   eICU-CRD version: ____
