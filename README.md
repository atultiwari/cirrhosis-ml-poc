# cirrhosis-ml-poc

Proof-of-concept ML prediction work ahead of an AI study in liver cirrhosis & portal hypertension
at the Dept of Gastroenterology, RNT Medical College, Udaipur.

| Track | Data | Question |
|---|---|---|
| **A** | Mayo Clinic PBC (UCI 878, open, CC BY 4.0) | Reusable survival-modelling pipeline; ML vs Mayo risk score |
| **B** | MIMIC-IV (dev + temporal) → eICU-CRD (external) — credentialed | ICU cirrhosis in-hospital / 28-day mortality and AKI vs MELD-Na, MELD 3.0, CLIF-C |

- **Plan:** `docs/PLAN.md`
- **Start in Claude Code:** `docs/STARTER_PROMPT.md`
- **Rules for Claude (incl. data governance):** `CLAUDE.md`
- **PhysioNet access checklist:** `docs/PHYSIONET_ACCESS.md`

⚠️ Credentialed MIMIC-IV / eICU data stays in `data/restricted/`, is never committed, and is never shown to any AI assistant.
