# Data folder — read before adding anything

| Folder | Contents | Licence / rule | May Claude read it? |
|---|---|---|---|
| `open/pbc/` | Mayo Clinic PBC dataset (UCI id 878) | CC BY 4.0 | ✅ Yes |
| `open/mimic_iv_demo/` | MIMIC-IV Clinical Database **Demo** (100 patients) | Open (ODbL — verify on PhysioNet page) | ✅ Yes |
| `open/eicu_demo/` | eICU-CRD **Demo** | Open (ODbL — verify on PhysioNet page) | ✅ Yes |
| `restricted/mimic_iv/` | Full MIMIC-IV (credentialed) | PhysioNet Credentialed DUA | ❌ **NO** |
| `restricted/eicu/` | Full eICU-CRD (credentialed) | PhysioNet Credentialed DUA | ❌ **NO** |
| `processed/` | Derived cohorts/feature tables | Same rule as the source they came from | Only if derived from `open/` |

Credentialed data must not be sent to any online AI service. Code is developed against the
`open/*_demo` copies (identical schema), then **Atul runs it locally** against `restricted/`.
Only aggregate, non-identifying outputs (metrics, plots, counts ≥ 10) leave the machine.
