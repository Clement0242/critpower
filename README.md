# critpower

Critical power and critical speed in Python — from laboratory tests or from
field data — with a designed PDF report.

```python
import critpower as cp

fit = cp.fit_critical([150, 270, 480, 840], [413, 355, 321, 301])   # durations (s), mean power (W)
print(fit.summary())          # CP 277.9 W (SEE 0.5 %), W' 20.4 kJ (SEE 1.7 %)
```

The same mathematics serves power (CP, W′) and running speed (CS, D′):
pass `kind="speed"`.

## What it does

| | |
|---|---|
| **Five models** | hyperbolic P = CP + W′/t (Hill 1993), linear work-time (Moritani et al. 1981), linear 1/time, 3-parameter (Morton 1996), omni-domain (Puchowicz, Baker & Clarke 2020). Standard errors and SEE % for every parameter. |
| **Model comparison** | `compare_models` ranks the fits by summed SEE % of CP and W′; models with fewer than two degrees of freedom left (3 parameters on 4 efforts) go last, since they pass through every point. |
| **3-min all-out test** | `all_out_test` (Vanhatalo et al. 2007; Pettitt et al. 2012 for running). |
| **Field data** | `mean_maximal` turns a recording (power file, GPS) into its best rolling means per duration in milliseconds; `best_of` combines sessions; `fit_field` fits a chosen window (default 2–20 min). Field efforts are not always maximal: the result is a lower bound. |
| **W′ balance** | `w_prime_balance`: differential (Clarke & Skiba 2013; Skiba et al. 2014) or integral model (Skiba et al. 2012). |
| **PDF report** | `critpower.report.critical_report`: key figures, automatic reading, intensity-duration curve, model table, predictions (sustainable intensity, time limit, race times for running), session W′ balance, method and references. French or English; your organisation and logo as options, nothing added by default. |

```python
from critpower.report import critical_report

models = cp.compare_models(durations, powers)
bal = cp.w_prime_balance(session_time, session_power, models[0].cp, models[0].w_prime)
critical_report(models[0], "report.pdf", compare=models, session=bal, session_intensity=session_power,
                athlete="A. Martin", body_mass=72, lang="fr", organisation="My club", logo="logo.png")
```

Example reports: [`docs/example_power_fr.pdf`](docs/example_power_fr.pdf) (laboratory, synthetic athlete) and
[`docs/example_speed_fr.pdf`](docs/example_speed_fr.pdf) (critical speed from a season of GNSS data), in English too.

## Example data

* `load_tte_example()` — **synthetic**: four time trials and an interval session generated from CP 280 W and
  W′ 20 kJ with noise. For examples and tests, not a real athlete.
* `load_soccermon_mms()` — **real**: mean-maximal speed curves of one player of an elite women's football team,
  276 sessions recorded at 10 Hz in 2020-2021, from **SoccerMon** (Midoglu et al. 2024, *Scientific Data*
  11:553; Zenodo 10033832, CC-BY 4.0). Only the curves are shipped (pseudonymous identifier of the dataset);
  `examples/build_soccermon_example.py` rebuilds them from the archives by HTTP range requests. Eight days with
  GNSS faults (speed pinned at 10 m/s for minutes) are left out by default.

On that player, critical speed over 2–20 min is 2.25 m/s (linear work-time, SEE 2.1 %) to 2.45 m/s
(hyperbolic): the model choice matters, which is why the report shows all of them.

## Install

```bash
pip install "critpower[report] @ git+https://github.com/Clement0242/critpower"
```

PDF reports use WeasyPrint (needs Pango; on Debian/Ubuntu `apt install libpango-1.0-0 libpangoft2-1.0-0`).
The report typeface is Jost (SIL Open Font License, shipped).

## En français

`critpower` calcule la puissance critique (CP, W′) ou la vitesse critique (CS, D′) à partir de tests de
laboratoire, d'un test « all-out » de 3 min ou de données de terrain (courbes des meilleures moyennes),
compare cinq modèles avec leurs erreurs standard, suit la réserve W′ pendant une séance et produit un
rapport PDF soigné, en français ou en anglais.

## Licence

MIT. Clément Verdier.
