"""Designed PDF report of a critical power (or critical speed) assessment.

    from critpower.report import critical_report
    critical_report(fit, "report.pdf", athlete="A. Martin", test="4 time trials",
                    compare=cp.compare_models(d, p), session=wbal)

Needs the optional dependencies: ``pip install "critpower[report]"``.
Branding is the user's: ``organisation=`` and ``logo=`` put your own name and
logo on the report; nothing is added by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import __version__
from ..models import CriticalFit
from ..wbal import WBalance
from .engine import PALETTE, fig_to_uri, fmt, mpl_theme, render_pdf

TEMPLATES = Path(__file__).resolve().parent / "templates"

MODEL_NAMES = {
    "fr": {"hyperbolic": "Hyperbolique (2 param.)", "linear_work": "Linéaire travail-temps",
           "linear_inverse": "Linéaire 1/temps", "three_parameter": "3 paramètres (Morton)",
           "omni": "Omni-domaine (OmPD)"},
    "en": {"hyperbolic": "Hyperbolic (2-param.)", "linear_work": "Linear work-time",
           "linear_inverse": "Linear 1/time", "three_parameter": "3-parameter (Morton)",
           "omni": "Omni-domain (OmPD)"},
}

LABELS = {
    "fr": {
        "summary": "Synthèse", "reading": "Lecture", "curve": "Relation intensité-durée", "efforts": "Efforts retenus",
        "duration": "Durée", "measured": "Mesuré", "model": "Modèle", "residual": "Écart",
        "models": "Comparaison des modèles", "predictions": "Prédictions", "session": "Bilan W′ de la séance",
        "method": "Méthode", "sustainable": "Intensité tenable", "tte": "Temps limite", "target": "Cible",
        "distance": "Distance", "time": "Temps", "pace": "Allure", "min_bal": "Minimum de W′", "below": "Sous 25 %",
        "work_title": "Travail et durée", "best": "retenu",
        "cap_curve": "Efforts maximaux (points), modèle ajusté (trait) et valeur critique (pointillés). L'aire au-dessus "
                     "de la valeur critique est la réserve finie.",
        "cap_field": "Courbe des meilleures moyennes sur toutes les séances (gris) ; points rouges : durées utilisées "
                     "pour l'ajustement.",
        "cap_work": "Travail (ou distance) selon la durée : la pente est la valeur critique, l'ordonnée à l'origine la "
                    "réserve.",
        "cap_session": "Intensité de la séance (gris), valeur critique (pointillés) et réserve restante (bleu).",
        "power": "Puissance critique", "speed": "Vitesse critique",
    },
    "en": {
        "summary": "Summary", "reading": "Reading", "curve": "Intensity-duration relationship", "efforts": "Efforts used",
        "duration": "Duration", "measured": "Measured", "model": "Model", "residual": "Residual",
        "models": "Model comparison", "predictions": "Predictions", "session": "Session W′ balance",
        "method": "Method", "sustainable": "Sustainable intensity", "tte": "Time limit", "target": "Target",
        "distance": "Distance", "time": "Time", "pace": "Pace", "min_bal": "W′ minimum", "below": "Below 25 %",
        "work_title": "Work and duration", "best": "selected",
        "cap_curve": "Maximal efforts (dots), fitted model (line) and critical value (dashed). The area above the "
                     "critical value is the finite reserve.",
        "cap_field": "Best mean values over all sessions (grey); red dots: durations used for the fit.",
        "cap_work": "Work (or distance) against duration: the slope is the critical value, the intercept the reserve.",
        "cap_session": "Session intensity (grey), critical value (dashed) and remaining reserve (blue).",
        "power": "Critical power", "speed": "Critical speed",
    },
}

REFERENCES = [
    "Monod H. & Scherrer J. (1965). The work capacity of a synergic muscular group. Ergonomics 8, 329–338.",
    "Moritani T. et al. (1981). Critical power as a measure of physical work capacity and anaerobic threshold. "
    "Ergonomics 24, 339–350.",
    "Hill D.W. (1993). The critical power concept: a review. Sports Med 16, 237–254.",
    "Morton R.H. (1996). A 3-parameter critical power model. Ergonomics 39, 611–619.",
    "Jones A.M. & Vanhatalo A. (2017). The 'critical power' concept: applications to sports performance with a "
    "focus on intermittent high-intensity exercise. Sports Med 47 (Suppl 1), 65–78.",
    "Muniz-Pumares D. et al. (2019). Methodological approaches and related challenges associated with the "
    "determination of critical power and curvature constant. J Strength Cond Res 33, 584–596.",
    "Puchowicz M.J., Baker J. & Clarke D.C. (2020). Development and field validation of an omni-domain "
    "power-duration model. J Sports Sci 38, 801–813.",
    "Skiba P.F. et al. (2012, 2014); Clarke D.C. & Skiba P.F. (2013). W′ balance models. Med Sci Sports Exerc "
    "44, 1526–1532; 46, 1433–1440; PLoS One 8, e65473.",
]
FIELD_REFS = [
    "Vanhatalo A. et al. (2007). Determination of critical power using a 3-min all-out cycling test. Med Sci "
    "Sports Exerc 39, 548–555.",
    "Smyth B. & Muniz-Pumares D. (2020). Calculation of critical speed from raw training data in recreational "
    "marathon runners. Med Sci Sports Exerc 52, 2637–2645.",
]


def _pace(v: float) -> str:
    s = int(round(1000.0 / v))
    return f"{s // 60}:{s % 60:02d}"


def _dur(s: float, lang: str) -> str:
    if not np.isfinite(s):
        return "∞"
    s = int(round(s))
    if s < 120:
        return f"{s} s"
    return f"{s // 60} min {s % 60:02d}"


def _figures(fit: CriticalFit, curve, session: WBalance | None, session_intensity, lang: str, L) -> dict:
    import matplotlib.pyplot as plt

    mpl_theme()
    P = PALETTE
    u = fit.units[0]
    out = {}

    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    lo = min(fit.duration.min(), min(curve) if curve else np.inf) * 0.8
    hi = max(fit.duration.max(), max(curve) if curve else 0) * 1.25
    tt = np.logspace(np.log10(max(lo, 1)), np.log10(hi), 300)
    if fit.model in ("hyperbolic", "linear_work", "linear_inverse"):
        # 2-parameter models go to infinity at t → 0: draw them where they apply.
        tt = tt[tt >= 0.6 * fit.duration.min()]
    pred = fit.predict(tt)
    ax.fill_between(tt, fit.cp, pred, where=pred > fit.cp, color=P["gold_light"], alpha=.35, lw=0)
    if curve:
        d = np.array(sorted(curve))
        ax.plot(d, [curve[k] for k in d], "o-", color=P["mut"], ms=3, lw=1, alpha=.7)
    ax.plot(tt, pred, color=P["navy"], lw=2.2)
    ax.axhline(fit.cp, ls="--", color=P["navy"], lw=1)
    ax.plot(fit.duration, fit.intensity, "o", color=P["red"], ms=6, zorder=5)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ymax = max(fit.intensity.max(), max(curve.values()) if curve else 0, pred.max())
    ax.set_ylim(min(fit.cp, fit.intensity.min()) * 0.9, ymax * 1.06)
    ticks = [t for t in (1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600) if lo <= t <= hi]
    ax.set_xticks(ticks, [f"{t}s" if t < 60 else f"{t // 60}min" for t in ticks])
    ax.set(xlabel=L["duration"], ylabel=f"{'P' if fit.kind == 'power' else 'v'} ({u})")
    ax.text(tt[-1], fit.cp, f" {'CP' if fit.kind == 'power' else 'CS'} {fmt(fit.cp, 2 if fit.kind == 'speed' else 0, lang)}",
            va="bottom", ha="right", color=P["navy"], fontsize=8)
    out["fig_curve"] = fig_to_uri(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    w = fit.intensity * fit.duration
    ax.plot(fit.duration, w / (1000 if fit.kind == "power" else 1), "o", color=P["red"], ms=5)
    x = np.linspace(0, fit.duration.max() * 1.1, 50)
    ax.plot(x, (fit.cp * x + fit.w_prime) / (1000 if fit.kind == "power" else 1), color=P["navy"], lw=1.8)
    ax.set(xlabel="t (s)", ylabel="W (kJ)" if fit.kind == "power" else "d (m)", xlim=(0, None), ylim=(0, None))
    out["fig_work"] = fig_to_uri(fig)

    if session is not None:
        fig, ax = plt.subplots(figsize=(7.4, 2.4))
        tm = session.time / 60
        if session_intensity is not None:
            ax.plot(tm, session_intensity, color=P["mut"], lw=.6, alpha=.6)
        ax.axhline(fit.cp, ls="--", color=P["navy"], lw=1)
        ax.set(xlabel="t (min)", ylabel=f"{'P' if fit.kind == 'power' else 'v'} ({u})")
        ax2 = ax.twinx()
        ax2.plot(tm, 100 * session.balance / session.w_prime, color=P["navy"], lw=2)
        ax2.axhline(25, color=P["red"], lw=.8, ls=":")
        ax2.set_ylabel("W'bal (%)", color=P["navy"])
        ax2.set_ylim(min(0, 100 * session.minimum / session.w_prime - 5), 105)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        out["fig_session"] = fig_to_uri(fig)
    return out


def _reading(fit: CriticalFit, compare, body_mass, field, session, lang, see_limits) -> list[str]:
    f = lambda x, d=1: fmt(x, d, lang)
    u, c = fit.units
    pw = fit.kind == "power"
    names = ("CP", "W′") if pw else ("CS", "D′")
    cp_txt = f(fit.cp, 0 if pw else 2)
    res_txt = f(fit.w_prime / 1000, 1) + " kJ" if pw else f(fit.w_prime, 0) + " m"
    see = fit.see_pct
    lines = []
    if lang == "fr":
        lines.append(f"{'Puissance' if pw else 'Vitesse'} critique <b>{names[0]} = {cp_txt} {u}</b>"
                     + (f" ({f(fit.cp / body_mass, 2)} W/kg)" if pw and body_mass else "")
                     + (f", soit une allure de {_pace(fit.cp)} min/km" if not pw else "")
                     + f" ; réserve au-dessus <b>{names[1]} = {res_txt}</b>.")
        lines.append(f"{names[0]} est la frontière entre les intensités qu'on stabilise et celles qui mènent à "
                     f"l'épuisement : au-dessus, {names[1]} se vide d'autant plus vite que l'écart est grand.")
        if see:
            q = ("précise" if see.get("cp", 99) <= see_limits[0] and see.get("w_prime", 99) <= see_limits[1]
                 else "à confirmer")
            lines.append(f"Erreur standard : {f(see.get('cp', np.nan))} % sur {names[0]}, "
                         f"{f(see.get('w_prime', np.nan))} % sur {names[1]} — estimation <b>{q}</b> "
                         f"(repères indicatifs : ≤ {see_limits[0]:g} % sur {names[0]}, ≤ {see_limits[1]:g} % sur {names[1]}).")
        if compare and len(compare) > 1:
            cps = [m.cp for m in compare]
            lines.append(f"Selon le modèle, {names[0]} va de {f(min(cps), 0 if pw else 2)} à "
                         f"{f(max(cps), 0 if pw else 2)} {u} : l'écart dit la dépendance au choix du modèle.")
        if field:
            lines.append("Données de terrain : les meilleures moyennes ne sont pas toujours des efforts maximaux, "
                         "la valeur obtenue est donc une <b>borne basse</b> de la capacité réelle.")
        if session is not None:
            lines.append(f"Pendant la séance, la réserve est descendue à <b>{f(session.minimum_pct, 0)} %</b> et "
                         f"est restée {_dur(session.time_below(0.25), lang)} sous 25 %.")
    else:
        lines.append(f"Critical {'power' if pw else 'speed'} <b>{names[0]} = {cp_txt} {u}</b>"
                     + (f" ({f(fit.cp / body_mass, 2)} W/kg)" if pw and body_mass else "")
                     + (f", a pace of {_pace(fit.cp)} min/km" if not pw else "")
                     + f"; reserve above it <b>{names[1]} = {res_txt}</b>.")
        lines.append(f"{names[0]} separates the intensities that can be stabilised from those that lead to "
                     f"exhaustion: above it, {names[1]} is depleted the faster the further above one goes.")
        if see:
            q = ("precise" if see.get("cp", 99) <= see_limits[0] and see.get("w_prime", 99) <= see_limits[1]
                 else "to be confirmed")
            lines.append(f"Standard error: {f(see.get('cp', np.nan))} % on {names[0]}, "
                         f"{f(see.get('w_prime', np.nan))} % on {names[1]} — estimate <b>{q}</b> "
                         f"(indicative guides: ≤ {see_limits[0]:g} % on {names[0]}, ≤ {see_limits[1]:g} % on {names[1]}).")
        if compare and len(compare) > 1:
            cps = [m.cp for m in compare]
            lines.append(f"Across models, {names[0]} ranges from {f(min(cps), 0 if pw else 2)} to "
                         f"{f(max(cps), 0 if pw else 2)} {u}: the spread shows how much the model choice matters.")
        if field:
            lines.append("Field data: best mean values are not always maximal efforts, so the value is a "
                         "<b>lower bound</b> of the true capacity.")
        if session is not None:
            lines.append(f"During the session, the reserve fell to <b>{f(session.minimum_pct, 0)} %</b> and stayed "
                         f"{_dur(session.time_below(0.25), lang)} below 25 %.")
    return lines


def _method(fit, field, session, lang) -> list[str]:
    pw = fit.kind == "power"
    if lang == "fr":
        p = ["La relation entre l'intensité d'un effort maximal et sa durée est hyperbolique : P = CP + W′/t "
             "(Monod &amp; Scherrer 1965 ; Hill 1993). Les formes linéaires (travail-temps, intensité-1/temps) "
             "donnent les mêmes paramètres par régression ; le modèle à 3 paramètres (Morton 1996) ajoute une "
             "intensité maximale finie à t → 0." + ("" if pw else " Pour la course, la même relation lie la "
                                                    "vitesse à la durée : CS (m/s) et D′ (m)."),
             "Chaque paramètre est accompagné de son erreur standard en % (SEE) ; par défaut, le modèle retenu est "
             "celui qui minimise la somme des SEE de CP et W′, parmi ceux qui gardent au moins deux degrés de "
             "liberté. Les choix de protocole et de modèle sont discutés par Muniz-Pumares et al. (2019) ; les "
             "efforts doivent durer entre 2 et 15-20 min environ pour que la relation s'applique."]
        if field:
            p.append("Données de terrain : pour chaque séance, la meilleure moyenne glissante est calculée pour chaque "
                     "durée (somme cumulée sur une grille régulière) ; les meilleures valeurs de toutes les séances "
                     "forment la courbe, ajustée entre les durées choisies (Smyth &amp; Muniz-Pumares 2020).")
        if session is not None:
            p.append(f"Bilan W′ : modèle {'différentiel (Clarke &amp; Skiba 2013 ; Skiba et al. 2014)' if session.model == 'differential' else 'intégral (Skiba et al. 2012)'}. "
                     "Au-dessus de la valeur critique la réserve se vide de l'excès ; en dessous, elle se reconstitue "
                     "d'autant plus vite que l'intensité est basse. Modèle établi en cyclisme : à lire comme un ordre "
                     "de grandeur.")
    else:
        p = ["The relationship between the intensity of a maximal effort and its duration is hyperbolic: "
             "P = CP + W′/t (Monod &amp; Scherrer 1965; Hill 1993). The linear forms (work-time, intensity-1/time) "
             "give the same parameters by regression; the 3-parameter model (Morton 1996) adds a finite maximal "
             "intensity at t → 0." + ("" if pw else " For running, the same relationship links speed to duration: "
                                      "CS (m/s) and D′ (m)."),
             "Every parameter comes with its standard error in % (SEE); by default, the selected model is the one "
             "minimising the summed SEE of CP and W′, among those keeping at least two degrees of freedom. "
             "Protocol and model choices are discussed by Muniz-Pumares et al. (2019); efforts should last between "
             "about 2 and 15-20 min for the relationship to hold."]
        if field:
            p.append("Field data: for each session, the best rolling mean is computed for each duration (cumulative "
                     "sum on a regular grid); the best values across sessions form the curve, fitted between the "
                     "chosen durations (Smyth &amp; Muniz-Pumares 2020).")
        if session is not None:
            p.append(f"W′ balance: {'differential model (Clarke &amp; Skiba 2013; Skiba et al. 2014)' if session.model == 'differential' else 'integral model (Skiba et al. 2012)'}. "
                     "Above the critical value the reserve is depleted by the excess; below it, it recovers the "
                     "faster the lower the intensity. Established in cycling: read it as an order of magnitude.")
    return p


def critical_report(
    fit: CriticalFit,
    out: str | Path,
    *,
    athlete: str | None = None,
    test: str | None = None,
    test_date: str | None = None,
    compare: list[CriticalFit] | None = None,
    curve: dict[float, float] | None = None,
    session: WBalance | None = None,
    session_intensity: np.ndarray | None = None,
    body_mass: float | None = None,
    lang: str = "fr",
    organisation: str | None = None,
    logo: str | Path | None = None,
    prediction_durations: tuple[float, ...] = (120, 180, 300, 600, 1200),
    prediction_distances: tuple[float, ...] = (800, 1500, 3000, 5000),
    see_limits: tuple[float, float] = (5.0, 10.0),
) -> Path:
    """Write a designed PDF report of a critical power or speed fit; return its path.

    ``compare`` (from :func:`compare_models`) adds the model table; ``curve``
    (a mean-maximal curve) marks the report as field data and draws the whole
    curve; ``session`` (from :func:`w_prime_balance`, with ``session_intensity``)
    adds the W′ balance of a session.
    """
    if lang not in LABELS:
        raise ValueError("lang must be 'fr' or 'en'")
    L = LABELS[lang]
    pw = fit.kind == "power"
    u = fit.units[0]
    dig = 0 if pw else 2
    efforts = [(_dur(d, lang), fmt(i, dig + (1 if pw else 0), lang), fmt(m, dig + (1 if pw else 0), lang),
                fmt(r, dig + 1, lang)) for d, i, m, r in zip(fit.duration, fit.intensity, fit.predict(fit.duration),
                                                              fit.residuals)]
    pred_dur = [(_dur(d, lang), fmt(float(fit.predict([d])[0]), dig, lang)
                 + ("" if pw else f" · {_pace(float(fit.predict([d])[0]))}")) for d in prediction_durations]
    targets = [fit.cp * k for k in (1.05, 1.10, 1.20, 1.35)]
    pred_tte = [(f"{fmt(x, dig, lang)} {u} (+{round((x / fit.cp - 1) * 100)} %)",
                 _dur(float(fit.time_to_exhaustion([x])[0]), lang)) for x in targets]
    pred_dist = []
    if not pw and not curve:  # race times make no sense from intermittent field data
        for d in prediction_distances:
            t = (d - fit.w_prime) / fit.cp if d > fit.w_prime else np.nan
            if np.isfinite(t) and t >= 120:
                pred_dist.append((f"{fmt(d, 0, lang)} m", _dur(t, lang), _pace(d / t)))
    rows = []
    for m in compare or []:
        s = m.see_pct
        rows.append({"name": MODEL_NAMES[lang][m.model], "cp": fmt(m.cp, dig + 1, lang),
                     "wp": fmt(m.w_prime / (1000 if pw else 1), 2 if pw else 0, lang),
                     "pmax": fmt(m.pmax, dig, lang) if m.pmax else "—",
                     "see_cp": fmt(s.get("cp", np.nan), 1, lang), "see_wp": fmt(s.get("w_prime", np.nan), 1, lang),
                     "r2": fmt(m.r2, 4, lang), "best": m.model == fit.model})
    names = ("CP", "W′") if pw else ("CS", "D′")
    see = fit.see_pct
    tiles = [{"val": fmt(fit.cp, dig, lang), "unit": u, "lab": names[0], "accent": True,
              "sub": (f"{fmt(fit.cp / body_mass, 2, lang)} W/kg" if pw and body_mass else
                      f"{fmt(fit.cp * 3.6, 1, lang)} km/h · {_pace(fit.cp)}/km" if not pw else
                      f"SEE {fmt(see.get('cp', np.nan), 1, lang)} %")},
             {"val": fmt(fit.w_prime / 1000, 1, lang) if pw else fmt(fit.w_prime, 0, lang), "unit": "kJ" if pw else "m",
              "lab": names[1], "sub": f"SEE {fmt(see.get('w_prime', np.nan), 1, lang)} %" if see else ""}]
    if fit.pmax:
        tiles.append({"val": fmt(fit.pmax, dig, lang), "unit": u, "lab": "Pmax" if pw else "Vmax", "sub": ""})
    tiles.append({"val": fmt(fit.r2, 3, lang), "unit": "", "lab": "R²",
                  "sub": f"{fit.duration.size} {'efforts' if lang == 'en' else 'efforts'}"})
    if session is not None:
        tiles_s = [{"val": fmt(session.minimum_pct, 0, lang), "unit": "%", "lab": L["min_bal"], "accent": True,
                    "sub": f"{fmt(session.minimum / (1000 if pw else 1), 1 if pw else 0, lang)} {'kJ' if pw else 'm'}"},
                   {"val": _dur(session.time_below(0.25), lang), "unit": "", "lab": L["below"], "sub": ""}]
    else:
        tiles_s = []
    title = L["power"] if pw else L["speed"]
    ctx = {
        "lang": lang, "L": L, "fit": fit, "title": title,
        "title_html": ("Puissance <em>critique</em>" if pw else "Vitesse <em>critique</em>") if lang == "fr" else
                      ("Critical <em>power</em>" if pw else "Critical <em>speed</em>"),
        "kicker": ("Endurance · relation intensité-durée" if lang == "fr" else "Endurance · intensity-duration"),
        "athlete": athlete, "test": test, "test_date": test_date,
        "extra_meta": ({("Masse" if lang == "fr" else "Body mass"): f"{fmt(body_mass, 1, lang)} kg"} if body_mass else {}),
        "organisation": organisation, "logo": logo, "tool": "critpower", "version": __version__,
        "tiles": tiles, "tiles_s": tiles_s, "efforts": efforts, "models": rows, "pred_dur": pred_dur,
        "pred_tte": pred_tte, "pred_dist": pred_dist, "field": bool(curve), "session": session,
        "reading": _reading(fit, compare, body_mass, bool(curve), session, lang, see_limits),
        "method": _method(fit, bool(curve), session, lang),
        "references": REFERENCES + (FIELD_REFS if curve else []),
        "disclaimer": ("Les valeurs dépendent de la maximalité des efforts et de la qualité de la mesure ; elles "
                       "s'interprètent avec le contexte de l'athlète." if lang == "fr" else
                       "Values depend on the efforts being maximal and on measurement quality; read them with the "
                       "athlete's context in mind."),
        **_figures(fit, curve, session, session_intensity, lang, L),
    }
    return render_pdf(TEMPLATES, "critical.html", ctx, out)


__all__ = ["critical_report"]
