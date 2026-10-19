"""
Simulateur interactif : autocall et delta hedging
Lancer en local : streamlit run app.py

Le moteur reprend la logique de autocall.py et delta_hedge.py, en version
vectorisée (numpy) pour que les résultats se mettent à jour instantanément :
- autocall : on ne simule que les dates d'observation annuelles, ce qui suffit
  pour un produit observé une fois par an et donne la même loi que la
  simulation jour par jour ;
- delta hedge : toutes les trajectoires sont simulées en même temps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

NOMINAL = 100

st.set_page_config(page_title="Simulateur autocall", layout="wide")


# =====================================================================
# Moteurs de calcul
# =====================================================================

@st.cache_data
def simuler_autocall(r, q, sigma, br, bp, T, N, seed=42):
    """Simule N trajectoires aux T dates d'observation annuelles.
    Rend, pour chaque trajectoire, l'année de rappel (0 si jamais rappelé)
    et la performance finale."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, T))
    log_perf = np.cumsum((r - q - 0.5 * sigma**2) + sigma * Z, axis=1)
    perf = np.exp(log_perf)                       # performance à chaque date, base 1

    annee_rappel = np.zeros(N, dtype=int)
    for k in range(T):
        rappel = (perf[:, k] >= br) & (annee_rappel == 0)
        annee_rappel[rappel] = k + 1
    return annee_rappel, perf[:, -1]


def prix_et_pente(annee_rappel, perf_finale, r, bp, T):
    """Le prix est une droite en fonction du coupon : prix = prix_0 + c x pente.
    Rend prix_0 (coupon nul) et la pente, calculés sur les mêmes trajectoires."""
    rappel = annee_rappel > 0
    actu_rappel = np.exp(-r * annee_rappel[rappel])

    flux_0 = np.zeros(len(annee_rappel))
    flux_0[rappel] = NOMINAL * actu_rappel
    fin = ~rappel
    remboursement = np.where(perf_finale[fin] >= bp, NOMINAL, NOMINAL * perf_finale[fin])
    flux_0[fin] = remboursement * np.exp(-r * T)

    coef = np.zeros(len(annee_rappel))
    coef[rappel] = NOMINAL * annee_rappel[rappel] * actu_rappel
    return flux_0.mean(), coef.mean(), flux_0, coef


@st.cache_data
def trajectoires_exemple(r, q, sigma, T, nb=15, seed=7):
    """Quelques trajectoires journalières, uniquement pour l'affichage."""
    rng = np.random.default_rng(seed)
    n = 252 * T
    dt = 1 / 252
    Z = rng.standard_normal((nb, n))
    log_S = np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    return np.hstack([np.ones((nb, 1)), np.exp(log_S)]) * NOMINAL


def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


@st.cache_data
def simuler_hedge(sigma_real, sigma_impl, nb, n=252, K=100, seed=0):
    """P&L d'un vendeur de call couvert en delta chaque jour, sur nb trajectoires."""
    rng = np.random.default_rng(seed)
    dt = 1 / n
    Z = rng.standard_normal((nb, n))
    log_S = np.cumsum(-0.5 * sigma_real**2 * dt + sigma_real * np.sqrt(dt) * Z, axis=1)
    S = np.hstack([np.ones((nb, 1)), np.exp(log_S)]) * 100

    def delta(S_t, T_restant):
        d1 = (np.log(S_t / K) + 0.5 * sigma_impl**2 * T_restant) / (sigma_impl * np.sqrt(T_restant))
        return norm.cdf(d1)

    cash = bs_call(100, K, 1.0, 0, sigma_impl) * np.ones(nb)
    shares = delta(S[:, 0], 1.0)
    cash -= shares * S[:, 0]
    for i in range(1, n):
        nouveau = delta(S[:, i], 1.0 - i * dt)
        cash -= (nouveau - shares) * S[:, i]
        shares = nouveau
    cash += shares * S[:, n]
    cash -= np.maximum(S[:, n] - K, 0)
    return cash


# =====================================================================
# Interface
# =====================================================================

st.title("Simulateur : autocall et delta hedging")
st.caption("Pricing Monte Carlo en univers risque-neutre. Modèle simplifié : "
           "vol, taux et dividendes constants, sans smile ni risque de crédit de l'émetteur.")

onglet_ac, onglet_dh, onglet_methode = st.tabs(["Autocall", "Delta hedging", "Méthode"])

# ------------------------------- Autocall -------------------------------
with onglet_ac:
    col_param, col_res = st.columns([1, 3])

    with col_param:
        st.subheader("Marché")
        sigma = st.slider("Volatilité (%)", 5.0, 50.0, 20.6, 0.1) / 100
        r = st.slider("Taux sans risque (%)", -1.0, 6.0, 2.5, 0.1) / 100
        q = st.slider("Rendement du dividende (%)", 0.0, 8.0, 3.0, 0.1) / 100

        st.subheader("Produit")
        T = st.slider("Maturité (années)", 1, 10, 5)
        br = st.slider("Barrière de rappel (%)", 70, 120, 100) / 100
        bp = st.slider("Barrière de protection (%)", 30, 100, 60) / 100
        coupon = st.slider("Coupon annuel (%)", 0.0, 25.0, 11.5, 0.1) / 100
        marge = st.slider("Marge et frais de la banque (points)", 0.0, 8.0, 2.0, 0.5)
        N = st.select_slider("Nombre de trajectoires", [10_000, 50_000, 100_000, 200_000], 100_000)

    annee_rappel, perf_finale = simuler_autocall(r, q, sigma, br, bp, T, N)
    prix_0, pente, flux_0, coef = prix_et_pente(annee_rappel, perf_finale, r, bp, T)
    prix = prix_0 + coupon * pente
    erreur = np.std(flux_0 + coupon * coef) / np.sqrt(N)
    coupon_100 = (100 - prix_0) / pente
    coupon_marge = (100 - marge - prix_0) / pente

    rappel = annee_rappel > 0
    perte = (~rappel) & (perf_finale < bp)
    duree_moyenne = np.where(rappel, annee_rappel, T).mean()

    with col_res:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Prix du produit", f"{prix:.2f}", help=f"Erreur Monte Carlo : ± {erreur:.2f}")
        m2.metric("Coupon juste (prix = 100)", f"{coupon_100 * 100:.2f} %")
        m3.metric(f"Coupon offert (marge {marge:g} pts)", f"{coupon_marge * 100:.2f} %")
        m4.metric("Probabilité de perte", f"{perte.mean() * 100:.1f} %")
        m5.metric("Durée de vie moyenne", f"{duree_moyenne:.1f} ans")

        g1, g2 = st.columns(2)

        with g1:
            st.markdown("**La vie du produit** : probabilité de chaque issue")
            issues = [f"Rappel année {k}" for k in range(1, T + 1)] + ["Capital protégé", "Perte en capital"]
            probas = [np.mean(annee_rappel == k) for k in range(1, T + 1)]
            probas += [np.mean((~rappel) & (perf_finale >= bp)), perte.mean()]
            couleurs = ["#2e7d32"] * T + ["#9e9e9e", "#c62828"]
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.barh(issues[::-1], np.array(probas[::-1]) * 100, color=couleurs[::-1])
            ax.set_xlabel("Probabilité (%)")
            for i, p in enumerate(probas[::-1]):
                ax.text(p * 100 + 0.5, i, f"{p * 100:.1f} %", va="center", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)

        with g2:
            st.markdown("**Trajectoires simulées** et barrières")
            chemins = trajectoires_exemple(r, q, sigma, T)
            fig, ax = plt.subplots(figsize=(6, 3.6))
            jours = np.arange(chemins.shape[1]) / 252
            for c in chemins:
                ax.plot(jours, c, lw=0.8, alpha=0.7)
            ax.axhline(br * 100, color="#2e7d32", ls="--", lw=1.2, label="Barrière de rappel")
            ax.axhline(bp * 100, color="#c62828", ls="--", lw=1.2, label="Barrière de protection")
            for k in range(1, T + 1):
                ax.axvline(k, color="#bdbdbd", lw=0.6)
            ax.set_xlabel("Années (lignes grises : dates d'observation)")
            ax.set_ylabel("Indice")
            ax.legend(fontsize=7, loc="upper left")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("**Sensibilité du coupon juste à la volatilité** (autres paramètres inchangés)")
        vols = np.arange(0.10, 0.41, 0.05)
        coupons_vol = []
        for s in vols:
            ar, pf = simuler_autocall(r, q, s, br, bp, T, 50_000)
            p0, pe, _, _ = prix_et_pente(ar, pf, r, bp, T)
            coupons_vol.append((100 - p0) / pe * 100)
        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.plot(vols * 100, coupons_vol, marker="o", color="#1565c0")
        ax.axvline(sigma * 100, color="#9e9e9e", ls=":", lw=1, label="Vol choisie")
        for v, c in zip(vols * 100, coupons_vol):
            ax.annotate(f"{c:.1f} %", (v, c), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8)
        ax.set_xlabel("Volatilité (%)")
        ax.set_ylabel("Coupon juste (%)")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("**Scénarios pour le client** (avec le coupon choisi)")
        st.table(pd.DataFrame({
            "Scénario": ["Rappel en année 1", f"Rappel en année {min(3, T)}",
                         f"Fin entre {bp * 100:.0f} % et {br * 100:.0f} %", "Indice final à 50 %"],
            "Le client reçoit": [f"{NOMINAL * (1 + coupon):.1f}",
                                 f"{NOMINAL * (1 + coupon * min(3, T)):.1f}",
                                 f"{NOMINAL:.0f}",
                                 f"{NOMINAL * 0.5:.0f}" if 0.5 < bp else f"{NOMINAL:.0f}"],
        }))

# ---------------------------- Delta hedging -----------------------------
with onglet_dh:
    st.markdown("Une banque vend un call à la monnaie (strike 100, 1 an) et se couvre en delta chaque jour. "
                "Son P&L dépend de l'écart entre la vol à laquelle elle a vendu et la vol réalisée.")
    c1, c2, c3 = st.columns(3)
    sigma_impl = c1.slider("Vol implicite de vente (%)", 5, 50, 20) / 100
    sigma_real = c2.slider("Vol réalisée (%)", 5, 50, 30) / 100
    nb = c3.select_slider("Nombre de trajectoires", [500, 1000, 2000, 5000], 2000)

    pnls = simuler_hedge(sigma_real, sigma_impl, nb)
    theorie = bs_call(100, 100, 1, 0, sigma_impl) - bs_call(100, 100, 1, 0, sigma_real)

    d1_, d2_, d3_ = st.columns(3)
    d1_.metric("P&L moyen simulé", f"{pnls.mean():.2f}")
    d2_.metric("Théorie : écart de prime", f"{theorie:.2f}")
    d3_.metric("Écart-type du P&L", f"{pnls.std():.2f}")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(pnls, bins=60, color="#1565c0", alpha=0.8)
    ax.axvline(pnls.mean(), color="black", lw=1.2, label="P&L moyen")
    ax.axvline(theorie, color="#c62828", ls="--", lw=1.2, label="Théorie")
    ax.set_xlabel("P&L du vendeur couvert")
    ax.set_ylabel("Nombre de trajectoires")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

# -------------------------------- Méthode -------------------------------
with onglet_methode:
    st.markdown("""
**Simulation risque-neutre.** L'indice dérive au rythme r − q : c'est le monde où le prix d'un produit
est l'espérance actualisée de ses flux.

**Payoff de l'autocall.** À chaque date d'observation annuelle, si l'indice est au-dessus de la barrière
de rappel, le client reçoit 100 + 100 × coupon × nombre d'années. Sinon, à l'échéance seulement :
100 au-dessus de la barrière de protection, 100 × performance finale en dessous.

**Coupon juste.** Avec des trajectoires fixées, les dates de rappel ne dépendent pas du coupon :
le prix est une droite en fonction du coupon, d'où le coupon qui donne un prix de 100.

**Delta hedging.** Le vendeur couvert achète après une hausse et revend après une baisse :
il est short gamma, donc il perd si la vol réalisée dépasse la vol implicite de vente.

**Limites.** Vol constante (pas de smile ni de skew), taux et dividendes constants,
pas de spread de crédit de l'émetteur.
""")
