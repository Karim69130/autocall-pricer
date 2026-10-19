"""
Étude du delta hedging d'un call vendu
======================================

Une banque vend un call à la monnaie (strike 100, 1 an) au prix Black-Scholes
calculé avec une vol implicite de 20 %, puis se couvre en delta chaque jour.
On simule 1 000 trajectoires avec une vol réalisée de 10 %, 20 % et 30 %.

Résultat : le P&L moyen du vendeur couvert vaut environ +4 / 0 / -4,
soit l'écart de prime Black-Scholes entre la vol implicite et la vol réalisée.
La couverture en delta élimine le risque de direction mais laisse un pari sur la vol.

Contenu du fichier :
  1. Black-Scholes (prix et delta d'un call, r = 0)
  2. Simulation de trajectoires
  3. Stratégie de couverture
  4. Tests de validation
  5. Étude Monte Carlo et histogrammes
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# =====================================================================
# 1. BLACK-SCHOLES
# =====================================================================

def calcul_d1(S, K, T, r, sigma):
    """d1 de la formule de Black-Scholes."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call(S, K, T, r, sigma):
    """Prix Black-Scholes d'un call européen."""
    d1 = calcul_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    """Delta d'un call européen : N(d1)."""
    d1 = calcul_d1(S, K, T, r, sigma)
    return norm.cdf(d1)


# =====================================================================
# 2. SIMULATION DE TRAJECTOIRES
# =====================================================================

def simule_path(S0, sigma, n):
    """Simule n jours de prix sur un an (dt = 1/n), sans dérive (r = 0).
    Le terme -0,5 sigma^2 garde le prix moyen égal au prix de départ.
    Rend la liste des n + 1 prix, du jour 0 au jour n."""
    dt = 1 / n
    S = [S0]
    for i in range(n):
        Z = np.random.normal()
        nouveau_prix = S[-1] * np.exp(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * Z)
        S.append(nouveau_prix)
    return S


# =====================================================================
# 3. STRATÉGIE DE COUVERTURE
# =====================================================================

def hedge_pnl(S, K, sigma_impl, n):
    """P&L d'une banque qui vend un call et se couvre en delta chaque jour.

    Jour 0      : encaisse la prime, achète delta actions.
    Jours 1..n-1: ajuste son nombre d'actions au nouveau delta.
    Échéance    : revend ses actions et paie le payoff au client.
    Le cash restant est le P&L.
    """
    dt = 1 / n
    T = 1.0

    # Jour 0 : vendre le call, acheter delta actions
    cash = bs_call(S[0], K, T, 0, sigma_impl)
    shares = bs_delta(S[0], K, T, 0, sigma_impl)
    cash = cash - shares * S[0]

    # Jours 1 à n-1 : rebalancement
    for i in range(1, n):
        T_restant = T - i * dt
        nouveau_delta = bs_delta(S[i], K, T_restant, 0, sigma_impl)
        cash = cash - (nouveau_delta - shares) * S[i]
        shares = nouveau_delta

    # Échéance : revendre les actions, payer le payoff
    cash = cash + shares * S[n]
    cash = cash - max(S[n] - K, 0)
    return cash


def simuler_pnls(sigma_real, nb, sigma_impl=0.20, n=252):
    """P&L du vendeur couvert sur nb trajectoires simulées à la vol sigma_real."""
    pnls = []
    for k in range(nb):
        chemin = simule_path(100, sigma_real, n)
        pnls.append(hedge_pnl(chemin, 100, sigma_impl, n))
    return pnls


# =====================================================================
# 4. TESTS DE VALIDATION
# =====================================================================

def tests():
    print("=== Tests de validation ===")
    print("Call ATM (attendu 7,97)   :", round(bs_call(100, 100, 1, 0, 0.2), 4))
    print("Delta ATM (attendu 0,54)  :", round(bs_delta(100, 100, 1, 0, 0.2), 4))

    # Delta par différence finie : pente du prix par rapport au spot
    h = 0.01
    delta_approx = (bs_call(100 + h, 100, 1, 0, 0.2) - bs_call(100 - h, 100, 1, 0, 0.2)) / (2 * h)
    print("Delta par différence finie:", round(delta_approx, 4))
    print()


# =====================================================================
# 5. ÉTUDE MONTE CARLO
# =====================================================================

if __name__ == "__main__":
    tests()
    np.random.seed(0)

    print("=== P&L du vendeur couvert (vol implicite 20 %, 1 000 trajectoires) ===")
    resultats = {}
    for sigma_real in [0.10, 0.20, 0.30]:
        pnls = simuler_pnls(sigma_real, 1000)
        resultats[sigma_real] = pnls
        theorie = bs_call(100, 100, 1, 0, 0.20) - bs_call(100, 100, 1, 0, sigma_real)
        print("Vol réalisée", sigma_real,
              "-> P&L moyen", round(np.mean(pnls), 2),
              "| écart-type", round(np.std(pnls), 2),
              "| théorie", round(theorie, 2))

    for sigma_real, pnls in resultats.items():
        plt.hist(pnls, bins=50, alpha=0.5, label=f"vol réalisée {int(sigma_real * 100)} %")
    plt.legend()
    plt.title("P&L du vendeur de call couvert (vol implicite 20 %)")
    plt.xlabel("P&L")
    plt.ylabel("Nombre de simulations")
    plt.savefig("histogrammes_delta_hedge.png", dpi=120)
    plt.show()
