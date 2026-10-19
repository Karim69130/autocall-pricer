"""
Pricer d'autocall par Monte Carlo
=================================

Produit : autocall sur indice (ex. Euro Stoxx 50)
- observations annuelles ;
- rappel anticipé si l'indice est >= barrière de rappel à une date d'observation :
  le client reçoit 100 + 100 x coupon x nombre d'années écoulées ;
- à l'échéance, si jamais rappelé :
    * indice >= barrière de protection : le client récupère 100 ;
    * indice <  barrière de protection : il récupère 100 x performance finale.

Méthode : simulation de l'indice en univers risque-neutre (dérive r - q),
puis moyenne des flux actualisés sur N trajectoires.

Contenu du fichier :
  1. Outils Black-Scholes (call, put)
  2. Fonctions du pricer (simulation, payoff, prix, coupon juste)
  3. Tests de validation
  4. Analyse : coupon juste aux conditions de marché et sensibilités
"""

import numpy as np
from scipy.stats import norm

NOMINAL = 100          # montant investi par le client
JOURS_PAR_AN = 252     # jours de bourse par an


# =====================================================================
# 1. OUTILS BLACK-SCHOLES (sans dividende)
#    Utilisés pour les vérifications et, plus tard, le reverse convertible.
# =====================================================================

def calcul_d1(S, K, T, r, sigma):
    """d1 de la formule de Black-Scholes."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call(S, K, T, r, sigma):
    """Prix Black-Scholes d'un call européen."""
    d1 = calcul_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    """Prix Black-Scholes d'un put européen."""
    d1 = calcul_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# =====================================================================
# 2. FONCTIONS DU PRICER
# =====================================================================

def simule_path_rn(S0, r, q, sigma, T):
    """Simule une trajectoire journalière de l'indice en univers risque-neutre.

    Entrées : niveau initial S0, taux r, dividende q, vol sigma, maturité T (années).
    Sortie  : liste des prix, du jour 0 au jour 252 x T (252 x T + 1 valeurs).
    """
    dt = 1 / JOURS_PAR_AN
    n = int(JOURS_PAR_AN * T)
    S = [S0]
    for i in range(n):
        Z = np.random.normal()
        nouveau_prix = S[-1] * np.exp((r - q - 0.5 * sigma**2) * dt
                                      + sigma * np.sqrt(dt) * Z)
        S.append(nouveau_prix)
    return S


def pay_off(S, S0, c, br, bp, T):
    """Applique les règles de l'autocall à une trajectoire.

    Entrées : trajectoire S, niveau initial S0, coupon annuel c,
              barrière de rappel br, barrière de protection bp (en décimal,
              ex. 1.0 et 0.6), maturité T (années entières).
    Sortie  : (montant reçu par le client, année du paiement).
    """
    # 1. Dates d'observation annuelles : rappel anticipé possible
    for annee in range(1, T + 1):
        jour = JOURS_PAR_AN * annee
        perf = S[jour] / S0
        if perf >= br:
            montant = NOMINAL + NOMINAL * c * annee
            return montant, annee

    # 2. Jamais rappelé : protection observée uniquement à l'échéance
    perf_finale = S[-1] / S0
    if perf_finale >= bp:
        return NOMINAL, T
    else:
        return NOMINAL * perf_finale, T


def prix_autocall(S0, r, q, sigma, c, br, bp, T, N):
    """Prix Monte Carlo de l'autocall.

    Simule N trajectoires, applique le payoff, actualise chaque flux
    à sa date de paiement et fait la moyenne.
    La graine est fixée : deux appels utilisent les mêmes trajectoires,
    ce qui permet de comparer proprement deux jeux de paramètres.
    Sortie : (prix, erreur Monte Carlo).
    """
    np.random.seed(42)
    flux = []
    for k in range(N):
        chemin = simule_path_rn(S0, r, q, sigma, T)
        montant, annee = pay_off(chemin, S0, c, br, bp, T)
        flux.append(montant * np.exp(-r * annee))

    prix = np.mean(flux)
    erreur = np.std(flux) / np.sqrt(N)
    return prix, erreur


def coupon_juste(S0, r, q, sigma, br, bp, T, N, cible):
    """Coupon annuel qui amène le prix de l'autocall au niveau 'cible'.

    Avec des trajectoires fixées, les dates de rappel ne dépendent pas
    du coupon : le prix est une droite en fonction du coupon.
    Deux prix suffisent (coupon 0 % et 10 %) pour trouver le coupon cherché.
    cible = 100 : coupon théorique ; cible = 98 : coupon avec 2 points de marge.
    """
    prix_0, _ = prix_autocall(S0, r, q, sigma, 0.0, br, bp, T, N)
    prix_10, _ = prix_autocall(S0, r, q, sigma, 0.10, br, bp, T, N)
    pente = (prix_10 - prix_0) / 0.10
    return (cible - prix_0) / pente


# =====================================================================
# 3. TESTS DE VALIDATION
# =====================================================================

def tests():
    print("=== Tests de validation ===")

    # Parité call-put : C - P = S - K e^(-rT)
    S, K, T, r, sigma = 110, 100, 0.5, 0.05, 0.2
    gauche = bs_call(S, K, T, r, sigma) - bs_put(S, K, T, r, sigma)
    droite = S - K * np.exp(-r * T)
    print("Parité call-put :", round(gauche, 6), "=", round(droite, 6))

    # Moyenne risque-neutre : E[S_T] = S0 e^((r - q) T)
    np.random.seed(0)
    finaux = [simule_path_rn(100, 0.03, 0.01, 0.2, 5)[-1] for k in range(2000)]
    erreur = np.std(finaux) / np.sqrt(len(finaux))
    print("Moyenne risque-neutre :", round(np.mean(finaux), 2),
          "| théorie :", round(100 * np.exp((0.03 - 0.01) * 5), 2),
          "| erreur MC :", round(erreur, 2))

    # Payoffs sur trajectoires construites à la main (coupon 8 %)
    longueur = JOURS_PAR_AN * 5 + 1
    print("Payoff, indice à 105 :", pay_off([105] * longueur, 100, 0.08, 1.0, 0.6, 5),
          "| attendu (108, 1)")
    print("Payoff, indice à 80  :", pay_off([80] * longueur, 100, 0.08, 1.0, 0.6, 5),
          "| attendu (100, 5)")
    print("Payoff, indice à 50  :", pay_off([50] * longueur, 100, 0.08, 1.0, 0.6, 5),
          "| attendu (50, 5)")
    chemin = [95] * 300 + [102] * (longueur - 300)
    print("Payoff, 95 puis 102  :", pay_off(chemin, 100, 0.08, 1.0, 0.6, 5),
          "| attendu (116, 2)")
    print()


# =====================================================================
# 4. ANALYSE
# =====================================================================

if __name__ == "__main__":
    tests()

    # Paramètres du produit
    br, bp, T = 1.0, 0.6, 5
    # Nombre de simulations (erreur d'environ +/- 0,5 point de coupon)
    N = 2000

    # Conditions de marché actuelles (à mettre à jour avant chaque utilisation)
    S0 = 100
    r_marche, q_marche, sigma_marche = 0.025, 0.03, 0.206

    print("=== Coupon juste aux conditions de marché ===")
    c_100 = coupon_juste(S0, r_marche, q_marche, sigma_marche, br, bp, T, N, 100)
    c_98 = coupon_juste(S0, r_marche, q_marche, sigma_marche, br, bp, T, N, 98)
    print("Coupon théorique (prix = 100)   :", round(c_100 * 100, 2), "%")
    print("Coupon avec 2 pts de marge (98) :", round(c_98 * 100, 2), "%")
    print()

    # Sensibilités (environ 5 minutes de calcul) : passer à True pour lancer
    LANCER_SENSIBILITES = False

    if LANCER_SENSIBILITES:
        # Valeurs de base pour l'étude
        r, q, sigma = 0.03, 0.03, 0.20

        print("=== Sensibilité du coupon juste (cible 100) ===")
        for s in [0.15, 0.20, 0.25, 0.30]:
            cj = coupon_juste(S0, r, q, s, br, bp, T, N, 100)
            print("Vol", s, "-> coupon juste", round(cj * 100, 2), "%")

        for rt in [0.01, 0.03, 0.05]:
            cj = coupon_juste(S0, rt, q, sigma, br, bp, T, N, 100)
            print("Taux", rt, "-> coupon juste", round(cj * 100, 2), "%")

        for div in [0.01, 0.03, 0.05]:
            cj = coupon_juste(S0, r, div, sigma, br, bp, T, N, 100)
            print("Dividendes", div, "-> coupon juste", round(cj * 100, 2), "%")
