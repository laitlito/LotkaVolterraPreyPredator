"""
Nom du fichier : ModelA2Equations.py
Auteur(s) : Astruc Lélio, Del Rosso Luca, Edery Nathan
Date de création : 08/04/2023
Dernière mise à jour : 09/05/2023
Version : 1.3

Description : Voici une implémentation en python du modèle proies-prédateurs de Lotka-Volterra.
"""

    ## Imports nécéssaires pour les opérations d'algèbre linéaire et d'affichage graphique
import numpy as np
import matplotlib.pyplot as plt

    ## Coefficients d'intéraction entre les populations.
alpha = 0.1
beta = 0.02
gamma = 0.3
delta = 0.01

    ## Conditions initiales 
# Population initiale de proies 
x0 = 50
# Population initale de prédateurs
y0 = 15
z0 = np.array([x0, y0])
#Intervalles de temps, et pas de discrétisation h
tempsInitial = 0
tempsFinal = 2500
h = 0.01
N = int((tempsFinal - tempsInitial)/h)
nbMaxIterations = 1000
erreur = 1e-6

    ## Fonctions euler explicite
def xPoint(x, y):
    return alpha *x - beta *x*y

def yPoint(x, y):
    return delta *x*y - gamma *y

def euler(x0, y0, h, N):
    x = np.zeros(N)
    y = np.zeros(N)
    x[0] = x0
    y[0] = y0
    for n in range(1, N):
        x[n] = x[n-1] + h*xPoint(x[n-1], y[n-1])
        y[n] = y[n-1] + h*yPoint(x[n-1], y[n-1])
    return x, y

    ## Fonctions Runge-Kutta à l'ordre 4
def lotka_volterra(x, y, alpha, beta, gamma, delta):
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return np.array([dxdt, dydt])

def runge_kutta_4(F, t0, z0, h, N):
    t = np.zeros(N+1)
    z = np.zeros((N+1, len(z0)))
    t[0] = t0
    z[0] = z0
    for n in range(N):
        k1 = F(t[n], z[n])
        k2 = F(t[n] + h/2, z[n] + h/2 * k1)
        k3 = F(t[n] + h/2, z[n] + h/2 * k2)
        k4 = F(t[n] + h, z[n] + h * k3)
        z[n+1] = z[n] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t[n+1] = t[n] + h 
    return t, z

F = lambda t, z: lotka_volterra(z[0], z[1], alpha, beta, gamma, delta)


x_euler, y_euler = euler(x0, y0, h, N)
t, z = runge_kutta_4(F, tempsInitial, z0, h, N)

x_rk = z[:, 0]
y_rk = z[:, 1]

    ## Troncature des vecteurs pour avoir la même taille
min_length = min(len(x_euler), len(x_rk))
x_euler = x_euler[:min_length]
x_rk = x_rk[:min_length]
y_euler = y_euler[:min_length]
y_rk = y_rk[:min_length] 

    ## Evolution de la population de proies et de prédateurs dans le temps
plt.figure()
plt.plot(np.linspace(tempsInitial, tempsFinal, N), x_euler, label='Proies (Euler)')
plt.plot(np.linspace(tempsInitial, tempsFinal, N), y_euler, label='Prédateurs (Euler)')
plt.plot(np.linspace(tempsInitial, tempsFinal, N), x_rk, label='Proies (Runge-Kutta-4)', color='green')
plt.plot(np.linspace(tempsInitial, tempsFinal, N), y_rk, label='Prédateurs (Runge-Kutta-4)', color='red')
plt.xlabel('Temps')
plt.ylabel('Proies')
plt.legend()
plt.title('Évolution de la population de proies et des prédateurs dans le temps')

    ## Proies en fonction des prédateurs
plt.figure()
plt.plot(x_euler, y_euler, label='Euler')
plt.plot(x_rk, y_rk, label='Runge-Kutta-4', color='orange')
plt.xlabel('Proies')
plt.ylabel('Prédateurs')
plt.legend()
plt.title('Évolution des population de proies en fonction des prédateurs')
plt.show()

"""
### Auteurs : Blond Alexis, Gérard Théva, Manuel Enzo
## Euler implicite avec Newton
 def f(y, alpha, beta, delta, gamma):
    x, y = y
    return [alpha * x - beta * x * y, delta * x * y - gamma * y]

def Jacobien(y, alpha, beta, delta, gamma):
    x, y = y
    return [[alpha - beta * y, -beta * x], [delta * y, -gamma + delta * x]]


def eulerNewton(f, j, y0, t0, tf, h, alpha, beta, delta, gamma, nbMaxIterations, erreur):
    n = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n)
    Y = np.zeros((n, 2))
    Y[0] = y0
    for i in range(1, n):
        yn = Y[i-1].copy()
        for k in range(nbMaxIterations):
            J = np.eye(2) - h * np.array(j(yn, alpha, beta, delta, gamma))
            F = np.array(yn) - np.array(Y[i-1]) - h * np.array(f(yn, alpha, beta, delta, gamma))
            dy = np.linalg.solve(J, -F)
            yn += dy
            if np.linalg.norm(dy) < erreur:
                break
        Y[i] = yn
    return t, Y 
#t, y = eulerNewton(f, Jacobien, y0, tempsInitial, tempsFinal, h, alpha, beta, delta, gamma, nbMaxIterations, erreur)
"""
