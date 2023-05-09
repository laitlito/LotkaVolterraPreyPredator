"""
Nom du fichier : ModelA3Equations.py
Auteur(s) : Astruc Lélio, Del Rosso Luca, Edery Nathan
Date de création : 08/04/2023
Dernière mise à jour : 09/05/2023
Version : 1.1

Description : Voici une implémentation en python du modèle proies-prédateurs à 3 populations.
"""

    ## Imports nécéssaires pour les opérations d'algèbre linéaire et d'affichage graphique
import numpy as np
import matplotlib.pyplot as plt

    ## Coefficients d'intéraction entre les populations.
alpha = 3
beta = 0.025
gamma = 0.1
delta = 2
epsilon = 0.05
zeta = 0.025
eta = 0.05
theta = 1
iota = 0.05
kappa = 0.05

    ## Conditions initiales 
# Population initiale de proies 
x0 = 50
# Population initale de prédateurs
y0 = 15
z0 = 10
w0 = np.array([x0, y0, z0])
#Intervalles de temps, et pas de discrétisation h
tempsInitial = 0
tempsFinal = 10
h = 0.01
N = int((tempsFinal - tempsInitial)/h)
nbMaxIterations = 1000
erreur = 1e-6

    ## Fonctions euler explicite
def xPoint(x, y):
    return alpha *x - beta *x*x - gamma*x*y

def yPoint(x, y, z):
    return delta*y - epsilon*y*y - zeta*x*y - eta*y*z

def zPoint(y, z):
    return theta*y*z - iota*z*z - kappa*z

def euler(x0, y0, z0, h, N):
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    x[0] = x0
    y[0] = y0
    z[0] = z0
    for n in range(1, N):
        x[n] = x[n-1] + h*xPoint(x[n-1], y[n-1])
        y[n] = y[n-1] + h*yPoint(x[n-1], y[n-1], z[n-1])
        z[n] = z[n-1] + h*zPoint(y[n-1], z[n-1])
    return x, y, z

    ## Fonctions Runge-Kutta à l'ordre 4
def lotka_volterra(x, y, z, alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa):
    dxdt = x * (alpha - beta * x - gamma * y)
    dydt = y * (delta - epsilon * y - zeta * x - eta * z)

    dzdt = z * (theta * y - iota * z - kappa * x)
    return np.array([dxdt, dydt, dzdt])

def runge_kutta_4(F, t0, z0, h, N, params):
    t = np.zeros(N+1)
    z = np.zeros((N+1, len(z0)))
    t[0] = t0
    z[0] = z0
    
    for n in range(N):
        k1 = F(t[n], z[n], params)
        k2 = F(t[n] + h/2, z[n] + h/2 * k1, params)
        k3 = F(t[n] + h/2, z[n] + h/2 * k2, params)
        k4 = F(t[n] + h, z[n] + h * k3, params)
        z[n+1] = z[n] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t[n+1] = t[n] + h
        
    return t, z

params = (alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa)
F = lambda t, w, params: lotka_volterra(w[0], w[1], w[2], *params)

t, w = runge_kutta_4(F, tempsInitial, w0, h, N, params)
x_euler, y_euler, z_euler = euler(x0, y0, z0, h, N)

x_rk = w[:, 0]
y_rk = w[:, 1]
z_rk = w[:, 2]

    ## Troncature des vecteurs pour avoir la même taille
min_length = min(len(x_euler), len(x_rk))
x_euler = x_euler[:min_length]
x_rk = x_rk[:min_length]
y_euler = y_euler[:min_length]  
y_rk = y_rk[:min_length]
z_euler = z_euler[:min_length]
z_rk = z_rk[:min_length]

    ## Evolution de la population de proies et de prédateurs dans le temps
plt.plot(np.linspace(tempsInitial, tempsFinal, N), x_euler, label='Proies (Euler)')
plt.plot(np.linspace(tempsInitial, tempsFinal, N), y_euler, label='Proies (Euler)')
plt.plot(np.linspace(tempsInitial, tempsFinal, N), z_euler, label='Prédateurs (Euler)')
plt.plot(np.linspace(tempsInitial, tempsFinal, N), x_rk, label='Proies (Runge-Kutta-4)')
plt.plot(np.linspace(tempsInitial, tempsFinal, N), y_rk, label='Proies (Runge-Kutta-4)')
plt.plot(np.linspace(tempsInitial, tempsFinal, N), z_rk, label='Prédateurs (Runge-Kutta-4)')
plt.xlabel('Temps')
plt.ylabel('Population')
plt.legend()
plt.title('Évolution de la population de proies et des prédateurs dans le temps')
plt.show()

    ## Proies en fonction des prédateurs
plt.figure()
plt.plot(x_euler, y_euler, z_euler, label='Euler')
plt.plot(x_rk, y_rk, z_rk, label='Runge-Kutta-4')
plt.xlabel('Proies')
plt.ylabel('Prédateurs')
plt.legend()
plt.title('Évolution des population de proies en fonction des prédateurs')
plt.show() 

# Création du graphique en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracé du graphique
ax.plot(x_euler, y_euler, z_euler)
ax.plot(x_rk, y_rk, z_rk)
ax.set_xlabel('Proies (x)')
ax.set_ylabel('Proies (y)')
ax.set_zlabel('Prédateurs (z)')
ax.set_title('Évolution des proies en fonction des prédateurs')
plt.show()


"""
### Auteurs : Blond Alexis, Gérard Théva, Manuel Enzo
## Méthode d'Euler implicite avec Newton
def f(y, alpha, beta, delta, gamma, epsilon, zeta, eta, theta, iota, kappa):
    x, y, z = y
    return np.array([x * (alpha - beta * x - gamma * y),
           y * (delta - epsilon * y - zeta * x - eta * z),
           z * (theta * y - iota * z - kappa)])

def Jacobien(y, alpha, beta, delta, gamma, epsilon, zeta, eta, theta, iota, kappa):
    x, y, z = y
    return np.array([[alpha - beta * 2 * x - gamma * y,     -gamma * x,    0],
            [-zeta * y,   delta - epsilon * 2 * y - zeta * x - eta * z,  -eta * y],
            [0,    z * theta,    theta * y - iota * 2 * z - kappa]])


def eulerNewton(f, j, y0, t0, tf, h, alpha, beta, delta, gamma, epsilon, zeta, eta, theta, iota, kappa, erreur=1e-6, nbMaxIterations=100):
    n = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n)
    Y = np.zeros((n, 3))
    Y[0] = y0
    for i in range(1, n):
        yn = Y[i-1].copy()
        for k in range(nbMaxIterations):
            J = np.eye(3) - h * np.array(j(yn, alpha, beta, delta, gamma, epsilon, zeta, eta, theta, iota, kappa))
            F = np.array(yn) - np.array(Y[i-1]) - h * np.array(f(yn, alpha, beta, delta, gamma, epsilon, zeta, eta, theta, iota, kappa))
            dy = np.linalg.solve(J, -F)
            yn += dy
            if np.linalg.norm(dy) < erreur:
                break
        Y[i] = yn
    return t, Y  """