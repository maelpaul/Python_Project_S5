import csv
import numpy as np
from scipy.stats import chi2_contingency as chi2_contingency
from scipy.stats import chi2
import matplotlib.pyplot as plt

fh = open('Chicago_Crimes_2012_to_2017.csv')
reader = csv.reader(fh, delimiter = ',', quotechar = '"')
    
L = []
T = []

for ligne in reader:
    if ligne[14] != '' and ligne[14] != '0.0':
        if not(ligne[6] in L):
            L.append(ligne[6])
        if not(ligne[14] in T):
            T.append(ligne[14])

del L[0]
del T[0]

for i in range(len(T)):
    j = 0 
    a = ''
    while T[i][j] != '.':
        a += T[i][j]
        j += 1
    T[i] = int(a)

for i in range(len(T)):
    for j in range(len(T)-1):
        if T[j] > T[j+1]:
            T[j], T[j+1] = T[j+1], T[j]
            
for i in range(len(T)):
    T[i] = str(T[i])

# print(L, '\n')
# print(T, '\n')

l = [0 for i in range(len(L))]

fh = open('Chicago_Crimes_2012_to_2017.csv')
reader = csv.reader(fh, delimiter = ',', quotechar = '"')

for ligne in reader:
    if ligne[14] != '' and ligne[14] != '0.0':
        for i in range(len(L)):
            if (ligne[6] == L[i]):
                l[i] += 1

# print(l, '\n')

Ll = [[L[i], l[i]] for i in range(len(L))]

for i in range(len(L)):
    for j in range(len(L)-1):
        if Ll[j][1] > Ll[j+1][1]:
            Ll[j], Ll[j+1] = Ll[j+1], Ll[j]

a = 0
b = 0
M = []
for i in range(len(L)):
    if Ll[i][0] == 'NON-CRIMINAL (SUBJECT SPECIFIED)' or Ll[i][0] == 'NON - CRIMINAL':
        a += Ll[i][1]
        M.append(i)
    elif Ll[i][0] == 'NON-CRIMINAL':
        Ll[i][1] += a
        b = i

for i in range(1,-1,-1):
    del(Ll[M[i]])
    
b -= 2
while Ll[b][1] > Ll[b+1][1]:
    Ll[b], Ll[b+1] = Ll[b+1], Ll[b]
    b += 1
    
# print(Ll, '\n')            
            
L = [Ll[i][0] for i in range(len(Ll))]

# print(L, '\n')

c = 0
for i in range(len(Ll)):
    c += Ll[i][1]

# print(c, '\n')    

d = 0
for i in range(len(Ll)):
    d += 1

# print(d, '\n')    
    
e = 0
for i in range(len(T)):
    e += 1

# print(e, '\n')

fh = open('Chicago_Crimes_2012_to_2017.csv')
reader = csv.reader(fh, delimiter = ',', quotechar = '"')

V = [['', 0] for i in range(c)]

i = 0
for ligne in reader:
    if ligne[14] != '' and ligne[14] != '0.0' and ligne[14] != 'Community Area':
        s = ligne[6]
        if ligne[6] == 'NON-CRIMINAL (SUBJECT SPECIFIED)' or ligne[6] == 'NON - CRIMINAL':
            s = 'NON-CRIMINAL'
        V[i][0] = s
        j = 0
        a = ''
        while ligne[14][j] != '.':
            a += ligne[14][j]
            j += 1
        V[i][1] = int(a)
        i += 1

# print('Attributs choisis: types de crime -> i, secteurs communautaires -> j\n')

X = np.zeros((d,e))

for i in range(c):
    for j in range(len(L)):
        if V[i][0] == L[j]:
            a = V[i][1]
            X[j,a-1] += 1

# print('Tableau des effectifs =\n', X, '\n')    

def wl(i):
    s1 = 0
    for j in range(e):
        s1 = s1 + X[i,j]
    wl = s1 / c
    return wl

def Nl(i, j):
    s = 0
    for k in range(e):
        s = s + X[i,k]
    Nl = X[i,j] / s
    return Nl
    
def pl(j):
    s = 0
    for i in range(d):
        s = s + wl(i) * Nl(i, j)
    return s

p_l = []
for j in range(e):
    p_l.append(pl(j))

# print('Profil ligne moyen =\n', p_l, '\n') 

N_l = np.zeros((d,e))
for i in range(d):
    for j in range(e):
        N_l[i,j] = Nl(i, j)
    
# print('Tableau des profils lignes =\n', N_l, '\n')  

def wc(j):
    s1 = 0
    for i in range(d):
        s1 = s1 + X[i,j]
    wc = s1 / c
    return wc

def Nc(i, j):
    s = 0
    for k in range(d):
        s = s + X[k,j]
    Nc = X[i,j] / s
    return Nc
    
def pc(i):
    s = 0
    for j in range(e):
        s = s + wc(j) * Nc(i, j)
    return s

p_c = []
for i in range(d):
    p_c.append(pc(i))

# print('Profil colonne moyen =\n', p_c, '\n') 

N_c = np.zeros((d,e))
for i in range(d):
    for j in range(e):
        N_c[i,j] = Nc(i, j)

# print('Tableau des profils colonnes =\n', N_c, '\n')

def xl(i):
    s = 0
    for j in range(e):
        s = s + X[i,j]
    return s

def xc(j):
    s = 0 
    for i in range(d):
        s = s + X[i,j]
    return s
    
Xhat = np.zeros((d,e))
for i in range(d):
    for j in range(e):
        Xhat[i,j] = (1/c) * xl(i) * xc(j)

# print('Effectifs attendus sous hypothèse d\'indépendance =\n', Xhat, '\n')

dchi2 = 0
for i in range(d):
    for j in range(e):
        a = X[i,j] - Xhat[i,j]
        dchi2 = dchi2 + a**2 / Xhat[i,j] 
    
# print('Distance du chi2 =', dchi2, '\n')

pvalue = 0
# print('pvalue =', pvalue, '\n')

chi2, p, dof, expected = chi2_contingency(X)
# print('chi2 =', chi2)
# print('p =', p)
# print('dof =', dof)
# print('expected =\n', expected)

D_l = np.zeros((d,d))
for i in range(d):
    D_l[i,i] = 1/np.sqrt(wl(i))

# print('Matrice de diagonale de poids lignes =\n', D_l, '\n')
 
D_c = np.zeros((e,e))
for j in range(e):
    D_c[j,j] = 1/np.sqrt(wc(j))

# print('Matrice de diagonale de poids colonnes =\n', D_c, '\n')

R = (np.matmul(np.matmul(D_l, X - Xhat), D_c)) / c

# print('Matrice des résidus normalisés =\n', R, '\n')

U,S,Vt = np.linalg.svd(R, full_matrices=False)
S = np.diag(S)

# print('U =\n', U, '\n')
# print('S =\n', S, '\n')
# print('Vt =\n', Vt, '\n')

# Inertie
I = 0
for i in range(d):
    I = I + S[i,i]**2
    
# print('Inertie totale =', I, '\n')

I_2 = 0
for i in range(2):
    I_2 = I_2 + S[i,i]**2
    
# print('Inertie partielle =', I_2, '\n')

# print('Proportion inertie:', (I_2 / I) * 100, '\n')

# Calcul coordonnees lignes et colonne
C_l = np.matmul(np.matmul(D_l, U), S)
# print('Matrice des cordonnées des projections des attributs lignes =\n', C_l, '\n')

C_c = np.matmul(np.matmul(D_c, np.transpose(Vt)), S)
# print('Matrice des cordonnées des projections des attributs colonnes =\n', C_c, '\n')

# Utiliser plt.scatter
plt.scatter([-row[0] for row in C_l], [row[1] for row in C_l])
plt.scatter([-row[0] for row in C_c], [row[1] for row in C_c])

# Ajouter des etiquettes avec plt.annotate
for i in range(d):
    label_1 = "(" + L[i] + ")"
    plt.annotate(label_1, (-C_l[i, 0], C_l[i, 1]))

for i in range(e):
    label_2 = "(" + T[i] + ")"
    plt.annotate(label_2, (-C_c[i, 0], C_c[i, 1]))

plt.title('Crimes à Chicago')
plt.show()
