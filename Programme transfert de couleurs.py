import matplotlib.image as im
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp




def vecteur(chaine): 
    I=im.imread(chaine)
    plt.imshow(I)
    plt.show() 
    plt.close()
    representer(np.array(I))
    n,m,r=np.shape(I) 
    d=[]
    for c in range(3): 
        L=[]
        for j  in range(m):
            for i in range(n):
                L.append(int(I[j,i,c]))
        d.append(L)
    vect=np.array(d) 
    return(vect,n)



def imagemodifie(L,n):
    V=[]
    l=[]
    P=[]
    for i in range (n):
        for j in range (n):
            for k in range(3):
                P.append(L[k][n*i+j])
            l.append(P)
            P=[]
        V.append(l) 
        l=[]
    M=np.array(V)
    plt.imshow(M)
    plt.show()
    plt.close()
    representer(M)
 


def representer(M): 
    r = M[:, :, 0].flatten()
    g = M[:, :, 1].flatten()
    b = M[:, :, 2].flatten()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    colors = M.reshape(-1, 3) / 255
    ax.scatter(r, g, b, c=colors, marker='o')
    ax.set_xlabel("Rouge")
    ax.set_ylabel("Vert")
    ax.set_zlabel("Bleu")
    ax.set_title("Représentation 3D des couleurs de l'image")
    plt.show()
    plt.close()

def distmat(X,Y):  
    return np.sum(X**2,0)[:,None] + np.sum(Y**2,0)[None,:] - 2*X.T@Y

def transport(support,palette):
    X,q = vecteur(support) 
    Y,s = vecteur(palette) 
    n=len(X[0])
    m=len(Y[0])
    a = np.ones((n,1))
    b = np.ones((m,1))
    P = cp.Variable((n,m))  
    C = distmat(X,Y)
    u = np.ones((m,1))
    v = np.ones((n,1))
    contrainte = [0 <= P, cp.matmul(P,u)==a, cp.matmul(P.T,v)==b] 
    objective = cp.Minimize( cp.sum(cp.multiply(P,C)))
    prob = cp.Problem(objective, contrainte)   
    result = prob.solve() #GLPK
    plt.figure(figsize = (5,5))
    plt.imshow(P.value);
    plt.show()
    plt.close()
    L=[]
    for i in range(3): 
        colonnecouleur=P.value@Y[i] 
        C=colonnecouleur.tolist() 
        Cbis=[int(round((elt),0)) for elt in C] 
        L.append(Cbis) 
    imagemodifie(L,q)
    
