import numpy as np
from matplotlib import pyplot as plt

class twoDLDA:

    def __init__(self,ncomponents):
        self.ncomponents = ncomponents
        self.mean_img_class = []
        
    def fit(self,X,y):
    
        class_labels, self.N = np.unique(y, return_counts=True)
        self.mean_img = np.mean(X,0)
        SB = np.zeros((X.shape[2],X.shape[2])) 
        for l in class_labels:   
            self.mean_img_class.append(np.mean(X[y == l,:,:],0))
            Y = self.mean_img_class[-1] - self.mean_img
            SB += self.N[l] * (Y.T @ Y)
        #SB /= len(class_labels)
        
        SW = np.zeros((X.shape[2],X.shape[2])) 
        for l in class_labels:  
            M = np.mean(X[y == l,:,:],0)
            for A in X[y == l,:,:]:
                SW += ((A - M).T @ (A - M))
        #SW /= len(class_labels)
        

        A = np.linalg.inv(SW) @ SB
        #sign, logdet = np.linalg.slogdet(SW)
        #assert(np.allclose(np.linalg.inv(SW) @ SW, np.eye(SW.shape[0])))
        self.l, self.components = np.linalg.eig(A)
        #assert(np.allclose(A-A.T,np.zeros_like(A)))
        #for i in range(A.shape[1]):
        #    assert(np.allclose(A@self.components[:,i], self.l[i]*self.components[:,i], 1e-5))

        self.componenents = self.components[:, np.argsort(self.l)[::-1]] 

    def transform(self,X, k=None):
        k = self.ncomponents if k is None else k
        return X @ self.components[:,:k]


#def proyect(lda, X, k):   
#    return lda.transform(X, k)
#
#def invert(lda,lda_embeddings,k):
#    V = lda.components[:,:k]
#    return [x @ V.T  for x in lda_embeddings]


class twoDLDASquare:

    def __init__(self,ncomponents):
        self.ncomponents = ncomponents
        self.mean_img_class = []
        
    def fit(self,X,y):
    
        class_labels, self.N = np.unique(y, return_counts=True)
        self.mean_img = np.mean(X,0)
        
        SB = np.zeros((X.shape[2],X.shape[2])) 
        SB_alternate = np.zeros((X.shape[1],X.shape[1]))
        for l in class_labels:   
            self.mean_img_class.append(np.mean(X[y == l,:,:],0))
            Y = self.mean_img_class[-1] - self.mean_img
            SB += self.N[l] * (Y.T @ Y)
            SB_alternate += self.N[l] * (Y@ Y.T)
            
        SB /= len(class_labels)
        SB_alternate /= len(class_labels)

        SW = np.zeros((X.shape[2],X.shape[2])) 
        SW_alternate = np.zeros((X.shape[1],X.shape[1])) 
        for l in class_labels:  
            M = np.mean(X[y == l,:,:],0)
            for A in X[y == l,:,:]:
                SW += ((A - M).T @ (A - M))
            for A in X[y == l,:,:]:
                SW_alternate += ((A - M) @ (A - M).T)
    
        SW /= len(class_labels)
        SW_alternate /= len(class_labels)
        
        self.l, self.components = np.linalg.eig(np.linalg.inv(SW) @ SB)
        self.l2, self.components2 = np.linalg.eig(np.linalg.inv(SW_alternate) @ SB_alternate)
       
        self.componenents = self.components[:, np.argsort(self.l)[::-1]]
        self.componenents2 = self.components2[:, np.argsort(self.l2)[::-1]] 
        
       
    def transform(self,X, k=None):
        k = self.ncomponents if k is None else k
        return self.components2[:,:k].T @ X @ self.components[:,:k]



class twoDPCA:

    def __init__(self,ncomponents):
        self.ncomponents = ncomponents

    def fit(self,X,Y=None):
        self.mean = np.mean(X,0)
        X_centered = X - np.mean(X,0)
        G = np.zeros((X.shape[2],X.shape[2]))
        for i in range(X.shape[0]): G += X_centered[i].T@X_centered[i]
        G = G/(X_centered.shape[0])
        self.l, self.components = np.linalg.eig(G)
        self.componenents = self.components[:, np.argsort(self.l)[::-1]]
        plt.plot(np.cumsum(self.l)/np.sum(self.l))


    def transform(self,X, k=None):
        k = self.ncomponents if k is None else k
        return X @ self.components[:,:k]