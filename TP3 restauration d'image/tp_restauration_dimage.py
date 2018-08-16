
# coding: utf-8

# In[1]:

import pickle
import matplotlib.pylab as plt
from matplotlib import cm
import scipy.sparse.linalg

# get_ipython().magic('matplotlib inline')


# In[2]:

data = pickle.load(open('data2018.pk','rb'))


# In[3]:

original = data.get('original')
observations = data.get('observations')

f, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(original, cmap=cm.gray)
axarr[0].set_title('original')

axarr[1].imshow(observations, cmap=cm.gray)
axarr[1].set_title('observations');


# In[4]:

def l1_prox(x, alpha):
    n = x.shape[0]
    result = np.zeros(n)
    for i in range(n):
        if x[i] < -alpha:
            result[i] = x[i] + alpha
        elif x[i] > alpha:
            result[i] = x[i] - alpha

    return result

def indicator_prox(x, amin, amax, gamma):
    return np.clip(x, a_min=amin, a_max=amax)

def indicator(x, amin, amax):
    return np.array([ 0 if amin <= x_i <= amax else np.inf  
                     for x_i in x])

def EQM_callback(x, x_real):
    return np.mean((x-x_real)**2)

def B_callback(x, x_real):
    return x-x_real


# In[5]:

import numpy as np
from collections import defaultdict
from functools import partial
import scipy


# In[8]:

def admm(y, H, T, rho, beta, max_iter, trace=False, display=True, callbacks=None):
    """
    admm pour restauration d'image
    """
    history = defaultdict(list) if trace else None

    p = len(y)
    m = T.shape[0]

    # initialize
    I = scipy.sparse.eye(p)
    A = scipy.sparse.vstack([I, T])
    print(A.shape)

    gamma_1 = np.random.rand(p, 1)
    gamma_2 = np.random.rand(m, 1)
    gamma = np.vstack([gamma_1, gamma_2])

    z_1 = np.random.rand(p, 1)
    z_2 = np.random.rand(m, 1)
    z = np.vstack([z_1, z_2])
    print(z.shape)

    def augmented_lagrangian(x, z, gamma, A, H, T, rho, beta):
        f = 0.5 * np.linalg.norm(y - H.dot(x))**2
        g = beta * np.linalg.norm(T.dot(x), ord=1) + indicator(x, 0, 1)
        reg = gamma.T.dot(A.dot(x) - z) 
        reg += 0.5 * rho * np.linalg.norm(A.dot(x) - z)**2
        
        return f + g + reg

    Lp = partial(augmented_lagrangian, 
                 A=A, H=H, T=T, rho=rho, beta=beta)

    for k in range(max_iter):
        print(k)
        # x-update
        B = H.T.dot(H) + rho * A.T.dot(A)
        b = H.T.dot(y) - A.T.dot(gamma) + rho * A.T.dot(z)
        x, _ = scipy.sparse.linalg.cg(B, b)
        x = x.reshape([-1, 1])

        # z-updated
        z_1 = indicator_prox(x + gamma_1 / rho, 0, 1, 1/rho)
        z_2 = l1_prox(T.dot(x) + gamma_2 / rho, beta/rho)
        z_2 = z_2.reshape([-1, 1])
        z = np.vstack([z_1, z_2])

        # gamma - update
        gamma_1 = gamma_1 + rho * (x - z_1)
        gamma_2 = gamma_2 + rho * (T.dot(x) - z_2)
        gamma = np.vstack([gamma_1, gamma_2])

        if trace:
            # history['EQM'].append(None)
            # history['Bias']
            history['Lp'].append(Lp(x, z, gamma))
            if callbacks:
                for key, callback in callbacks.items():
                    history[key].append(callback(x))
                    
        if display:
            print('iteration: %d/%d' % (k, max_iter))

    return x, history


# In[6]:

original = data.get('original')
observations = data.get('observations')

K, L = original.shape


x = np.reshape(original, [K*L, 1])
y = np.reshape(observations, [K*L, 1])

callbacks = {
    'EQM' : lambda xk : EQM_callback(xk, x),
    'Bias': lambda xk : B_callback(xk, x)
}

# filter
H = data.get('H')
# Total variation
T = data.get('T')

rho = 0.01
beta = 0.01
k_max = 100


# In[ ]:

x_restored, history = admm(y, H, T, rho, beta, 
                          max_iter=k_max, trace=True, 
                          display=True, callbacks=callbacks)

