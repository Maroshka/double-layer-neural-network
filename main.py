
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def forward(x, w, v):
	x2 = np.zeros((x.shape[0], x.shape[1]+1))
	x2[:, 1:] = x

	x2[:, 0] = 1
	z = x2.dot(w)
	a = 1/ (1+np.exp(-z))

	a2 = np.zeros((a.shape[0], a.shape[1]+1))
	a2[:, 1:] = a
	a2[:, 0] = 1

	z3 = a2.dot(v)
	expz = np.exp(z3)

	h = expz / expz.sum(axis=1, keepdims=True)
	return h, a

def cost(y, p):
	cost = -y*np.log(p)-(1-y)*np.log(1-p)
	return cost.sum()


def classification_rate(y, p):
	return np.mean(y==p)*100
def derv_v(y, p, z):
	N, M = z.shape
	K = y.shape[1]
	z2 = np.zeros((N,M+1))
	z2[:, 1:] = z
	z2[:, 0] = 1
	# ret = np.zeros((M+1, K))
	# for n in range(N):
	# 	for k in range(K):
	# 		ret[:, k] += (p[n, k] - y[n, k])*z2[n,:]
	ret = z2.T.dot(p-y)

	return ret

def derv_w(y, p, z, v, x):
	N, D = x.shape
	M = z.shape[1]
	K = y.shape[1]
	x2 = np.zeros((N, D+1))
	x2[:,1:] = x
	x2[:, 0] = 1

	# ret = np.zeros((D+1, M))
	# for n in range(N):
	# 	for k in range(K):
	# 		for m in range(1, M+1):
	# 			ret[:, m] += (p[n, k]-y[n, k])*v[m, k]*z[n, m]*(1-z[n, m])*x2[n, :]
	dz = (p-y).dot(v.T)[:, 1:]*z*(1-z)
	ret = x2.T.dot(dz)
	return ret

def main():
	D = 2
	M = 3
	K = 3
	N = 500

	X1 = np.random.randn(N, D) + np.array([-2, 2])
	X2 = np.random.randn(N, D) + np.array([2, 2])
	X3 = np.random.randn(N, D) + np.array([0, -2])
	X = np.vstack([X1, X2, X3])

	Y = np.array([0]*N + [1]*N + [2]*N)
	N = len(Y)
	T = np.zeros((N, K))
	for n in range(N):
		T[n, Y[n]] = 1

	w = np.random.rand(3, 3)
	v = np.random.rand(4, 3)

	p, z = forward(X, w, v)
	alpha = .000005
	cs = []
	rs = []
	for i in range(10000):
		p, z = forward(X, w, v)
		r = classification_rate(Y, p.argmax(axis=1))
		c = cost(T, p)
		print "cost: ",c,", classification rate: ",r
		cs.append(c)
		rs.append(r)
		v -= alpha*derv_v(T, p, z)
		w -= alpha*derv_w(T, p, z, v, X)

	plt.plot(cs)
	plt.show()
	plt.plot(rs)
	plt.show()

if __name__ == '__main__':
	main()