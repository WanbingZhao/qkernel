import os
import math
from schwimmbad import MultiPool
from tqdm import tqdm, trange

import numpy as np
from numpy import pi as PI

import paddle
from paddle import trace, matmul 
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.state import density_op


class QKernel:
    def __init__(self, n_qubit=7, c1=1, noise_free=True, gamma=0.1, p=0.1):
        self.n_qubit = n_qubit
        self.c1 = c1
        self.noise_free = noise_free
        if not noise_free:
            self.gamma = gamma
            self.p = p
        return 

    def U_encoding(self, x1, x2):
        n = self.n_qubit
        d = self.d
        r = d // (3*n)
        cir = self.cir

        for i in range(r):
            # 加上一层 Hadamard 门
            cir.superposition_layer()
            # 加上一层旋转门 Rz
            for j in range(n):
                cir.rz(x1[3*j + 3*n*i], j)
            # 加上一层旋转门 Ry
            for j in range(n):
                cir.ry(x1[3*j + 1 + 3*n*i], j)
            # 加上一层旋转门 Rz
            for j in range(n):
                cir.rz(x1[3*j + 2 + 3*n*i], j)
            # 加上 sqrt_iSWAP 门
            for j in range(n // 2):
                cir.rxx(paddle.to_tensor(np.array([PI/-4])), [2*j, 2*j+1])
                cir.ryy(paddle.to_tensor(np.array([PI/-4])), [2*j, 2*j+1])
            for j in range(math.ceil(n/2) - 1):
                cir.rxx(paddle.to_tensor(np.array([PI/-4])), [2*j+1, 2*j+2])
                cir.ryy(paddle.to_tensor(np.array([PI/-4])), [2*j+1, 2*j+2])
        
        # 加上一层 Hadamard 门
        cir.superposition_layer()

        for j in range(min(n, d % (3*n))):
            cir.rz(x1[j + 3*n*r] ,j)
        if d % (3*n) > n:
            for j in range(min(n, d % (3*n) - n)):
                cir.ry(x1[j + 1 + 3*n*r] ,j)
            if d % (3*n) > 2*n:
                for j in range(min(n, d % (3*n) - n)):
                    cir.rz(x1[j+2+3*n*r] ,j)
                #inverse
                for j in range(min(n, d % (3*n) - n)):
                    cir.rz(-x2[j + 2 + 3*n*r] ,j)
            for j in range(min(n, d % (3*n) - n)):
                cir.ry(-x2[j + 1 + 3*n*r] ,j)
        for j in range(min(n, d % (3*n))):
            cir.rz(-x2[j + 3*n*r] ,j)

        # 加上一层 Hadamard 门
        cir.superposition_layer()

        for i in reversed(range(r)):        
            # 加上 sqrt_iSWAP 门
            for j in range(math.ceil(n/2) - 1):
                cir.rxx(paddle.to_tensor(np.array([PI/4])), [2*j+1, 2*j+2])
                cir.ryy(paddle.to_tensor(np.array([PI/4])), [2*j+1, 2*j+2])
            for j in range(n // 2):
                cir.rxx(paddle.to_tensor(np.array([PI/4])), [2*j, 2*j+1])
                cir.ryy(paddle.to_tensor(np.array([PI/4])), [2*j, 2*j+1])
            # 加上一层旋转门 Rz
            for j in range(n):
                cir.rz(-x2[3*j+2+3*n*i] ,j)
            # 加上一层旋转门 Ry
            for j in range(n):
                cir.ry(-x2[3*j+1+3*n*i] ,j)
            # 加上一层旋转门 Rz
            for j in range(n):
                cir.rz(-x2[3*j+3*n*i] ,j)
            # 加上一层 Hadamard 门
            cir.superposition_layer()   
        
        # Add noise
        if not self.noise_free:
            for j in range(n):
                cir.generalized_amplitude_damping(self.gamma, self.p, j)


    # The QKE circuit simulated by paddle quantm
    def q_kernel_estimator(self, x1, x2):
        assert(x1.shape[0] == x2.shape[0])
        self.d = x1.shape[0]    # 数据维度

        # Create the circuit
        self.cir = UAnsatz(self.n_qubit)

        # Transform data vectors into tensors
        # Multiply c1
        x1 = paddle.to_tensor(x1 * self.c1)
        x2 = paddle.to_tensor(x2 * self.c1)
        
        # Add the encoding circuit for data
        self.U_encoding(x1, x2)
        
        # Run the circuit with state vector mode
        # fin_state = self.cir.run_state_vector()
        fin_state = self.cir.run_density_matrix()
        
        # Return the probability of measuring 0...0 
        # result = self.cir.measure(shots = 2048)
        # return (fin_state[0].conj() * fin_state[0]).real().numpy()[0]  
        # return fin_state[0].real().numpy()[0]  
        ini_state = paddle.to_tensor(density_op(self.n_qubit))
        tr = trace(matmul(ini_state, fin_state))
        tr_squeezed = paddle.fluid.layers.squeeze(tr, axes = [0])
        return paddle.real(tr_squeezed)



    # Define a kernel matrix function, for which the input should be two list of vectors
    # This is needed to customize the SVM kernel
    def q_kernel_matrix(self, X1, X2):
        n1 = len(X1)
        n2 = len(X2)
        self.X1 = X1
        self.X2 = X2
        self.train = (n1 == n2) and (X1 == X2).all()
        n_total = n1 * n2

        xv, yv = np.meshgrid(range(n1), range(n2), indexing='ij')
        pos = list(zip(xv.reshape(-1).tolist(), yv.reshape(-1).tolist()))

        n_iter = 1000
        n_form = 0
        form_res = np.array([])
        tmp_file = 'kernel_tmp.npy'
        if os.path.exists(tmp_file):
            form_res = np.load(tmp_file, allow_pickle=True).astype(np.float64)
            n_form = (form_res.shape[0]) // n_iter
            form_res = form_res[:n_form * n_iter]
        for i in trange(int(np.ceil(n_total / n_iter)), desc=f'Computing {n1} by {n2} kernel'):
            if i >= n_form:
                with MultiPool() as pool:
                    if (i + 1) * n_iter <= n_total:
                        res = pool.imap(self.worker, pos[i * n_iter:(i + 1) * n_iter])
                    else:
                        res = pool.imap(self.worker, pos[i * n_iter:n_total])
                    res = np.array(list(res))
                    res = np.append(form_res, res)
                    np.save(tmp_file, res)

        mat = res.reshape(n1, n2)
        if self.train:
            mat = mat + mat.T + np.eye(n1, n2) * self.c1

        return mat

        # return np.array([[self.q_kernel_estimator(x1, x2) for x2 in X2] for x1 in X1])


    def worker(self, pos):
        i, j = pos
        if self.train and i >= j:
            return 0
        else:
            val = self.q_kernel_estimator(self.X1[i, :], self.X2[j, :])
        return float(val.numpy())

