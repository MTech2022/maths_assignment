import numpy as np


def QR_Decomposition(A):
    n, m = A.shape # get the shape of A
    print("--------------------------------------------------------")
    print(f"Given matrix shape is : {n}x{m}")
    
    #init Q and u
    Q = np.empty((n, n))
    u = np.empty((n, n))
    
    #To begin, we set u=a and then normalize:
    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])
    print("--------------------------------------------------------")
    #repeat the process as per 
    for i in range(1, n):
        print(f"STEP : {i}")
        print("--------------------------------------------------------")
        print(f"\n==> u{i}")
        u[:, i] = A[:, i]
        print(u)
        
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] # get each u vector
        
        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vetor
        print(f"\n==> e{i}")
        print(Q)
        print(f"\nEND OF STEP {i}");
        print("--------------------------------------------------------\n")
    
    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]
    return Q, R
	
	
def diag_sign(A):
    "Compute the signs of the diagonal of matrix A"

    D = np.diag(np.sign(np.diag(A)))

    return D

def adjust_sign(Q, R):
    """
    Adjust the signs of the columns in Q and rows in R to
    impose positive diagonal of Q
    """

    D = diag_sign(Q)

    Q[:, :] = Q @ D
    R[:, :] = D @ R

    return Q, R
    

#A = np.array([[1, -1], [4, 2]])
A = np.array([[0, -1, 1], [4, 2, 0], [3, 4, 0]])
#A = np.array([[2, 1, 3, 3], [2, 1, -1,1], [2, -1, 3, -3], [2, -1, -1, -1]])
#A = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
# A = np.array([[1.0, 0.5, 0.2], [0.5, 0.5, 1.0], [0.0, 1.0, 1.0]])
# A = np.array([[1.0, 0.5, 0.2], [0.5, 0.5, 1.0]])

print("Input array")
print("[A] = \n", A)

Q, R = adjust_sign(*QR_Decomposition(A))

print('\n==> Q: \n', Q)
print('\n')
print('==> R: \n', R)
print('\n')
