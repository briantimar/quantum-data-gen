
def get_H_XY(N):
    """ Returns local terms, without coupling coefficients, that make up the XY hamiltonian.
        returns: Hx, Hz, HPM
        Hx = list of pauli X's
        Hz = list of Pauli Z's
        HPM = list of sigma_plus * sigma_minus """

    I=qt.tensor([qt.qeye(2)] * N)
    c=list(it.combinations(range(N), 2))
    Hz,Hx, HPM=[0 * I for x in range(N)],[0 * I for x in range(N)], [0 * I for x in c]
    for i in range(N):
        l=[qt.qeye(2)] * N
        l[i]=qt.sigmaz()
        Hz[i]=qt.tensor(l).data.todense()
        l[i]=qt.sigmax()
        Hx[i]=qt.tensor(l).data.todense()
    for s in range(len(c)):
        i, j=c[s]
        l=[qt.qeye(2)] * N
        l[i]=qt.sigmap()
        l[j]=qt.sigmam()
        HPM[s]=qt.tensor(l).data.todense()
        HPM[s]+=np.conjugate(np.transpose(HPM[s]))
    return Hx,Hz, HPM


def construct_Hamiltonian(N):
    """Construct a matrix representation of the XY hamiltonian.
    N: the system size.
    The interaction strengths are loaded from Jij_10.npy; hence system size can be at most 10"""

    if N > 10:
        raise ValueError("coupling strengths for N > 10 not provided")
        
    __, Hz, HPM = get_H_XY(N)   # returns lists containing the individual terms of the Hamiltonian
                            # Hz: local sigma_z terms
                            # HPM: flip flop (between every to sites)

    disorder=[0.]*N  # optional disorder potential (here put to zero)

    ### Construct the Jij matrix
    Jij = np.load("Jij_10.npy")

    ### Construct the Hamiltonian
    H = sum([disorder[i]  * h for i,h in enumerate(Hz)], 0) # Local terms
    c=list(it.combinations(range(N), 2)) # all combinations (but lower triangular)
    for s, cs in enumerate(c):
        i, j = cs
        H+=HPM[s]*(Jij[i,j]+Jij[j,i])

    return H
