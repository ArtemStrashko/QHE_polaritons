import numpy as np


# Fermi distribution
def Fermi(energy, temperature):
    xx = energy / temperature
    nf = 0.5 * (1 - np.tanh(xx/2))
    return nf

# log function for entropy term
def log_funct(energy, temperature):
    lf = (temperature * np.log(1.0 + np.exp(- abs(energy)/temperature)) + 
              0.5 * (abs(energy) - energy))
    return lf


def eps_matr_build(eta, phys_params):
    N_ll = phys_params['number of LLs']
    eps = np.zeros([N_ll])
    n = 0
    for i in range(len(eps)):
        eps[i] = eta[n]
        n += 1  
            
    eps_matr = np.diag(eps)
    
    return eps_matr


def M_matrix(variables, phys_params):
    
    '''
    This function converts a 1D array of variables, which me need for optimization, 
    to a 2D M-matrix, which we need for calculations of expectation values
    
    input = (X_1, X_2)
    X_1 - array of variables
    X_2 - dictionary of physical parameters
    
    output = M-matrix specifying mean-field matter ansatz
    '''

    # number of variables
    N_v = len(variables)
    
    # number of LLs
    N_ll = phys_params['number of LLs']

    # vectors of variables
    eta_e = np.copy(variables[0 : N_ll])
    eta_h = np.copy(variables[N_ll : 2*N_ll]) 
    lambdas = np.copy(variables[2*N_ll : 2*N_ll + N_ll*(N_ll - 1)])
    deltas = np.copy(variables[2*N_ll + N_ll*(N_ll - 1) : -1])

    # Building E and H matrices
    E_matr = np.zeros([N_ll, N_ll])
    H_matr = np.zeros([N_ll, N_ll])
    h_shift = int(len(lambdas)/2)
    n = 0
    # filling lambdas
    for i in range(0, N_ll):
        for j in range(i+1, N_ll):
            E_matr[i,j] = lambdas[n]
            H_matr[i,j] = lambdas[n + h_shift]
            n += 1               
    
    # Building epsilon matrices 
    eps_e = eps_matr_build(eta_e, phys_params)
    eps_h = eps_matr_build(eta_h, phys_params)
    
    # Building M_00 and M_11 blocks
    M_00 = E_matr + np.transpose(E_matr) + eps_e
    M_11 = H_matr + np.transpose(H_matr) - eps_h
    
    # Building Delta matrix
    Delta = np.zeros([N_ll, N_ll])
    n = 0
    for i in range(N_ll):
        for j in range(N_ll):
            Delta[i,j] = deltas[n]
            n += 1
    
    # constructing full M matrix
    M = np.block([[M_00, Delta], [np.transpose(Delta), M_11]])
    
    
    return M



# calculating basic expectations
def expect(W_matrix, eigen, phys_params, is_hole = False):
    
    '''
    This function calculates expectation values
    
    input = (W_matrix, eigen, phys_params, is_hole)
    W_matrix - matrix for calculations of a particular expectation value
    eigen - eigenenergies and corresponding normalised eigenvalues
    phys_params - dictionary of physical parameters
    is_hole = True if calculate expectations value of a number of holes
    is_hole = False otherwise
    
    output = corresponding expectation value
    '''
    
    eps, U = eigen
    D = np.transpose(U) @ W_matrix @ U
    n_ii = Fermi(eps, phys_params['temperature'])
    expect = np.einsum('i,ii->', n_ii, D)
    
    if is_hole:
        expect = 1 - expect

    return expect


def set_one(i,j, phys_params):
    N_l = phys_params['number of LLs']
    W = np.zeros([2*N_l, 2*N_l])
    W[i,j] = 1
    return W


def expectations_and_eps(var, phys_params):
    
    from numpy import linalg as LA
    
    '''
    This function calculates all expectation values and eigenenergies
    
    input = (var, phys_params)
    var - 1D array of all variables
    phys_params - dictionary of physical parameters
    
    output = a dictionary of expectation values and eigenenergies
    '''

    # calculating expectations
    M = M_matrix(var, phys_params)

    # eigenvalues and eigenvectors of M
    eps_and_U = LA.eig(M)

    N_ll = phys_params['number of LLs']

    # set up an empty dictionary of expectation values
    expect_dict = {}

    # N^{e_i/h_i} terms
    for i in range(N_ll):

        name_e = 'N_e_' + str(i)
        name_h = 'N_h_' + str(i)

        W_e = set_one(i, i, phys_params)
        W_h = set_one(i + N_ll, i + N_ll, phys_params)

        expect_dict[name_e] = expect(W_e, eps_and_U, phys_params)
        expect_dict[name_h] = expect(W_h, eps_and_U, phys_params, True)
         
        
    # C-temrs (intraband coherences)
    for i in range(0, N_ll):
        for j in range(i+1, N_ll):

            name_Ce = 'C_e_' + str(i) + str(j)
            name_Ch = 'C_h_' + str(i) + str(j)

            W_e = set_one(i, j , phys_params)
            expect_dict[name_Ce] = expect(W_e, eps_and_U, phys_params)
            expect_dict['C_e_' + str(j) + str(i)] = expect_dict[name_Ce]
            
            W_h = set_one(N_ll + j , N_ll + i , phys_params)
            expect_dict[name_Ch] = expect(W_h, eps_and_U, phys_params)
            expect_dict['C_h_' + str(j) + str(i)] = expect_dict[name_Ch]     
            

    # P-terms
    for i in range(0, N_ll):
        for j in range(0, N_ll):
            
            name_p = 'P_' + str(i) + str(j) 
            W_P = set_one(i, N_ll + j, phys_params)
            expect_dict[name_p] = expect(W_P, eps_and_U, phys_params)

     
    return expect_dict, eps_and_U[0]



def F_funct(n, m, Q):
    
    '''
    here I assume that Q = (0, Q), 
    i.e. Q has only y component
    
    Also, notice that this is exactly chi_ij^{l-m} in Eq.H18
    '''
    
    from scipy.special import genlaguerre 
    from numpy import exp, sqrt
    from scipy.special import factorial as fac
    
    f = 1
    f *= sqrt( fac(min(n,m)) / fac(max(n,m)) )
    f *= ((np.sign(m - n) * Q) / sqrt(2))**(max(n,m) - min(n,m))
    f *= exp(-Q**2/4)
    f *= genlaguerre(min(n,m), max(n,m) - min(n,m))(Q**2 / 2)
    
    return f


def M_integral(n1, m1, n2, m2, l, Q):
    
    from scipy.special import hyp1f1 as hyp
    from scipy.special import genlaguerre
    from numpy import exp, pi, sqrt
    from scipy.special import factorial as fac
    from scipy.special import factorial2 as fac2
    
    # first term
    M_1 = sqrt( pi * fac(m1) * fac(m2) / ( 2 * fac(n1) * fac(n2) ) )

    # second term 
    M_2 = Q**l * exp( - Q**2 / 2 )

    # third term
    M_3 = (sqrt(2))**(m1 + m2 - n1 - n2 - 2*l)

    # fourth term requires summation
    M_4 = 0
    # if m_i = 0, then range(m_i) doesn't return anything, while range(m_i+1) returns zero as we need
    for k1 in range(m1+1):
        for k2 in range(m2+1):

            t = n1 + n2 - m1 - m2 + 2 * (k1 + k2)
            x = 1
            x *= 2**(- k1 - k2)
            x *= genlaguerre(m1, (n1 - m1)).c[::-1][k1]
            # [::-1] because coefficients returned for x^n, x^(n-1), while we need for 1, x, x^2, ...
            x *= genlaguerre(m2, (n2 - m2)).c[::-1][k2]
            x *= fac2(t + l - 1) / fac(l)
            x *= hyp((1 + l - t)/2, l + 1, Q**2 / 2)

            M_4 += x 

    # now we need to multiply all the terms
    M = M_1 * M_2 * M_3 * M_4


    return M


def X_integral(n1, m1, m2, n2, Q):
    
    from numpy import exp, pi, sqrt
    
    l = n1 - m1 - n2 + m2
    s1 = max(n1, m1)
    s2 = max(n2, m2)
    i1 = min(n1, m1)
    i2 = min(n2, m2)
    
    X  = (1 / (4 * pi)) * sqrt(2 / pi)
    X *= exp(1j * l * pi/2)
    X *= (float(np.sign(l)))**l
    X *= (1j)**(s1 - i1 - s2 + i2)
    X *= M_integral(s1, i1, s2, i2, abs(l), Q)
    
    
    return X


# should be run whenever Q changes!
def get_prefactors(phys_params):
    
    from numpy import exp, pi, sqrt
    
    N_ll = phys_params['number of LLs']
    Q = phys_params['Q-vector']
    
    # add number of variables to phys params
    phys_params['number of variables'] = 2*N_ll + N_ll* (N_ll - 1) + N_ll**2 + 1 

    f_F_finite_Q = np.zeros([N_ll, N_ll, N_ll, N_ll], dtype = complex)
    f_F_zero_Q = np.zeros([N_ll, N_ll, N_ll, N_ll], dtype = complex)
    
    chi_lm   = np.zeros([N_ll, N_ll], dtype = complex)
    
    
    for n in range(N_ll):
        for l in range(N_ll):
            
            # matter-light coupling term
            chi_lm[n,l] = F_funct(n, l, Q)
            
            for m in range(N_ll):
                for t in range(N_ll):
                    
                    # fock terms (prefactor in X_integral is already included)            
                    f_F_finite_Q[n, t, l, m] = X_integral(n, t, l, m, Q)    
                    f_F_zero_Q[n, t, l, m] = X_integral(n, t, l, m, 0)    
                    
            
            
    all_prefactors = {"f_Fock finite Q" : f_F_finite_Q,
                      "f_Fock zero Q"   : f_F_zero_Q,
                      "chi_matter_light" : chi_lm}
        
    return all_prefactors


# setting up free energy
def Free_energy(v, phys_params, coefficients):
    
    # aux function
    def C(e_or_h, i, j):

        if i != j:
            name = 'C_' + e_or_h + f'_{i}{j}'

        else:
            name = 'N_' + e_or_h + f'_{i}' 

        return name


    # auxiliary stuff
    n0 = phys_params['target charge density n_0']
    E_G = phys_params['gap energy']
    om0 = phys_params['photon cut-off freq']
    omc = phys_params['cyclotron freq']
    Q = phys_params['Q-vector']
    om_phot = om0 + 10**4 * (phys_params['cyclotron freq'] / 2) * Q**2
    T = phys_params['temperature']
    phi = v[-1] # phot_ampl_dens
    g = phys_params['matt-light coupling g_0']
    n_ex = phys_params['target ex dens']
    N_ll = phys_params['number of LLs']


    # LL energies
    E_e = np.zeros([N_ll])
    E_h = np.zeros([N_ll])
    for i in range(N_ll):
        E_e[i] = (0.5 + i) * omc + E_G 
        E_h[i]  = (0.5 + i) * omc 

    # eigenvalues of M and expectations
    expectations, eps = expectations_and_eps(v, phys_params)



    # constructing free energy

    # entropic term
    entrop = - np.einsum('i->', eps * Fermi(eps, T) + log_funct(eps, T)) / (2*np.pi)

    # non-interacting part
    nonint = 0
    for i in range(N_ll):
        name_e = 'N_e_' + str(i)
        name_h = 'N_h_' + str(i)
        nonint += (expectations[name_e] * E_e[i] + expectations[name_h] * E_h[i]) / (2*np.pi)
    nonint += om_phot * phi**2

    # l-m interaction
    chi_lm   = coefficients['chi_matter_light']
    lm_interaction = 0
    for i in range(N_ll):
        for j in range(N_ll):
            lm_interaction += g * phi * expectations[f'P_{i}{j}'] * chi_lm[i,j]

    # fixing exc dens and charge dens
    d_nex = 0
    d_nc = 0
    for i in range(N_ll):
        name_e = 'N_e_' + str(i)
        name_h = 'N_h_' + str(i)
        d_nex += 0.5 * (expectations[name_e] + expectations[name_h] )
        d_nc += expectations[name_e] - expectations[name_h]
    d_nex += phi**2/(2*np.pi) - n_ex
    d_nc -= n0
    fix_nex = phys_params['alpha_nex'] * d_nex**2
    fix_nc = phys_params['alpha'] * d_nc**2


    # Hartree terms

    # prefactors
    f_F_zero_Q = coefficients['f_Fock zero Q']
    f_F_fin_Q  = coefficients['f_Fock finite Q']

    HF = 0
    for n in range(N_ll):
        for l in range(N_ll):
            for m in range(N_ll):
                for t in range(N_ll):

                    HF -= f_F_zero_Q[n,t,l,m] * ( expectations[C('e', n, m)] * expectations[C('e', t, l)] + 
                                                  expectations[C('h', n, m)] * expectations[C('h', t, l)] ) 
                    
                    HF -= f_F_fin_Q[n,t,l,m] * 2 * expectations[f'P_{n}{l}']  * expectations[f'P_{t}{m}']


    # np.real just to remove zero imag part 
    F = np.real(entrop + nonint + lm_interaction + fix_nex + fix_nc + HF)       
    

    return F



# free energy of a normal state
def Free_energy_normal(var_s, phys_params, coefficients):
    
    variables = np.copy(var_s)
    N_ll = phys_params['number of LLs']
    
    # zero coherences and no photon
    variables[2*N_ll : ] = 0


    return Free_energy(variables, phys_params, coefficients)




# generating good initial conditions
def init(phys_params, coefficients):
    
    from scipy import optimize
    from scipy.optimize import minimize
    
    # free energy
    Free_en = lambda var_s: Free_energy(var_s, phys_params, coefficients)
    Free_en_norm = lambda var_s: Free_energy_normal(var_s, phys_params, coefficients)
    
    # initial conditions
    T_target = phys_params['temperature']
    omc = phys_params['cyclotron freq']
    E_G = phys_params['gap energy']
    N_v = phys_params['number of variables'] 
    N_ll = phys_params['number of LLs']

    var0 = np.zeros([N_v])
    for i in range(N_ll):
        var0[i] = (0.5 + i)*omc + E_G
        var0[i + N_ll] = (0.5 + i)*omc
    var0[2*N_ll:] = np.random.rand(len(var0) - 2*N_ll) # coherences and a photon

    
    # go from alpha = 0 to target alpha at high temp
    l_alpha = 100
    phys_params['temperature'] = 10
    alpha_target = phys_params['alpha']
    alpha_target_nex = phys_params['alpha_nex']
    alpha_array = np.linspace(0, alpha_target, l_alpha)
    alpha_array_nex = np.linspace(0, alpha_target_nex, l_alpha)
    for i in range(l_alpha):
        
        phys_params['alpha'] = alpha_array[i]
        phys_params['alpha_nex'] = alpha_array_nex[i]

        ee  = minimize(Free_en, var0, method='CG',
                            options={'disp': False, 'maxiter': 50000})
        var0 = ee.x
      
    # going from high to target temperature
    var0_coh  = np.copy(var0)
    var0_norm = np.copy(var0)
            
    l_forward_1 = 100
    log_targ = np.log(T_target)/np.log(10)
    temp_array_1 = np.logspace(10, log_targ, l_forward_1)
    temp_array_2 = np.logspace(log_targ, log_targ - 1, 20)
    temp_array = temp_array_1
    temp_array_backward = temp_array[::-1]
    l = len(temp_array)

    # going forward
    for i in range(l):

        phys_params['temperature'] = temp_array[i]

        # normal solution
        ee_norm = minimize(Free_en_norm, var0_norm, method='CG', 
                           options={'disp': False, 'maxiter': 50000})
        var0_norm = ee_norm.x
        E_norm = ee_norm.fun
        solution_norm = var0_norm

        # with mem forward calc-s
        ee_coh_forward  = minimize(Free_en, var0_coh, method='CG', 
                                   options={'disp': False, 'maxiter': 50000})
        var0_coh = ee_coh_forward.x
        E_forward = ee_coh_forward.fun
        solution_forward = var0_coh
        

    if E_norm < E_forward:
        E = E_norm
        solution = solution_norm
    else:
        E = E_forward
        solution = solution_forward

    phys_params['temperature'] = T_target
    
        
    return E, solution




def fflo_calc(Q_min, Q_max, tol=1e-5):
    """
    Golden-section search.

    Given a function f with a single local minimum in
    the interval [Q_min, Q_max], fflo_calc returns 
    1. Value of Q-vector minimizing free energy F(Q)
    2. Corresponding energy F(Q)
    3. Total solution
    """
    
    import math

    invphi = (np.sqrt(5) - 1) / 2  
    invphi2 = (3 - np.sqrt(5)) / 2  
    
    a = Q_min
    b = Q_max

    
    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))
    
    print('number of steps ' + str(n) + '\n')

    c = a + invphi2 * h
    d = a + invphi * h
    
    phys_params['Q-vector'] = c 
    yc = find_min(phys_params)[0]
    print('Qc = ' + str(c))
    print('F(Qc) = ' + str(np.round(yc,8)) + '\n')
    
    phys_params['Q-vector'] = d 
    yd = find_min(phys_params)[0]
    print('Qd = ' + str(d))
    print('F(Qd) = ' + str(np.round(yd,8)) + '\n')

    for k in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            
            phys_params['Q-vector'] = c
            yc = find_min(phys_params)[0]
            print('step k = ' + str(k+1))
            print('Qc = ' + str(c))
            print('F(Qc) = ' + str(np.round(yc,8)) + '\n')
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            
            phys_params['Q-vector'] = d
            yd = find_min(phys_params)[0]
            print('step k = ' + str(k+1))
            print('Qd = ' + str(d))
            print('F(Qd) = ' + str(np.round(yd,8)) + '\n')

    if yc < yd:
        Q = (a + d) / 2
    else:
        Q = (c + b) / 2
    
    phys_params['Q-vector'] = Q
    en_and_sol = find_min(phys_params)
    
    print('Finally, Q = ' + str(Q) + ', F = ' + str(np.round(en_and_sol[0],8)))
    
    return Q, en_and_sol



# performs minimization by trying noisy initial conditions
def find_min(phys_params):
    
    Free_en = lambda var_s: Free_energy(var_s, phys_params, coefficients)

    optimize = lambda function, variable: minimize(function, variable, method='CG',
                                                        options={'gtol': 1e-12, 'disp': False, 'maxiter': 50000})
    
    # precalculations
    coefficients = get_prefactors(phys_params)
    
    # simulated annealing
    Energy_no_mem, solution_no_mem = init(phys_params, coefficients)
    
    # extra optimization
    num_of_attempts = 5 
    
    Energy = np.zeros([num_of_attempts])
    Solution = np.zeros([num_of_attempts, len(solution_no_mem)])
    
    # generate a bunch of initial conditions and minimize
    for i in range(num_of_attempts):
        try_min = optimize(Free_en, solution_no_mem + np.random.rand(len(solution_no_mem))/10 )
        Energy[i] = try_min.fun
        Solution[i,:] = try_min.x
    
    i_min = np.argmin(Energy)
    min_energy = Energy[i_min]
    min_sol = Solution[i_min]
    
    return min_energy, min_sol




# funtion for visualization
def vizual(solution, phys_params):
    
    all_expect = expectations_and_eps(solution, phys_params)[0]
    N_l = phys_params['number of LLs']
    
    from math import factorial as fac

    p_x = p_y = phys_params['grid for plotting']

    phi_ee_sq = np.zeros([len(p_x), len(p_y)], dtype = complex)
    phi_hh_sq = np.zeros([len(p_x), len(p_y)], dtype = complex)
    phi_eh_sq = np.zeros([len(p_x), len(p_y)], dtype = complex)

    for x in range(len(p_x)):
        for y in range(len(p_y)):

            alpha = p_x[x] + 1j*p_y[y]
            alpha_star = np.conj(alpha)

            first = np.exp( - abs(alpha)**2 )

            second_e = 0
            second_h = 0
            second_eh = 0
            
            for n in range(N_l):
                for m in range(N_l):
                    name_eh = f'P_{n}{m}'
                    if n == m:
                        name_ee = f'N_e_{n}'  
                        name_hh = f'N_h_{n}'
                    else:
                        name_ee = f'C_e_{n}{m}'
                        name_hh = f'C_h_{n}{m}'

                    second_e  += all_expect[name_ee] * alpha**m * alpha_star**n / np.sqrt( fac(n) * fac(m) )
                    second_h  += all_expect[name_hh] * alpha**m * alpha_star**n / np.sqrt( fac(n) * fac(m) )
                    second_eh += all_expect[name_eh] * alpha**m * alpha_star**n / np.sqrt( fac(n) * fac(m) )


            phi_ee_sq[x,y] = first * second_e
            phi_hh_sq[x,y] = first * second_h
            phi_eh_sq[x,y] = first * second_eh

    
    result = { 'el distrib' : phi_ee_sq, 
               'hole distrib' : phi_hh_sq,
               'coherence' : phi_eh_sq, 
               'y, x grid' : np.meshgrid(p_x, p_x), 
               'p-grid' : p_x
                }


    return result


# assigns a label to a state
def which_state(solution, Q_vector, phys_params):
    
    # get 2D coherence and el distribut
    p_x = phys_params['grid for plotting']
    vis = vizual(solution, phys_params)
    coh = vis['coherence']
    
    
    # if Q is finite (of order of 1), then we call is FFLO
    if Q_vector > 0.5:
        # check (absolute) phase winding
        phase = np.angle(coh)
        aa = len(p_x)//2 + 5
        bb = len(p_x)//2 - 5
        W_phi = 0
        for i in range(bb,aa):
            if abs(phase[aa,i] - phase[aa,i-1]) > 4:
                W_phi += 1
            if abs(phase[bb,i] - phase[bb,i-1]) > 4:
                W_phi += 1 
            if abs(phase[i,aa] - phase[i-1,aa]) > 4:
                W_phi += 1
            if abs(phase[i,bb] - phase[i-1,bb]) > 4:
                W_phi += 1
   
        return W_phi
    
    if Q_vector < 0.5:
        
        # coherent or not 
        if np.einsum('ij->', np.abs(coh)) == 0:
            return 0
        else:
            
            # characterise anisotropy
            e = 0
            ne = 0

            o = 0
            no = 0
            
            for i in range(1, 10):
                
                eith = phys_params['eith_' + str(i)]
                l = np.einsum('ij,ij->', np.abs(coh), eith) / np.einsum('ij->', np.abs(coh))
                l = np.round(l,5)

                if i % 2 == 0:
                    e += abs(l)**2
                    ne += 1
                if i % 2 != 0:
                    o += abs(l)**2
                    no += 1

                #print(f'l_{i} = ' + str(l))

            e = np.sqrt(e/ne)
            o = np.sqrt(o/no)

            if (e > 1e-3 and o < 1e-3) or (e < 1e-3 and o > 1e-3):
                # brakes rotational symmetry, but not inversion
                return -1 
            if (e > 1e-3 and o > 1e-3 and e != o):
                # brakes rot and inv symmetry
                return -2
            if (e < 1e-3 and o < 1e-3):
                # no rot/inv symm breaking
                return -3 
        
# just generates e^{i n k l }
def eiltheta(k, l):
    e = np.zeros([len(k), len(k)], dtype = complex)
    for i in range(len(k)):
        k_y = k[i]
        for j in range(len(k)):
            k_x = k[j]
            if l == 0:
                e[i,j] = 1.0
            else:
                if k_x == 0.0:
                    e[i,j] = 0
                else:
                    if l%2 == 1:
                        if k_x > 0:
                            e[i,j] = np.exp( 1j * l * np.arctan(k_y/k_x) )
                        else:
                            e[i,j] =  - np.exp( 1j * l * np.arctan(k_y/k_x) ) 
                    else:
                        e[i,j] = np.exp( 1j * l * np.arctan(k_y/k_x) )
    return e

