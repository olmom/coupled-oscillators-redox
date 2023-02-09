import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton 
import scipy.interpolate
import statsmodels.api as sm


class PoincareOscillator:

    def __init__ (self, amp, lam, tau, eps, F_zg, K_coup):
        """Parameters of a Poincare oscillator"""
        self.amp = amp #amplitude of Poincare oscillator
        self.lam = lam #amplitude relaxation rate of Poincare osc.
        self.tau = tau #inner period of Poincare oscillator
        self.eps = eps #twist
        self.F_zg = F_zg #strength of TTFL input
        self.K_coup = K_coup #strength of mean-field coupling

    def compute_M(self, y0):
        """
        Coupling through mean field: the effective concentration of x
        can be approximated with the average level of all x_i signals 
        or mean field MF, which acts on individual oscillators at a 
        strength K_coup
        """
        mf = np.zeros_like(y0)
        xi, yi = y0[0::5], y0[1::5]
        mf_x, mf_y = mf[0::5], mf[1::5]

        n_oscs = xi.shape[0]
        MF     = np.sum(xi, axis=0) / n_oscs 

        meanfx = self.K_coup * (MF)
        mf_x[:] = meanfx
        return mf

    def f_PoincareGonze(self, y0, t): 
        """
        Deterministic Poincare system with twist,
        in Cartesian coordinates. Also the equations of 
        a Goodwin-like system (the Gonze model, see [1]) 
        are included to model the input of a CLOCK:BMAL 
        term in the Poincare x variable

        [1] Gonze et al. (2005) Biophys J
        """ 
        f = np.zeros_like(y0)
        xi = y0[0::5]
        yi = y0[1::5]
        Xgon = y0[2::5]
        Ygon = y0[3::5]
        Zgon = y0[4::5]

        fx = f[0::5]
        fy = f[1::5]
        fXgon = f[2::5]
        fYgon = f[3::5]
        fZgon = f[4::5]
            
        r       = np.sqrt(xi**2 + yi**2)
        phi_dot = 2*np.pi/self.tau + self.eps*(self.amp-r)
    
        fx[:] = self.lam*xi*(self.amp-r) - yi*phi_dot + \
                self.compute_M(y0)[0::5] 
        fy[:] = self.lam*yi*(self.amp-r) + xi*phi_dot + \
                self.F_zg *(Xgon/0.12296 - 1) 
                # the Xgon zeitgeber is normalized to its
                # relative amplitude value by dividing by the
                # mean of the Xgon oscillations(0.12296). Then 
                # -1 is subtracted to make it oscillate around 0,
                # like the Poincare variables
        fXgon[:] =  0.7*((1.**4)/(1.**4+ Zgon**4)) - \
                0.35*(Xgon/(1.+Xgon)) 
        fYgon[:] = 0.7*Xgon - 0.35*(Ygon/(1.+Ygon)) 
        fZgon[:] = 0.7*Ygon - 0.35*(Zgon/(1.+Zgon)) 
        
        return f


#############################
#############################


class KineticOscillator:

    def __init__ (self, a, b, d, e, p, q, K_coup, F_zg):
        """
        Parameters of the deterministic kinetic
        redox oscillator: see Del Olmo et al. (2019) Int J Mol Sci
        """
        self.a = a #hyperoxidation of A to I
        self.b = b #reduction of I to A
        self.d = d #D1 translocation to cytosol (D2)
        self.e = e #R translocation to mitochondria
        self.p = p #D1 production
        self.q = q #R degradation
        self.K_coup = K_coup #mean-field D2 coupling
        self.F_zg = F_zg #strength of TTFL input

    def compute_M_Kinetic (self, y0):
        """
        Coupling through mean field: the effective concentration 
        of D2 and X from the Gonze model (Gonze et al. 2005, Biophys J)
        can be approximated with the average level of all D2_i or 
        Xgon_i signals or mean field MF, which act on individual 
        oscillators at a strength K_coup
        """      
        mf = np.zeros_like(y0)
        D1i, D2i, Ri = y0[0::6], y0[1::6], y0[2::6]
        Xgon, Ygon, Zgon = y0[3::6], y0[4::6], y0[5::6]
        mf_D1, mf_D2, mf_R = mf[0::6], mf[1::6], mf[2::6]
        mf_Xgon, mf_Ygon, mf_Zgon = mf[3::6], mf[4::6], mf[5::6]

        n_oscs = D2i.shape[0]
        MF_D2  = np.sum(D2i, axis=0) / n_oscs - D2i
        MF_Xgon = np.sum(Xgon, axis=0) / n_oscs - Xgon 

        meanf_D2 = self.K_coup * (MF_D2)
        meanf_Xgon = self.K_coup * (MF_Xgon)
        mf_D2[:], mf_Xgon[:] = meanf_D2, meanf_Xgon
        return mf
    
    def Kinetic_Gonze(self, y0, t): 
        """
        Deterministic Kinetic redox ODE system and Gonze model [1],
        which is used to model the input of a CLOCK:BMAL 
        term in the production of D1

        [1] Gonze et al. (2005) Biophys J
        """ 
        f = np.zeros_like(y0)
        D1i = y0[0::6]
        D2i = y0[1::6]
        Ri  = y0[2::6]
        Xgon = y0[3::6]
        Ygon = y0[4::6]
        Zgon = y0[5::6]

        fD1 = f[0::6]
        fD2 = f[1::6]
        fR  = f[2::6]
        fXgon = f[3::6]
        fYgon = f[4::6]
        fZgon = f[5::6]
          
        Ai = self.b*Ri / (self.b*Ri + self.a*D1i)  
    
        fD1[:] = self.p - self.a*Ai*D1i - self.d*D1i + \
                self.F_zg * Xgon
        fD2[:] = self.d*D1i - self.e*D2i 
        fR[:] = self.e*D2i + self.e*(self.compute_M_Kinetic(y0)[1::6]) - self.q*Ri
        fXgon[:] =  0.7*((1.**4)/(1.**4+ Zgon**4)) - \
                0.35*(Xgon/(1.+Xgon)) 
        fYgon[:] = 0.7*Xgon - 0.35*(Ygon/(1.+Ygon)) 
        fZgon[:] = 0.7*Ygon - 0.35*(Zgon/(1.+Zgon))
        
        return f

#####################
#####################

class sdeIntegrator:

    def __init__ (self, sig_x, sig_y):
        """
        Parameters of Wiener process: 
        variances of noise terms 
        """
        self.sig_x = sig_x 
        self.sig_y = sig_y 

    def G_PoincareGonze(self, y0, t):
        """
        Noise coefficients acting in x and y coordinates
        of a Poincare oscillator
        """
        sig = np.zeros_like(y0)
        sig[0::5] = self.sig_x
        sig[1::5] = self.sig_y
        sig[2::5] = 0
        sig[3::5] = 0
        sig[4::5] = 0
        return sig

    def G_KineticGonze (self, y0, t):
        """
        Noise coefficients acting in D2 variable
        from the kinetic oscillator
        """
        sig = np.zeros_like(y0)
        sig[0::6] = 0
        sig[1::6] = self.sig_y
        sig[2::6] = 0
        sig[3::6] = 0
        sig[4::6] = 0
        sig[5::6] = 0
        return sig

    def deltaW(self, N, m, h, generator=None):
        """
        Generate sequence of Wiener increments for m independent Wiener
        processes W_j(t) j=0...m-1 for each of N time intervals of length h.
        Args:
          N (int): number of time steps
          m (int): number of Wiener processes
          h (float): the time step size
          generator (numpy.random.Generator, optional)
        Returns:
          dW (array of shape (N, m)): The [n, j] element has the value
          W_j((n+1)*h) - W_j(n*h)
        """
        if generator is None:
            generator = np.random.default_rng()
        return generator.normal(0.0, np.sqrt(h), (N, m))


    def itoEuler(self, f, G, y0, t, dW=None, m=None, generator=None):
        """
        Euler-Maruyama algorithm to integrate the Ito equation
        dy = f(y,t)dt + G(y,t) dW(t)
        where y is the d-dimensional state vector, f is a vector-valued function,
        G is an d x m matrix-valued function giving the noise coefficients and
        dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Wiener increments
        Args:
          f: callable(y, t) returning (d,) array
             Vector-valued function to define the deterministic part of the system
          G: callable(y, t) returning (d,m) array
             Matrix-valued function to define the noise coefficients of the system
          y0: array of shape (d,) giving the initial state vector y(t==0)
          t (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            t[0] is the intial time corresponding to the initial state y0.
          dW: optional array of shape (len(t)-1, d). This is for advanced use,
            if you want to use a specific realization of the d independent Wiener
            processes. If not provided Wiener increments will be generated randomly
          m: number of Wiener processes (integer)
          generator (numpy.random.Generator, optional) Random number generator
            instance to use. If omitted, a new default_rng will be instantiated.
        Returns:
          y: array, with shape (len(t), len(y0))
             With the initial value y0 in the first row
        Raises:
          SDEValueError
        See also:
          G. Maruyama (1955) Continuous Markov processes and  stochastic equations
          Kloeden and Platen (1999) Numerical Solution of Differential Equations
        """
        if generator is None and dW is None:
            generator = np.random.default_rng()
        N = len(t)
        h = (t[N-1] - t[0])/(N - 1)
        d = len(y0) #dimension d of the system
        # allocate space for result
        y = np.zeros((N, d), dtype=y0.dtype)
        if dW is None:
            # pre-generate Wiener increments (for m independent Wiener processes):
            dW = self.deltaW(N - 1, m, h, generator)
        y[0] = y0;
        for n in range(0, N-1):
            tn = t[n]
            yn = y[n]
            dWn = dW[n,:]
            y[n+1] = yn + f(yn, tn)*h + \
                    G(yn, tn)*dWn #G(yn, tn).dot(dWn)
        return y

#####################
#####################

class Fits:
    def __init__ (self):
        return

    def sqrt_func(t, a, b):
        """
        Fit a curve to a square-root function
        """
        return a*np.sqrt(b*t)
    
    def exp_func(t, a, b, c):
        """
        Fit a curve to an exponential function
        """        
        return a*np.exp(b*t) + c

    def lin_func_intercept0(t, a):        
        """
        Fit a linear function passing through origin
        """   
        return a*t

    def cos_func(t, A, w, phi):  
        """
        Fit a curve to a cosine function,
        NOTE: omega and phi are in radians
        """
        return A * np.cos(w*t + phi) 

    def MM_func(x, k1, k2):
        """
        Fit a curve to a Michaelis-Menten-like function
        """   
        return k1*x / (k2+x)

    def sigmoidal_func(x, k1, k2, n):
        """
        Fit a curve to a Hill-like sigmoidal function
        """          
        return k1*x**n / (k2+x**n)

    def smooth_curve(x, y, xgrid, frac, it, random_percentage):
        """
        Smooth a curve fitting a Lowess model in a fraction of the 
        points (fraction given by the argument random_percentage)
        Can also provide confidence interval around the Lowess fit
        Arguments:
            x: xdata on which Lowess fit should be performed
            y: ydata on which Lowess fit should be performed
            xgrid: specifies grid in which solution should be interpolated
            frac: fraction of data used when estimating each y-value
            it: number of residual-based reweightings to perform
            random_percentage: fraction of points for each Lowess fit
        see: https://james-brennan.github.io/posts/lowess_conf/
        """
        lowess = sm.nonparametric.lowess
        samples = np.random.choice(len(x), random_percentage, replace=True)
        y_s = y[samples]
        x_s = x[samples]
        y_sm = lowess(y_s,x_s, frac=frac, it=it,
                      return_sorted = False)
        # regularly sample it onto the grid
        y_grid = scipy.interpolate.interp1d(x_s, y_sm, 
                                            fill_value='extrapolate')(xgrid)
        return y_grid


#####################
#####################

class RhythmicParameters:

    def __init__ (self, interp_length=15):
        self.interp_length = interp_length
        return 
        
    def determine_zeros(self, t, singleosc):                    
        """
        Compute zeros of a single oscillation through Newton method:
        the Newton method takes a point slightly bigger than
        the value "fed" to the function and another slightly smaller. Then
        computes the line that joins al three points together and calculates
        root (iterating this algorithm)
        NOTE that the try/except is done because sometimes the line that
        joins 3 points together is almost parallel to x axis. In this case
        it is not able to calculate the root, and a ValueError arises (a
        value in x_new is above/below the interpolation range
        """ 
      
        singleosc = np.asarray(singleosc)
        results = []
        l = len(t)                         
        mask = np.diff(singleosc >= 0.0)
        idx = np.arange(l, dtype=int)       
    
        for m in np.where(np.diff(singleosc >= 0 ) == True)[0]:
            res = []
            if idx[m] < (self.interp_length) or \
                    idx[m] > l - (self.interp_length):
                continue             
            min_i = max(idx[m] - self.interp_length, 0)   
            max_i = min(idx[m] + self.interp_length, l)
    
            time =         t[min_i:max_i]
            data = singleosc[min_i:max_i]
    
            t0 = time[0] #shift interval to avoid error by large numbers
            time = time - t0
    
            interpol = interp1d(time,data,kind='cubic',axis=0)
            try:
                root_t = newton(interpol, t[m]-t0)    
                singleosc_root = interpol(root_t).tolist()
                res.append([root_t+t0,singleosc_root])
                res = np.asarray(res[0])
    
            except ValueError:
                pass
            except RuntimeError:
                pass
            results.append(res)
        results = np.asarray(results)
        return results

    def determine_period_singleosc(self, t, singleosc):        
        """
        Determination of the period of a single oscillator:
        Two times the distance between two consecutive zeros
        """
        period = []
        times = np.hstack(self.determine_zeros(t, singleosc))[0::2]
        halfperiod = np.diff(times)
        halfperiod = halfperiod[halfperiod > 2] #filter close 0 crossings
        halfperiod_avg = halfperiod.mean()
        period = 2*halfperiod_avg
        return period

    def periods(self, t, x):
        """
        Determine periods of N oscillators and return the mean and std dev
        """
        try:
            results = []
            for so in range(np.shape(x)[1]):
                period = self.determine_period_singleosc(t,x.iloc[:,so]).tolist()
                results.append(period)
            mean = np.mean(results)
            std  = np.std(results)
            return np.array(results), mean, std

        except IndexError:
            period = self.determine_period_singleosc(t,x)
            mean   = period.copy()
            std    = 0.0
            return np.array(period), mean, std

    def autocovariance_signal (self, x, N, k):
        """
        Compute autocovariance signal, as in [2]:
        x = signal (can be signal with or without noise)
        N = number of samples
        k = sampling interval = tau (correlation time) / delta_t

        [2] Westermark (2009) PLoS Comp Biol
        """
        
        sum_m0 = 0
        sum_mk = 0
        for i in range(N-k):
            sum_m0 = sum_m0 + x[i]
            sum_mk = sum_mk + x[i+k]
        m0 = sum_m0 / (N-k) # unbiased mean
        mk = sum_mk / (N-k) # unbiased mean @ +k units of time
    
        sumatorio = 0
        for i in range(N-k):
            sumatorio = sumatorio + ( (x[i] - m0) * (x[i+k] - mk) )
         
        C_k = sumatorio / (N-k)
    
        return C_k
    
