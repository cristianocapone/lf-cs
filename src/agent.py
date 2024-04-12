import numpy as np
from optimizer import Adam

class LFCS:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['N'], par['I'], par['O'], par['T']

        self.dt = par['dt']

        self.itau_m = np.exp (-self.dt / par['tau_m'])
        self.itau_s = np.exp (-self.dt / par['tau_s'])
        self.itau_ro = np.exp (-self.dt / par['tau_ro'])
        self.itau_star = np.exp (-self.dt / par['tau_star'])

        self.dv = par['dv']

        self.hidden = par['hidden_steps'] if 'hidden_steps' in par else 1

        # This is the network connectivity matrix
        self.J = np.random.normal (0., 1 / np.sqrt(self.N), size = (self.N, self.N))

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_Jin'], size = (self.N, self.I))
        self.Jout = np.random.normal (0., par['sigma_Jout'], size = (self.O, self.N))
        self.JoutV = np.random.normal (0., par['sigma_JoutV'], size = (1, self.N))
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O))

        # This is the hint signal
        try:
            self.Hint = par['hint_shape']
            self.Jhint = np.random.normal (0., par['sigma_hint'], size = (self.N, self.H))
        except KeyError:
            self.Hint = 0
            self.Jhint = None


        # Remove self-connections
        np.fill_diagonal (self.J, 0.)

        # Impose reset after spike
        self.s_inh = -par['s_inh']
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h']

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo']
        self.Vo = par['Vo']

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N)
        self.S_hat = np.zeros (self.N)

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N)
        self.policy_thr_tau = par['policy_thr_tau'] if 'policy_thr_tau' in par else 20

        # Check whether output should be put through a sigmoidal gate
        self.outsig = par['outsig'] if 'outsig' in par else False

        # Save the way policy should step in closed-loop scenario
        self.step_mode = 'amax'#par['step_mode'] if 'step_mode' in par else 'UNSET'
        
        # Set the optimizer for the weights
        self.adam_rec  = Adam(alpha=par['alpha_rec'], drop = .99, drop_time = 10000)
        self.adam_out  = Adam(alpha=par['alpha_out'], drop = .99, drop_time = 10000)
        self.adam_outV = Adam(alpha=par['alpha_outV'], drop = .99, drop_time = 10000)

        self.surrogate = 0

        # Here we save the params dictionary
        self.par = par

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv

        out = np.zeros (x.shape)
        mask = x > 0
        out [mask] = 1. / (1. + np.exp (-y [mask]))
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]))

        return out
    
    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv))
    
    def step (self, inp):
        itau_m = self.itau_m
        itau_s = self.itau_s

        self.S_hat = self.S_hat * itau_s + self.S [:] * (1. - itau_s)

        self.H = self.H * itau_m + (1. - itau_m) * (self.J @ self.S_hat + self.Jin @ inp + self.h) + self.Jreset @ self.S
        
        self.lam = self._sigm ( self.H, dv = self.dv )
        self.dH = self.dH  * itau_m + self.S_hat * (1. - itau_m)
        
        self.S =  np.heaviside(self.lam - np.random.rand(self.N,),0)
        
        # Here we return the chosen next action
        action, out = self.policy (self.S)
        self.entropy = - np.sum(self.prob*np.log(self.prob))

        return action, out

    def step_det (self, inp):
        itau_m = self.itau_m
        itau_s = self.itau_s

        self.S_hat = self.S_hat * itau_s + self.S * (1. - itau_s)

        self.H = self.H * itau_m + (1. - itau_m) * (self.J @ self.S_hat + self.Jin @ inp + self.h) + self.Jreset @ self.S

        self.lam = self._sigm ( self.H, dv = self.dv )
        self.dH = self.dH  * itau_m + self.S_hat * (1. - itau_m)

        self.S = self._sigm (self.H, dv = self.dv) - 0.5 > 0.

        # Here we return the chosen next action
        action, out = self.policy (self.S)
        self.out = out
        self.action = action
        self.entropy = - np.sum(self.prob*np.log(self.prob))

        return action, out

    def learn (self, r):
        alpha_J = 0.01
        
        dJ = np.outer ((self.S - self.lam), self.dH)
        self.dJfilt = self.dJfilt*(1-alpha_J) + dJ

        self.J += 0.1*r*self.dJfilt
        self.J -= self.J*0.000

    def learn_error (self, r, ch = 2.):
        alpha_J = 0.002

        ac_vector = np.zeros((self.par["O"],))
        ac_vector[self.action] = 1
        dJ = np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)

        dJ_ent = - np.outer (self.Jout.T@(self.prob*np.log(self.prob ) + self.entropy)*self._dsigm (self.H, dv = 1.), self.dH)
        dJ_ent_out = - np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)

        dJ_out =  np.outer((ac_vector - self.out), self.state_out.T)

        self.dJfilt = self.dJfilt*(1-alpha_J) + dJ
        self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out

        self.dJ_aggregate += (r*self.dJfilt + ch*dJ_ent*0)
        self.dJout_aggregate += (r*self.dJfilt_out + ch*dJ_ent_out*0)

    def learn_error_ppo (self, a_old, gamma, p_old, p_new, action ,eps):
        ch = 0.0
        alpha_J = 1-gamma
        ac_vector = np.zeros((self.par["O"],))
        ac_vector[action] = 1
        
        ppo_mask = 1
        
        p_new = (p_new+0.01)/(1.+0.01*3)
        p_old = (p_old+0.01)/(1.+0.01*3)
        
        p_frac = p_new/p_old
        
        rho = p_frac[action]
        rho = np.clip(rho,1-eps,1+eps)

        self.p_filt = self.p_filt*alpha_J + rho

        if (p_frac[action] >= 1+eps) | (p_frac[action] <= 1 - eps):
            ppo_mask = 0

        dJ = np.outer (self.Jout.T@( (ac_vector - p_new)*(p_frac)*ppo_mask )*self._dsigm (self.H, dv = 1.), self.dH)

        self.prob = (self.prob + .001)/(1.+.001)
        self.entropy = - np.sum(self.prob*np.log(self.prob))

        dJ_ent     = - np.outer(self.Jout.T @ (self.prob*( np.log(self.prob ) + self.entropy))*self._dsigm (self.H, dv = 1.), self.dH)
        dJ_ent_out = - np.outer(self.prob   * (np.log(self.prob ) + self.entropy ), self.state_out.T)

        dJ_out =  np.outer((ac_vector - self.out)*(p_frac)*ppo_mask , self.state_out.T)
        
        self.dJ_aggregate += (a_old*self.dJfilt + ch*dJ_ent)
        self.dJout_aggregate += (a_old*self.dJfilt_out  + ch*dJ_ent_out)
        
        self.dJfilt = self.dJfilt*(1-alpha_J) + dJ
        self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out

    def learn_V(self, a_old, gamma, p_old, p_new, action ,eps):
        alpha_J = 1 - gamma
        
        dJ_outV =   self.state_out
        self.dJoutV_filt = self.dJoutV_filt*(1-alpha_J) + dJ_outV

    def update_J (self, r):
        np.fill_diagonal (self.J, 0.)

        self.Jout = self.adam_out.step (self.Jout, self.dJout_aggregate)
        
        self.dJ_aggregate=0
        self.dJout_aggregate=0

    def policy (self, state):
        self.state_out = self.state_out * self.itau_ro  + state * (1 - self.itau_ro)

        out = self.Jout @ self.state_out*10.*.5*.5

        self.value = self.JoutV @ self.state_out

        if self.outsig:
            out = np.exp(out) / np.sum(np.exp(out))
        
        prob = out
        self.prob = prob

        action = np.random.choice(len(out), p = prob)

        return int(action),out

    def policy_(self, state, mode = 'amax'):
        valid_modes = ('prob', 'amax', 'raw')

        self.state_out [:] = self.state_out [:] * self.itau_ro  + state [:] * (1 - self.itau_ro)
        out = self.Jout @ self.state_out

        # We apply the sigmoid if required
        out = self._sigm(out) if self.outsig else out

        # We select deterministic action based on mode
        if   mode == 'prob': det_act = np.random.choice(len(out), p = out / np.sum(out))
        elif mode == 'amax': det_act = np.argmax(out)
        elif mode == 'raw' : det_act = out
        else: raise ValueError(f'Unknown policy mode {mode}. Should be one of: {valid_modes}')

        action = det_act

        return action, out

    def reset (self, init = None):
        self.S [:] = init if init is not None else np.zeros (self.N)
        self.S_hat [:] = self.S [:] * self.itau_s if init is not None else np.zeros (self.N)

        self.state_out [:] *= 0

        self.H [:] = self.Vo

    def forget (self, J = None, Jout = None):
        self.J    = np.random.normal (0., self.par['sigma_Jrec'], size = (self.N, self.N)) if J    is None else J.copy()
        self.Jout = np.random.normal (0., self.par['sigma_Jout'], size = (self.O, self.N)) if Jout is None else Jout.copy()

    def save (self, filename):
        # Here we collect the relevant quantities to store
        data_bundle = (self.Jin, self.Jteach, self.Jout, self.J, self.JoutV, self.par)

        np.save (filename, np.array (data_bundle, dtype=object))

    @classmethod
    def load (cls, filename):
        data_bundle = np.load (filename, allow_pickle = True)

        Jin, Jteach, Jout, J, JoutV,par = data_bundle

        obj = cls(par)
        
        obj.Jin = Jin.copy ()
        obj.Jteach = Jteach.copy ()
        obj.Jout = Jout.copy ()
        obj.J = J.copy ()
        obj.JoutV = JoutV.copy()

        return obj