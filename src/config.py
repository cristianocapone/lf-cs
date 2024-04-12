dt = 0.001

PONG_V4_PAR_I4 = {
       'dt' : dt,
       'tau_m'  : 6 * dt,
       'tau_s'  : 4 * dt,
       'tau_ro' : 1. *dt,
       'tau_star' : dt,
       
       'N' : 500,
       'T' : 800,
       'I' : 4,
       'O' : 3,
       
	'dv' : 0.05 ,
       'Vo' : -4,
       'h' : -8,
       's_inh' : 100,
       
       'gamma' : .99,
       'lerp'  : 0.01,
       
       'sigma_Jrec' : 0.0,
       'sigma_Jout' : 0.001,
       
       'alpha_rec' : 0.0015,
       'alpha_out' : 0.0015,
       'alpha_outV' : 0.0015,

	'sigma_Jin' : 10.,
	'sigma_Jout' : 1.,
	'sigma_JoutV' : 0.,

       'sigma_teach' : 10.,
       'hidden_steps' : 1,
       
       'policy_thr_tau' : 1,
       
       'outsig' : True,
       'step_mode' : 'amax',
       
       'epochs'     : 0,
       'epochs_out' : 0,
       
       'clump'    : False,
       'validate' : False,
       'feedback' : 'diagonal',
       'verbose'  : True,
       'rank'     : None
}