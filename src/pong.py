# import gymnasium as gym
import os
import gym
import os.path
import numpy as np

from tqdm import trange
from copy import deepcopy

from argparse import Namespace
from argparse import ArgumentParser

from .agent import LFCS
from .utils import cat2act, parse_ram
from .config import PONG_V4_PAR_I4 as par

def test_policy(env : gym.Env, agent : LFCS) -> float:
    env.reset()
    agent.reset()

    TEST_R : float = 0
    agent.dH = np.zeros (agent.N)

    ram_all, r, done, *_ = env.step (0)
    ram = parse_ram(ram_all)

    for _ in range(20):
        action, _ = agent.step_det(ram/255)
        ram_all, r, done, *_ = env.step (0)
        ram = parse_ram(ram_all)

    time = 0
    frame = 0

    while not done:
        frame += 1
        time += 1
        ram = parse_ram(ram_all)

        action, _ = agent.step_det(ram/255)
        ram_all, r, done, *_ = env.step (cat2act(action))

        TEST_R += float(r)

    return TEST_R

def main(args : Namespace):
    for rep in range(args.n_rep):

        TIMETOT = 100
        N_ITER = int(np.floor(100000/TIMETOT)*2.5)


        env = gym.make('ALE/Pong-ram-v5', difficulty=0, render_mode='rgb_array')

        print (f'Pong: Observation space: {env.observation_space}')
        print (f'Pong: Action Meaning: {env.unwrapped.get_action_meanings()}') # type: ignore
        
        agent = LFCS(par)
        
        agent.forget()
        agent.reset()
        
        REWARDS = []
        REWARDS_MEAN = []
        REWARDS_STANDARD = []
        REWARDS_STANDARD_MEAN = []
        REWARDS_20 = []
        INTERACTIONS = []
        ENTROPY = []
        TOT_INTER = []

        S = []

        tot_interactions=0

        agent_old = deepcopy(agent)

        for iteration in trange(N_ITER):
            env.reset()
            agent.reset()
            agent_old.reset()
            R = []

            RR_standard=0
            RAM = []
            ADV = []
            ACT = []

            PRED_V0 = []

            agent.dH = np.zeros (agent.N)
            agent_old.dH = np.zeros (agent.N)

            RTOT = 0

            agent.dJfilt =0
            agent.dJfilt_out = 0
            agent.dJoutV_filt =0

            ram_all, r, done, trunc, info = env.step (0)
            #ram = ram_all[[49, 50, 51, 54]]
            ram = parse_ram(ram_all)
            ram_old = ram

            for skip in range(20):

                act_vec = np.zeros((3,))
                act_vec = act_vec*0
                act_vec[0]=1

                #action, out = agent.step_det(ram/255)
                #action_old, out_old = agent_old.step_det(ram/255)

                ram_all, r, done, *_ = env.step (0)

                ram_old = ram
                ram = parse_ram(ram_all)

                dram = ram - ram_old
                dram[np.abs(dram)>10]=0.

            time = 0
            frame = 0
            ifplot = 1
            entropy=0
            
            OUT = []
            S = []
            
            images = []
            
            vt = 0

            while not done and time<TIMETOT:
                frame += 1
                time += 1
                tot_interactions += 1

                if ifplot&(frame%2==0)&(iteration%500==0)&(iteration>0):
                    img = env.render()
                    images.append(img)
                ram_old = ram

                action , out = agent_old.step_det(ram/255)

                ACT.append(action)

                act_vec = np.copy(out)

                act_vec = act_vec*0
                act_vec[action]=1

                r_old = r

                ram_all, r, done, *_ = env.step (int(np.array([cat2act(action)])))

                PRED_V0.append(agent_old.value)
                
                gamma = 0.99
                vt_old = vt
                vt = agent_old.value
                a_old = r_old - vt_old + gamma*vt

                if time==1: a_old=0
                
                ADV.append(a_old)
                RAM.append(ram)

                ram = parse_ram(ram_all)

                entropy+=agent_old.entropy

                dram[np.abs(dram)>10]=0.

                OUT.append(out)
                S.append(agent.S[:])

                RR_standard += r
                
                RTOT +=r
                R += [r]

            surrogate = []

            for replay_step in range(args.n_replays):
                agent.reset()
                agent.dH = np.zeros (agent.N)

                agent.dJfilt =0
                agent.dJfilt_out = 0
                agent.dJoutV_filt =0

                agent.p_filt = 0
                agent.surrogate=0

                time=-1

                while time<TIMETOT-2:
                    time+=1
                    ram = RAM[time]
                    a_old = ADV[time]
                    adv = ADV[time+1]

                    action = ACT[time]
                    agent_old.prob = OUT[time]

                    _ = agent.step_det(ram/255)
                    agent.learn_error_ppo(a_old * args.awake_learn, gamma, agent_old.prob, agent.prob, action, args.eps)

                    agent.surrogate += agent.p_filt * adv

                    if replay_step==0:
                        agent.learn_V(a_old, gamma, agent_old.prob, agent.prob, action, args.eps)

                    if args.tau == 'fast':
                        agent.update_J(r)


                if args.tau == 'slow':
                    agent.update_J(r)

                surrogate.append(agent.surrogate)

            Value = np.zeros((len(S),))

            for tt in range(len(S)-2):
                t_ndx =  len(S) - tt -2
                Value[t_ndx] = R[t_ndx] + gamma* Value[t_ndx+1]

            REWARDS.append(RTOT)
            REWARDS_STANDARD.append(RR_standard)

            ENTROPY.append(entropy)

            if (iteration%500==0)&(iteration>0):
                    
                reward_collection = [test_policy(env, agent) for _ in range(10)]

                REWARDS_20.append(np.mean(reward_collection))
                INTERACTIONS.append(tot_interactions)

            if (iteration%1==0)&(iteration>0):
                agent.update_J(r)

            if (iteration%1==0)&(iteration>0):
                agent_old = deepcopy(agent)
                
            if (iteration%50 == 49)&(iteration>1):

                REWARDS_MEAN.append(np.mean(REWARDS[-50:]))
                TOT_INTER.append(tot_interactions)

                REWARDS_STANDARD_MEAN.append((np.mean(REWARDS_STANDARD[-50:])))

                savename = os.path.join(args.save_dir, f"tau-{args.tau}_rep_{rep}.npy")
                np.save(savename, REWARDS_MEAN)
            
if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument('-eps', required=False, type=float, default=0.2, help='Stiffness of the policy')
    parser.add_argument('-tau', required=False, type=str, choices=['fast', 'slow'], default='fast', help='Scale of update of the policy')
    parser.add_argument('-n_rep', required=False, type=int, default=5, help='Number of experiment repetitions')
    
    parser.add_argument('-n_replays', required=False, type=int, default=2, help='Number of offline iterations')
    parser.add_argument('-awake_learn', required=False, type=float, default=1, help='Learning rate of the policy')
    parser.add_argument('-save_dir', required=False, type=str, default='.', help='Directory to save the results')
    
    args = parser.parse_args()
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(args.save_dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(args.save_dir)
        print("The new directory is created!")
    
    main(args)
