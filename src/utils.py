import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def save_frames_as_gif(frames, filename):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(filename, writer='imagemagick', fps=60)
    plt.close()

def action2cat(action : np.ndarray, n_actions : int = 3) -> np.ndarray:
    # Here we reduce the action space from 6D to 3D thanks to the reduction
    # NOOP = FIRE | RIGHT = RIGHTFIRE | LEFT = LEFTFIRE
    action[action == 1] = 0 # Fire -> NoOp
    action[action == 4] = 2 # RightFire -> Right
    action[action == 5] = 3 # LeftFire -> Left

    # Convert action to be categorical: one-hot encoding.
    # ? NOTE: For indexing we would require numbers from
    # ?       from 0 to 2, however actions are coded as
    # ?       {0 : NOOP, 2 : RIGHT, 3 : LEFT}, we thus
    # ?       introduce an encoding & decoding map.
    act_idx_map = {0 : 0, 2 : 1, 3 : 2}

    action_cat = np.eye(n_actions)[[act_idx_map[int(act)] for act in action]]

    return action_cat

def act2cat(act : int) -> int:
    dic_map = {0 : 0, 1 : 0, 2 : 1, 3 : 2, 4 : 1, 5 : 2}

    return dic_map[act]

def cat2act(cat : int) -> int:
    dic_map = {0 : 0, 1 : 2, 2 : 3}

    return dic_map[cat]

def parse_ram(obs : np.ndarray) -> np.ndarray:
    ram = np.zeros((4,))
    ram[0] = int(obs[49])
    ram[1] = int(obs[50])
    ram[2] = int(obs[51])
    ram[3] = int(obs[54])

    return ram

