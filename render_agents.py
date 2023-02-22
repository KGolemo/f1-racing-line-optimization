import numpy as np
import pygame, os
from race import Race
import time


def event_to_action(eventlist):
    global run
    for event in eventlist:
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            env.reset()


# set the path of the states
# all states in the folder will be rendered
states_path = os.path.join('exported_states','')    


filelist = [file for file in os.listdir(states_path) if file.endswith('.npy')]
print('loading these checkpoints:')
print(filelist)
filepathlist = [os.path.join(states_path,file) for file in filelist]

numpy_import = np.load(filepathlist[0])   
statematrix = np.ones((numpy_import.shape[0], 6, len(filepathlist)))
for i, file in enumerate(filepathlist):
    numpy_import = np.load(file)
    statematrix[:,:,i] = numpy_import
    # change collided rockets to black color black (state = 0)
    statematrix[1:, -1, :] = np.where(statematrix[1:, 0, :] == 0, 0, statematrix[1:, -1, :])

# ─── RUN GAME FOR HUMAN MODE ────────────────────────────────────────────────────
env_config = {
    # "max_frames": 1000,
    # "export_frames": True,
    "export_states": False,
    # "export_string": 'render',,
    'gui': False,
    'gui_draw_background': False,
    'gui_draw_goal_all': False,
    'gui_draw_goal_next': False,
    }

if __name__ == "__main__":
    env = Race(env_config)
    env.rule_collision = False
    env.rule_timelimit = False
    env.render()
    run = True
    counter = 0
    while run:
        env.clock.tick(30)
        if counter == 0:
            time.sleep(5)
        get_event = pygame.event.get()
        event_to_action(get_event)
        matrix = statematrix[counter,:,:].T
        if counter > 0:
            if np.count_nonzero(matrix[:,0].astype(int)) == 0:
                run = False
        env.set_spectator_state(matrix, frame=counter)
        env.render()
        counter += 1
    pygame.quit()