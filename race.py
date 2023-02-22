import enum
import gym
import numpy as np
import utils
from car import Car
from track import Track
import random


TRACK_SIZE = 2463, 1244
MAIN_WINDOW_SIZE = int(TRACK_SIZE[0]/1.3), int(TRACK_SIZE[1]/1.3)

START_POS = 1501, 870 #1490/1.3, 1150/1.3

F1_PURPLE = (181, 22, 177)
F1_GREEN = (38, 218, 48)
F1_YELLOW = (244, 246, 59)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (80, 80, 80)
RED = (255, 0, 0)

ECHO_VECTOR_COLOR = GRAY
ECHO_COLLISION_COLOR = RED
GOAL_COLOR = F1_GREEN
NEXT_GOAL_COLOR = GRAY
FINISH_COLOR = WHITE
TRACK_COLOR = F1_YELLOW

FONT_SIZE = 24
FONT_COLOR = BLACK
FONT_BACKGROUND = WHITE

class Race(gym.Env):
    def __init__(self, env_config={}):
        self.parse_env_config(env_config)
        self.win = None
        self.action_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(2,),
            dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(8,),
            dtype=np.float32)

        self.track = Track(self)
        self.car = Car(self, self.track)
        self.spectator = None

        self.reset()
        # exit()

    def parse_env_config(self,env_config):
        keyword_dict = {
            # these are all available keyboards and valid values respectively
            # the first value in the list is the default value
            'gui'                   : [True, False],
            'camera_mode'           : ['fixed','centered'],
            'env_name'              : ['default', 'wide', 'extra'],
            'env_visible'           : [True, False],                 
            'export_frames'         : [False, True],                       # export rendered frames
            'export_states'         : [False, True],                       # export every step
            'export_string'         : ['', 'any', str],                    # string for export filename
            'export_highscore'      : [0, 'any', int],                     # only export if highscore is beat
            'max_steps'             : [1000, 'any', int],
            'max_laps'              : [1, 2, 3],
            'reward_mode'           : ['laptime', 'level'],
            'rule_collision'        : [True, False],
            'rule_max_steps'        : [False, True],
            'rule_max_laps'         : [True, False],
            'rule_keep_on_screen'   : [False, True],
            'gui_input'             : [True, False],
            'gui_state'             : [True, False],
            'gui_observations'      : [True, False],
            'gui_draw_background'   : [True, False],
            'gui_draw_echo_points'  : [True, False],
            'gui_draw_echo_vectors' : [True, False],
            'gui_draw_goal_all'     : [True, False],
            'gui_draw_goal_next'    : [True, False],
        }
        
        # ─── STEP 1 GET DEFAULT VALUE ────────────────────────────────────
        assign_dict = {}
        for keyword in keyword_dict:
            # asign default value form keyword_dict
            assign_dict[keyword] = keyword_dict[keyword][0]
            
        # ─── STEP 2 GET VALUE FROM env_config ─────────────────────────────
        for keyword in env_config:
            if keyword in keyword_dict:
                # possible keyword proceed with assigning
                if env_config[keyword] in keyword_dict[keyword]:
                    # valid value passed, assign
                    assign_dict[keyword] = env_config[keyword]
                elif 'any' in keyword_dict[keyword]:
                    # any value is allowed, assign if type matches
                    if isinstance(env_config[keyword],keyword_dict[keyword][2]):
                        print('type matches')
                        assign_dict[keyword] = env_config[keyword]
                    else:
                        print('error: wrong type. type needs to be: ', keyword_dict[keyword][2])
                else:
                    print('given keyword exists, but given value is illegal')
            else:
                print('passed keyword does not exist: ',keyword)

        # ─── ASSIGN DEFAULT VALUES ───────────────────────────────────────
        self.camera_mode           = assign_dict['camera_mode']
        self.env_name              = assign_dict['env_name']
        self.env_visible           = assign_dict['env_visible']
        self.export_frames         = assign_dict['export_frames']
        self.export_states         = assign_dict['export_states']
        self.export_string         = assign_dict['export_string']
        self.export_highscore      = assign_dict['export_highscore']
        self.max_steps             = assign_dict['max_steps']
        self.max_laps              = assign_dict['max_laps']

        self.reward_mode           = assign_dict['reward_mode']

        self.rule_collision        = assign_dict['rule_collision']
        self.rule_max_steps        = assign_dict['rule_max_steps']
        self.rule_max_laps         = assign_dict['rule_max_laps']
        self.rule_keep_on_screen   = assign_dict['rule_keep_on_screen']

        self.gui                   = assign_dict['gui']
        self.gui_input             = assign_dict['gui_input']
        self.gui_state             = assign_dict['gui_state']
        self.gui_observations      = assign_dict['gui_observations']
       
        self.gui_draw_background   = assign_dict['gui_draw_background']
        self.gui_draw_echo_points  = assign_dict['gui_draw_echo_points']
        self.gui_draw_echo_vectors = assign_dict['gui_draw_echo_vectors']
        self.gui_draw_goal_all     = assign_dict['gui_draw_goal_all']
        self.gui_draw_goal_next    = assign_dict['gui_draw_goal_next']
        

    def reset(self):            
        if self.camera_mode == 'centered':
            self.track.load_level()

        # ─── RESET EXPORT VARIALBES ──────────────────────────────────────
        # give unique session id for export
        self.session_id = str(int(np.random.rand(1)*10**6)).zfill(6)
        # dim0 : n_steps | dim1 : frame, x,y,ang,vel,action
        # self.statematrix = np.zeros((self.max_steps, 5))
        self.statematrix = np.zeros((self.max_steps, 6))

        # ─── RESET car ──────────────────────────────────────────────────
        self.reset_car_state()
        # generate observation
        self.car.update_observations()
        distances = self.car.echo_collision_distances_interp
        velocity = self.car.vel_interp
        observations = np.concatenate((distances, np.array([velocity])))
        return observations

    def set_spectator_state(self, state, frame=None):
        self.car.visible = False
        self.spectator = state
        if frame:
            self.car.framecount_total = frame

    def reset_car_state(self, x=START_POS[0], y=START_POS[1], ang=-92, vel_x=0, vel_y=0, level=0):  # ang=1e-10
        # if camera_mode is centerd, the car needs to go center too
        if self.camera_mode == 'centered':
            diff_x = MAIN_WINDOW_SIZE[0]/2  - x
            diff_y = MAIN_WINDOW_SIZE[1]/2 - y
            # move environment
            self.track.move_env(-diff_x,-diff_y)
            # move player
            x, y = MAIN_WINDOW_SIZE[0]/2, MAIN_WINDOW_SIZE[1]/2
        return self.car.reset_game_state(x, y, ang, vel_x, vel_y, level)

    def set_done(self):
        self.car.done = True
        self.car.action_state = 0
        if (self.export_states) and (self.car.reward_total > self.export_highscore):
            import os
            # copy last state to remaining frames
            i = self.car.framecount_total
            n_new = self.max_steps - i
            # self.statematrix[i:, :] = np.repeat(self.statematrix[i-1, :].reshape((1, 5)), n_new, axis=0)
            self.statematrix[i:, :] = np.repeat(self.statematrix[i-1, :].reshape((1, 6)), n_new, axis=0)
            # mark at which frame agent is done
            self.statematrix[i:, 0] = 0

            # export
            filename = '_'.join([self.export_string,
                                 '-'.join([self.session_id, str(int(self.car.reward_total)).zfill(4)])
                                 ])
            filenamepath = os.path.join('exported_states', filename)
            np.save(filenamepath, self.statematrix)

    def step(self, action=[0, 0]):
        # ─── NORMALIZE ACTION ────────────────────────────────────────────
        # action = [action_turn, action_acc]
        action[0] = max(min(action[0], 1), -1)
        action[1] = max(min(action[1], 1), -1)
        self.car.action = action.copy()

        # ─── PERFORM STEP ────────────────────────────────────────────────
        if not self.car.done:
            self.car.move(action)
            self.car.update_echo_vectors()
            if self.rule_collision:
                self.car.check_collision_goal()
                self.car.check_collision_echo()
                self.car.check_collision_track()
            self.car.update_observations()

            # ─── EXPORT GAME STATE ───────────────────────────────────────────
            if self.export_states:
                i = self.car.framecount_total
                self.statematrix[i, :] = [i, self.car.position.x, self.car.position.y, self.car.angle, self.car.net_velocity.y, self.car.action_state]

            self.car.framecount_goal += 1
            self.car.framecount_total += 1

            if self.reward_mode == 'level':
                self.car.update_reward_level()
            elif self.reward_mode == 'laptime':
                self.car.update_reward_laptime()

            # if (self.car.level - self.car.level_previous == -1) or (self.car.level == self.track.n_goals-1 and self.car.level_previous == 0):
            #     self.set_done()

            if self.rule_max_steps:
                if self.car.framecount_total == self.max_steps-1:
                    self.set_done()
            if self.rule_max_laps:
                if self.car.framecount_total == self.max_steps-1:
                    self.set_done()
                if self.car.level == 1 and self.car.n_lap == self.max_laps:
                    self.set_done()

        # ─── GET RETURN VARIABLES ────────────────────────────────────────
        distances = self.car.echo_collision_distances_interp
        velocity = self.car.vel_interp

        observations = np.concatenate((distances, np.array([velocity])))

        reward = self.car.reward_step
        done = self.car.done
        info = {
            "x": self.car.position.x,
            "y": self.car.position.y,
            "ang": self.car.angle}

        # ─── RESET ITERATION VARIABLES ───────────────────────────────────
        self.car.reward_step = 0
        self.car.level_previous = self.car.level
        return observations, reward, done, info

    def render(self, mode=None):
        # initialize pygame only when render is called once
        import pygame
        import os
        from PIL import Image

        middle_echo_index = (self.car.N_SENSORS - 1) // 2

        def init_renderer(self):
            pygame.display.set_caption('Racing line optimization using reinforcement learning')
            self.clock = pygame.time.Clock()
            self.win = pygame.display.set_mode(MAIN_WINDOW_SIZE)
            self.FERRARI_IMG = utils.scale_image(pygame.image.load('imgs/ferrari.png').convert_alpha(), 0.1)
            self.REDBULL_IMG = utils.scale_image(pygame.image.load('imgs/redbull.png').convert_alpha(), 0.1)
            self.MERCEDES_IMG = utils.scale_image(pygame.image.load('imgs/mercedes.png').convert_alpha(), 0.1)
            self.CAR_IMG = random.choice([self.FERRARI_IMG, self.REDBULL_IMG, self.MERCEDES_IMG])
            self.CAR_IMG_OFF = utils.scale_image(pygame.image.load('imgs/ferrari_off.png').convert_alpha(), 0.1)
            self.BG_IMG = utils.scale_image(pygame.image.load('imgs/Monza_background.png').convert_alpha(), 1/1.3)
            self.TRACK_IMG = utils.scale_image(pygame.image.load('imgs/Monza_track_extra_wide_2.png').convert_alpha(), 1/1.3)

            pygame.init()
            
            if self.export_frames:
                self.display_surface = pygame.display.get_surface()
                self.image3d = np.ndarray((MAIN_WINDOW_SIZE[0], MAIN_WINDOW_SIZE[1], 3), np.uint8)
                
            self.gui_interface = []
            if self.gui_observations:
                self.gui_interface.append('Distances')
                self.gui_interface.append('Velocity')
            if self.gui_input:
                self.gui_interface.append('Acceleration')
                self.gui_interface.append('Steering')
            if self.gui_state:
                self.gui_interface.append('Total_reward')
                self.gui_interface.append('Level')
                self.gui_interface.append('Lap')
                self.gui_interface.append('Total_frames')
                # self.gui_interface.append('Position')
                # self.gui_interface.append('Angle')

        def draw_level():
            pygame.draw.lines(self.win, TRACK_COLOR, False, self.track.line1_list, 4)
            pygame.draw.lines(self.win, TRACK_COLOR, False, self.track.line2_list, 4)

        def draw_goal_next():
            goal = tuple(self.track.goals[self.car.level, :])
            pygame.draw.lines(self.win, GOAL_COLOR, False,
                              (goal[0:2],goal[2:4]), 4)

        def draw_finish_line():
            finish = tuple(self.track.goals[0, :])
            pygame.draw.lines(self.win, FINISH_COLOR, False,
                              (finish[0:2],finish[2:4]), 4)

        def draw_goal_all():
            for i in range(self.track.goals.shape[0]):
                goal = tuple(self.track.goals[i, :])
                pygame.draw.lines(self.win, NEXT_GOAL_COLOR, False,
                                  (goal[0:2],goal[2:4]),4)

        def draw_car():
            if self.car.action_state == 0:
                img = self.CAR_IMG_OFF
            else:
                img = self.CAR_IMG
            rotated_image = pygame.transform.rotate(img, -self.car.angle)
            new_rect = rotated_image.get_rect(center=img.get_rect(center=(self.car.position.x, self.car.position.y)).center)
            self.win.blit(rotated_image, new_rect.topleft)

        def draw_spectators():
            if not self.car.visible and self.spectator is not None:
                for row in self.spectator:
                    # frame, x,y,ang,vel, action
                    _, x, y, ang, _, action_state = row
                    if action_state == 0:
                        image = self.CAR_IMG_OFF
                    else:
                        image = self.CAR_IMG
                    rotated_image = pygame.transform.rotate(image, -ang)
                    new_rect = rotated_image.get_rect(center=image.get_rect(center=(x, y)).center)
                    self.win.blit(rotated_image, new_rect.topleft)

        def draw_echo_vector():
            n =self.car.N_SENSORS
            echo_vectors_short = self.car.echo_vectors
            if len(self.car.echo_collision_points) == n:
                echo_vectors_short = self.car.echo_vectors
                for i in range(n):
                    echo_vectors_short[i,[2,3]] = self.car.echo_collision_points[i]                    
            for vector in echo_vectors_short:
                pygame.draw.line(self.win, ECHO_VECTOR_COLOR, vector[0:2], vector[2:4], 2)

        def draw_echo_collision_points():
            for point in self.car.echo_collision_points:
                pygame.draw.circle(self.win, ECHO_COLLISION_COLOR, (int(point[0]), int(point[1])), 2)

        def draw_text(surface, text=None, size=12, x=0, y=0, 
                      font_name=pygame.font.match_font('consolas'), 
                      position='topleft'):
            font = pygame.font.Font(font_name, size)
            text_surface = font.render(text, True, FONT_COLOR)
            text_back = pygame.Surface(text_surface.get_size())
            text_rect = text_surface.get_rect()
            if position == 'topleft':
                text_rect.topleft = (x, y)
            text_back.fill(FONT_BACKGROUND)
            surface.blit(text_back, text_rect)
            surface.blit(text_surface, text_rect)
            
        def get_gui_value(value: str):
            if value == 'Total_reward':
                return str(round(self.car.reward_total, 2))
            elif value == 'Level':
                return str(self.car.level)
            elif value == 'Distances':
                distances = [round(dist, 2) for dist in self.car.echo_collision_distances_interp[:]]
                return str(list(distances))
            elif value == 'Velocity':
                return str(round(self.car.vel_interp, 2))
            elif value == 'Position':
                return str(self.car.position)
            elif value == 'Frames_remaining':
                return str(self.max_steps-self.car.framecount_total)
            elif value == 'Total_frames':
                return str(self.car.framecount_total)
            elif value == 'Angle':
                return str(round(self.car.angle, 2))
            elif value == 'Acceleration':
                return str(round(self.car.action[1], 2))
            elif value == 'Steering':
                return str(round(self.car.action[0], 2))
            elif value == 'Lap':
                return str(self.car.n_lap)
            else:
                return 'value not found'


        # ─── INIT RENDERER ───────────────────────────────────────────────
        if self.win is None:
            init_renderer(self)

        # ─── RECURING RENDERING ──────────────────────────────────────────
        if self.gui_draw_background:
            self.win.blit(self.BG_IMG, (0, 0))
        else:
            self.win.fill(GRAY)
        self.win.blit(self.TRACK_IMG, (0, 0))
        
        if self.gui_draw_goal_all:
            draw_goal_all()
        if self.gui_draw_goal_next:
            draw_goal_next()
        if self.env_visible:
            draw_level()
        if self.gui_draw_echo_points and self.car.visible and self.car.action_state:
            draw_echo_collision_points()
        if self.gui_draw_echo_vectors and self.car.visible and self.car.action_state:
            draw_echo_vector()

        draw_finish_line()
        if self.car.visible:
            draw_car()
        draw_spectators()

        # ─── INTERFACE ───────────────────────────────────────────────────
        if self.gui:
            POS = 'topleft'
            if self.gui_state:
                for i, k in enumerate(['Total_reward', 'Level', 'Lap', 'Total_frames']):
                    gui_x_list = np.linspace(FONT_SIZE, MAIN_WINDOW_SIZE[0]-FONT_SIZE, 8)
                    draw_text(self.win, text=k, size=FONT_SIZE//2, x=gui_x_list[(i+2)], y=FONT_SIZE, position=POS)
                    draw_text(self.win, text=get_gui_value(k), size=FONT_SIZE, x=gui_x_list[(i+2)], y=FONT_SIZE+FONT_SIZE//2, position=POS)
            if self.gui_observations:
                for i, k in enumerate(['Velocity', 'Distances']):
                    gui_x_list = np.linspace(FONT_SIZE, MAIN_WINDOW_SIZE[0]-FONT_SIZE, 8)
                    draw_text(self.win, text=k, size=FONT_SIZE//2, x=gui_x_list[(i+2)], y=2*(FONT_SIZE+FONT_SIZE//2), position=POS)
                    draw_text(self.win, text=get_gui_value(k), size=FONT_SIZE, x=gui_x_list[(i+2)], y=2*(FONT_SIZE+FONT_SIZE//2)+FONT_SIZE//2, position=POS)
            if self.gui_input:
                for i, k in enumerate(['Steering', 'Acceleration']):
                    gui_x_list = np.linspace(FONT_SIZE, MAIN_WINDOW_SIZE[0]-FONT_SIZE, 8)
                    draw_text(self.win, text=k, size=FONT_SIZE//2, x=gui_x_list[(i+2)], y=3*(FONT_SIZE+FONT_SIZE//2)+FONT_SIZE//2, position=POS)
                    draw_text(self.win, text=get_gui_value(k), size=FONT_SIZE, x=gui_x_list[(i+2)], y=3*(FONT_SIZE+FONT_SIZE//2)+FONT_SIZE, position=POS)

        # ─── RENDER GAME ─────────────────────────────────────────────────
        pygame.display.update()

        # ─── EXPORT GAME FRAMES ──────────────────────────────────────────
        if self.export_frames:
            pygame.pixelcopy.surface_to_array(
                self.image3d, self.display_surface)
            self.image3dT = np.transpose(self.image3d, axes=[1, 0, 2])
            im = Image.fromarray(self.image3dT)  # monochromatic image
            imrgb = im.convert('RGB')  # color image

            filename = ''.join([
                self.export_string,
                self.session_id,
                '-frame-',
                str(self.car.framecount_total).zfill(5),
                '.jpg'])
            filenamepath = os.path.join('exported_frames', filename)
            imrgb.save(filenamepath)

    def get_car_state(self):
        return np.array([
            self.car.position,
            self.car.angle,
            self.car.net_velocity.y,
            self.car.action_state
        ])

    def update_car_state(self, car_state):
        self.car.update_state(car_state)

    def update_interface_vars(self, action_next):
        self.action_next = action_next
