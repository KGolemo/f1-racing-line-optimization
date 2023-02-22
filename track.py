import numpy as np
import cv2
import matplotlib.pyplot as plt


TRACK_SIZE = 2463, 1244
MAIN_WINDOW_SIZE = int(TRACK_SIZE[0]/1.3), int(TRACK_SIZE[1]/1.3)


class Track:
    def __init__(self, game):
        self.game = game

        line1, line2, goals = self.get_checkpoints()
        self.L1_line1_array_source = line1
        self.L1_line2_array_source = line2
        self.L1_goals_array_source = goals
        self.load_level()

    @staticmethod
    def get_checkpoints():
        img = cv2.imread('imgs/Monza_track_extra_wide_contour.png')
        img = cv2.resize(img, MAIN_WINDOW_SIZE, fx=1/1.3, fy=1/1.3)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bin = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        line1 = np.zeros((len(contours[0])+1, 2))
        line2 = np.zeros((len(contours[1])+1, 2))

        for i, x in enumerate(contours[0]):
            line1[i] = x[0]
        line1[len(contours[0])] = line1[0]

        for i, x in enumerate(contours[1]):
            line2[i] = x[0]
        line2[len(contours[1])] = line2[0]

        contours_2, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        line1_2 = np.zeros((len(contours_2[0]), 2))
        line2_2 = np.zeros((len(contours_2[1]), 2))

        for i, x in enumerate(contours_2[0]):
            line1_2[i] = x[0]

        for i, x in enumerate(contours_2[1]):
            line2_2[i] = x[0]

        num_of_lines = 121
        a_index_vect = np.linspace(0, len(contours_2[0])-1, num_of_lines).astype(int)
        b_index_vect = np.linspace(0, len(contours_2[1])-1, num_of_lines).astype(int)

        fig, ax = plt.subplots(figsize=(18, 10))
        plt.gca().invert_yaxis()
        # ax.imshow(bin, cmap="gray")

        temp_goals = np.zeros((num_of_lines, 4))
        for i in range(num_of_lines):
            a_x = line1_2[a_index_vect[i]][0]
            a_y = line1_2[a_index_vect[i]][1]
            b_x = line2_2[b_index_vect[num_of_lines-1-i]][0]
            b_y = line2_2[b_index_vect[num_of_lines-1-i]][1]

            temp_goals[i] = [a_x, a_y, b_x, b_y]

        temp_goals = temp_goals[:-1]
        index = list(range(61, 120)) + list(range(0, 61))
        index = index[::-1]
        temp_goals = temp_goals[index]

        temp_goals = temp_goals[:-1]
        goals = np.zeros((num_of_lines-1, 4))
        goals[0] = [1501, 882, 1501, 857]
        goals[1:] = temp_goals[:]
        
        return line1, line2, goals

    
    def load_level(self):
        line1, line2, goals = np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,4))
        line1 = self.L1_line1_array_source.copy()
        line2 = self.L1_line2_array_source.copy()
        goals = self.L1_goals_array_source.copy()

        self.set_level_vectors(line1, line2, goals)
        self.n_goals = goals.shape[0]
        self.generate_collision_vectors(line1, line2)

    def move_env(self,d_x,d_y):
        # move the environment in fixed camera mode
        self.line1[:,0] = self.line1[:,0] - d_x
        self.line1[:,1] = self.line1[:,1] - d_y
        self.line2[:,0] = self.line2[:,0] - d_x
        self.line2[:,1] = self.line2[:,1] - d_y
        self.line1_list = self.line1.tolist()
        self.line2_list = self.line2.tolist()
        self.goals[:,[0,2]] = self.goals[:,[0,2]] - d_x
        self.goals[:,[1,3]] = self.goals[:,[1,3]] - d_y
        self.level_collision_vectors[:,[0,2]] = self.level_collision_vectors[:,[0,2]] - d_x
        self.level_collision_vectors[:,[1,3]] = self.level_collision_vectors[:,[1,3]] - d_y
    
    def set_level_vectors(self, line1, line2, goals, level_collision_vectors=None):
        self.line1 = line1
        self.line2 = line2
        self.goals = goals
        # list for pygame draw
        self.line1_list = line1.tolist()
        self.line2_list = line2.tolist()
        if level_collision_vectors:
            self.level_collision_vectors = level_collision_vectors

    def generate_collision_vectors(self,line1,line2):
        # for collision calculation, is numpy array
        # only call once to generate single line structe
        n1, n2 = line1.shape[0], line2.shape[0]
        line_combined = np.zeros((n1 + n2 - 2, 4))        
        line_combined[:n1-1,[0,1]] = line1[:n1-1,[0,1]]
        line_combined[:n1-1,[2,3]] = line1[1:n1,[0,1]]
        line_combined[n1-1:n1+n2-2,[0,1]] = line2[:n2-1,[0,1]]
        line_combined[n1-1:n1+n2-2,[2,3]] = line2[1:n2,[0,1]]
        self.level_collision_vectors = line_combined

    def get_goal_line(self, level):
        return self.goals[level, :]