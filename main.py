# from cbs_mapf.planner import Planner
# planner = Planner(grid_size=1, robot_radius=2, static_obstacles=[[0, 0], [191, 107]])

# print(planner.plan(starts=[[33, 16]], goals=[(54, 56)], debug=True))

'''
Author: Haoran Peng
Email: gavinsweden@gmail.com
'''
import os
# import os
import time
import sys
from copy import deepcopy
import cv2
import numpy as np
import yaml

from planner import Planner
import pickle as pickle


class Simulator:

    def __init__(self):
        # Set up a white 1080p canvas
        self.canvas = np.ones((1080,1920,3), np.uint8)*255 
        # Draw the rectangluar obstacles on canvas
        self.draw_rect(np.array([np.array(v) for v in RECT_OBSTACLES.values()]))

        # Transform the vertices to be border-filled rectangles
        static_obstacles = self.vertices_to_obsts(RECT_OBSTACLES)

        # Call cbs-mapf to plan
        self.planner = Planner(GRID_SIZE, ROBOT_RADIUS, static_obstacles)
        self.path = self.planner.plan(START, GOAL, debug=False)
        
        # self.calculate_goal_times = self.planner.calculate_goal_times( CTNode, Agent, List[Agent])
        # print("cal goal time: ",self.calculate_goal_times )

        # Assign each agent a colour
        self.colours = self.assign_colour(len(self.path))

        # Put the path into dictionaries for easier access
        d = dict()
        for i, path in enumerate(self.path):
            self.draw_path(self.canvas, path, i)  # Draw the path on canvas
            d[i] = path
            # print("d[i]: ",d[i])
        self.path = d
        
        # print("self.path",d)
        # print("self.path 0 = ",d[0])
        # print("self.path 1 = ",d[1])
        # print("length of self.path", len(d[0]))
        # print("length of self.path", len(d[1]))
        

    '''
    Transform opposite vertices of rectangular obstacles into obstacles
    '''
    @staticmethod
    def vertices_to_obsts(obsts):
        def drawRect(v0, v1):
            o = []
            base = abs(v0[0] - v1[0])
            side = abs(v0[1] - v1[1])
            for xx in range(0, base, 30):
                o.append((v0[0] + xx, v0[1]))
                o.append((v0[0] + xx, v0[1] + side - 1))
            o.append((v0[0] + base, v0[1]))
            o.append((v0[0] + base, v0[1] + side - 1))
            for yy in range(0, side, 30):
                o.append((v0[0], v0[1] + yy))
                o.append((v0[0] + base - 1, v0[1] + yy))
            o.append((v0[0], v0[1] + side))
            o.append((v0[0] + base - 1, v0[1] + side))
            return o
        static_obstacles = []
        for vs in obsts.values():
            static_obstacles.extend(drawRect(vs[0], vs[1]))
        return static_obstacles

    '''
    Randomly generate colours
    '''
    @staticmethod
    def assign_colour(num):
        def colour(x):
            x = hash(str(x+42))
            return ((x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF))
        colours = dict()
        for i in range(num):
            colours[i] = colour(i)
        return colours

    def draw_rect(self, pts_arr: np.ndarray) -> None:
        for pts in pts_arr:
            cv2.rectangle(self.canvas, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), thickness=3)

    def draw_path(self, frame, xys, i):
        for x, y in xys:
            cv2.circle(frame, (int(x), int(y)), 10, self.colours[i], -1)

    '''
    Press any key to start.
    Press 'q' to exit.
    '''
    def start(self):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', (1280, 720))
        wait = True
        try:
            i = 0
            while True:
                frame = deepcopy(self.canvas)
                for id_ in self.path:
                    x, y = tuple(self.path[id_][i])
                    cv2.circle(frame, (x, y), ROBOT_RADIUS-5, self.colours[id_], 5)
                cv2.imshow('frame', frame)
                if wait:
                    cv2.waitKey(0)
                    wait = False
                k = cv2.waitKey(100) & 0xFF 
                if k == ord('q'):
                    break
                i += 1
        except Exception:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def load_scenario(fd):
    with open(fd, 'r') as f:
        global GRID_SIZE, ROBOT_RADIUS, RECT_OBSTACLES, START, GOAL
        data = yaml.load(f, Loader=yaml.FullLoader)
        GRID_SIZE = data['GRID_SIZE']
        ROBOT_RADIUS = data['ROBOT_RADIUS']
        RECT_OBSTACLES = data['RECT_OBSTACLES']
        START = data['START']
        GOAL = data['GOAL']

'''
Use this function to show your START/GOAL configurations
'''
def show_pos(pos):
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', (1280, 720))
    frame = np.ones((1080,1920,3), np.uint8)*255
    for x, y in pos:
        cv2.circle(frame, (x, y), ROBOT_RADIUS-5, (0, 0, 0), 5)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # From command line, call:
    # python3 visualizer.py scenario1.yaml
    # load_scenario(sys.argv[1])
    load_scenario("scenario1.yaml")
    # show_pos(START)
    r = Simulator()
    r.start() 
    # call simulator first
    os.system('python simulator.py')
    
    # # Load goal times dictionaryc
    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
    #     print(loaded_dict)