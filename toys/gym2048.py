import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

import itertools
import logging
from six import StringIO
import sys

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

black = (0, 0, 0)
grey = (128, 128, 128)
white = (255, 255, 255)
tile_colour_map = {
    2: (255, 0, 0),
    4: (224, 32, 0),
    8: (192, 64, 0),
    16: (160, 96, 0),
    32: (128, 128, 0),
    64: (96, 160, 0),
    128: (64, 192, 0),
    256: (32, 224, 0),
    512: (0, 255, 0),
    1024: (0, 224, 32),
    2048: (0, 192, 64),
    4096: (0, 160, 96),
}

class CannotMove(Exception):
    "Informs the env the player cannot move in this direction."

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['ansi', 'human', 'rgb_array']}
    grid_size = 70
    size = 4
    squares = size ** 2
    max_tile = 2 ** squares
    illegal_move_reward = 0
    
    def __init__(self):
        # Definitions for game. Board must be square.
        self.w = self.h = self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        # layers = self.squares
        # self.observation_space = spaces.Box(0, 1, (self.w, self.h, layers), dtype=np.int)
        self.observation_space = spaces.Box(0, self.max_tile, (self.w, self.h), dtype=np.int)

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        done = None
        info = {
            'illegal_move': False,
        }
        try:
            reward = float(self.move(action))
            self.score += reward
            assert reward <= 2**(self.w*self.h)
            self.add_tile()
            done = self.check_done()
        except CannotMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            done = True
            reward = self.illegal_move_reward

        #print("Am I done? {}".format(done))
        info['highest'] = self.highest()

        # Return observation (board state), reward, done and info dict
        return self.board.copy(), reward, done, info

    def reset(self):
        self.board = np.zeros((self.h, self.w), np.int)
        self.score = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return self.board

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            grid_size = self.grid_size

            # Render with Pillow
            pil_board = Image.new("RGB", (grid_size * 4, grid_size * 4))
            draw = ImageDraw.Draw(pil_board)
            draw.rectangle([0, 0, 4 * grid_size, 4 * grid_size], grey)
            fnt = ImageFont.truetype('arial.ttf', 30)

            for y in range(4):
              for x in range(4):
                 o = self.get(y, x)
                 if o:
                     draw.rectangle([x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size], tile_colour_map[min(4096, o)])
                     (text_x_size, text_y_size) = draw.textsize(str(o), font=fnt)
                     draw.text((x * grid_size + (grid_size - text_x_size) // 2, y * grid_size + (grid_size - text_y_size) // 2), str(o), font=fnt, fill=white)
                     assert text_x_size < grid_size
                     assert text_y_size < grid_size

            im_array = np.asarray(pil_board)#.swapaxes(0, 1)
            if not hasattr(self, 'im_plt'):
                fig, ax = plt.subplots()
                ax.axis('off')
                self.im_fig, self.im_ax = fig, ax
                self.im_plt = ax.imshow(im_array)
                self.im_fig.show()
            else:
                self.im_plt.set_data(im_array)
            self.im_ax.set_title(str(int(self.score)))
            self.im_fig.canvas.draw()
            self.im_fig.canvas.start_event_loop(0.1)
            return im_array

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.board)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.board[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.board[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.board == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.board)

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two
        # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if not changed: raise CannotMove
        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def check_done(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.highest() == self.max_tile:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except CannotMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.board

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.board = new_board
