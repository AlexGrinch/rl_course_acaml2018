import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display


class WindyGridWorld:

    def __init__(
            self,
            grid_size=(11, 14),
            stochasticity=0.1,
            visual=False):
        """
        Parameters
        ----------
        grid_size: tuple of the form (w, h)
            w: int, width of the grid
            h: int, height of the grid
        stochasticity: float from [0, 1]
            probability to take random action instead of
            the intended one
        visual: boolean
            False: state is the agent's position
            True: state is image of the grid
        """
        self.w, self.h = grid_size
        self.stochasticity = stochasticity
        self.visual = visual

        # x position of the wall, y position of the hole
        self.x_wall = self.w // 2
        self.y_hole = self.h - 4

        self.reset()

    def reset(self):
        """ reset the environment
        """
        self.field = np.zeros((self.w, self.h))
        self.field[self.x_wall, :] = 1
        self.field[self.x_wall, self.y_hole] = 0
        self.field[0, 0] = 2
        self.pos = (0, 0)
        state = self.get_state()
        return state

    def step(self, action):
        """ take a step in the environment
        Parameters
        ----------
        action: int from [0, 3], given action
        """

        if np.random.rand() < self.stochasticity:
            action = np.random.randint(4)

        self.field[self.pos] = 0
        self.pos = self.move(action)
        self.field[self.pos] = 2

        done = False
        reward = 0
        if self.pos == (self.w - 1, 0):
            # episode finished successfully
            done = True
            reward = 1
        next_state = self.get_state()
        return next_state, reward, done

    def next_states(self, state, action):
        """
        Parameters
        ----------
        state (s): environment state
        action (a): action taken in state

        Returns
        -------
        list of pairs [s', p(s'|s,a)]
        s': possible next state
        p(s'|s,a): transition probability
        """
        x, y = self.pos

        self.pos = state
        next_states = []
        for a in range(4):
            x_, y_ = self.move(a)
            prob = self.stochasticity / 4
            if a == action:
                prob += (1 - self.stochasticity)
            next_states.append([(x_, y_), prob])
        self.pos = (x, y)
        return next_states

    def clip_xy(self, x, y):
        """ clip coordinates if they go beyond the grid
        """
        x_ = np.clip(x, 0, self.w - 1)
        y_ = np.clip(y, 0, self.h - 1)
        return x_, y_

    def wind_shift(self, x, y):
        """ apply wind shift to areas where wind is blowing
        """
        if x == 1:
            return self.clip_xy(x, y + 1)
        elif x > 1 and x < self.x_wall:
            return self.clip_xy(x, y + 2)
        else:
            return x, y

    def move(self, action):
        """ find valid coordinates of the agent after executing action
        """
        x, y = self.pos
        x, y = self.wind_shift(x, y)
        if action == 0:
            x_, y_ = x + 1, y
        if action == 1:
            x_, y_ = x, y + 1
        if action == 2:
            x_, y_ = x - 1, y
        if action == 3:
            x_, y_ = x, y - 1
        # check if new position does not conflict with the wall
        if x_ == self.x_wall and y_ != self.y_hole:
            x_, y_ = x, y
        return self.clip_xy(x_, y_)

    def get_state(self):
        """ get state of the environment
        """
        if self.visual:
            state = np.rot90(self.field)[:, :, None]
        else:
            state = self.pos
        return state

    def draw_state(self):
        """ draw grid world
        """
        img = np.rot90(1-self.field)
        plt.imshow(img, cmap="gray")

    def play_with_policy(self, policy, max_iter=100, visualize=True):
        """ play with given policy
        Parameters
        ----------
        policy: function: state --> action
        max_iter: maximum number of time steps
        visualize: bool, if True visualize episode
        """
        self.reset()
        for i in range(max_iter):
            state = self.get_state()
            action = policy(state)
            next_state, reward, done = self.step(action)

            # plot grid world state
            if visualize:
                self.draw_state()
                display.clear_output(wait=True)
                display.display(plt.gcf())
                time.sleep(0.01)
            if done:
                break
        if visualize:
            display.clear_output(wait=True)
        return reward
