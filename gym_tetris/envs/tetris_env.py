import os
import random

import gym
import numpy as np
import pygame

from gym_tetris.board import Board
from gym_tetris.game import Game
from gym_tetris.view import View
from gym_tetris.ai.DQN import DQN
from gym_tetris.envs.tetris_enemy_env import TetrisEnemyEnv

from pathlib import Path
import sys
import tensorflow as tf

WIN_WIDTH = 1000
WIN_HEIGHT = 526
WEIGHT_PATH_HOLD_ENEMY = os.path.join(os.path.dirname(__file__), 'MODEL_B_50000_03')

class TetrisEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }
    # 敵のモデルを読み込む
    
    # ENEMY_MODEL: DQN = DQN(
    #     gamma=1, epsilon=0, epsilon_min=0, epsilon_decay=0, hold_mode=1)
    # if Path(WEIGHT_PATH_HOLD_ENEMY).is_dir():
    #     ENEMY_MODEL.model= tf.keras.models.load_model(WEIGHT_PATH_HOLD_ENEMY)
    # else:
    #     sys.exit()

    def __init__(self, action_mode=0, hold_mode=0):
        # self.enemy_env = TetrisEnemyEnv(action_mode=1, hold_mode=1)
        # self.obs = self.enemy_env.reset()
        self.view = None
        self.game = None
        self.action_mode = action_mode
        self.hold_mode = hold_mode
        if action_mode == 0:
            # Nothing, Left, Right, Rotate left, Rotate right, Drop, Full Drop, Hold
            self.action_space = gym.spaces.Discrete(8)
        elif action_mode == 1:
            self.action_space = gym.spaces.Tuple((
                gym.spaces.Discrete(10),  # X
                gym.spaces.Discrete(4),  # Rotation
            ))

    def step(self, action):
        """Performs one step/frame in the game and returns the observation, reward and if the game is over."""
        if self.action_mode == 0:
            if action == 1:  # Left
                self.game.board.move_piece(-1)
            elif action == 2:  # Right
                self.game.board.move_piece(1)
            elif action == 3:  # Rotate left
                self.game.board.rotate_piece(-1)
            elif action == 4:  # Rotate right
                self.game.board.rotate_piece(1)
            elif action == 5:  # Drop
                self.game.board.drop_piece()
                self.game.drop_time = self.game.get_drop_speed()
            elif action == 6:  # Full drop
                self.game.board.drop_piece_fully()
            elif action == 7:  # Hold
                self.game.board.hold_piece()
        elif self.action_mode == 1:
            x, rotation, change = action
            if change == 1:  # choose to change function
                self.game.board.hold_piece()
            self.game.board.move_and_drop(x, rotation)

        player1_attack = []
        player2_attack = []

        # action, state = TetrisEnv.ENEMY_MODEL.choose_action(self.obs)
        # self.obs, _reward, _done, info_enemy = self.enemy_env.step(action)
        
        player1_attack = self.game.send_attack()
        # player2_attack = self.enemy_env.game.send_attack()

        # self.game.attacked.extend(player2_attack)
        # self.enemy_env.game.attacked.extend(player1_attack)


        rows = self.game.tick()
        rows_count = len(rows)
        rows_count_enemy = 0
        # rows_count_enemy = info_enemy['clear_line']
        done = self.game.board.is_game_over()

        reward = 1

        if rows_count == 1:
            reward += 40
        elif rows_count == 2:
            reward += 100
        elif rows_count == 3:
            reward += 300
        elif rows_count == 4:
            reward += 1200

        REN_BONUS_LIST = (0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5)
        if self.game.combo >= len(REN_BONUS_LIST)-1:
            reward += REN_BONUS_LIST[len(REN_BONUS_LIST)-1] * 300
        elif self.game.combo != 0:
            reward += REN_BONUS_LIST[self.game.combo-1] * 10
        

        winner = None

        if done:
            reward -= 5
            winner = "プレイヤー2"
        # if _done:
        #     reward += 2000
        #     winner = "プレイヤー1"

        # if done or _done:
        #     done = True

        return np.array(self.game.board.get_possible_states(self.game.combo)), reward, done, {"winner":winner,"clear_line":rows_count,"clear_line_enemy":rows_count_enemy,"attack1":player1_attack,"attack2":player2_attack}

    def reset(self):
        # self.obs = self.enemy_env.reset()
        """Starts a new game."""
        if self.hold_mode == 0:
            self.game = Game(Board(10, 20))
        else:
            self.game = Game(Board(10, 20, 1))
        return np.array(self.game.board.get_possible_states(self.game.combo))

    def close(self):
        """Closes the window."""
        if self.view is not None:
            self.view = None
            pygame.quit()

    def render(self, mode='human', close=False, width=WIN_WIDTH, height=WIN_HEIGHT):
        """Renders the game."""
        if self.view is None:
            pygame.init()
            win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
            font = pygame.font.Font(os.path.join(
                os.path.dirname(__file__), '..', 'assets', 'font.ttf'), 20)
            pygame.display.set_caption("Tetris")
            self.view = View(win, font)

        # self.view.draw(self.game, self.enemy_env.game)
        self.view.draw(self.game)

    def seed(self, seed=None):
        """Set the random seed for the game."""
        random.seed(seed)
        return [seed]
