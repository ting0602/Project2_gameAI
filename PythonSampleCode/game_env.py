import gym
from gym import spaces
import numpy as np

class GameEnv(gym.Env):
    def __init__(self, subset_size_range=(12, 32)):
        super(GameEnv, self).__init__()
        
        self.subset_size_range = subset_size_range
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8), dtype=np.float32)
        
        # Initialize game board
        self.board = np.zeros((8, 8))
        
        # Randomly generate subset of cells used in the game
        self.used_cells = self._generate_used_cells()
        
        # Set initial player turn and game status
        self.player = 1
        self.done = False
        
    def step(self, action):
        # Make a move and update board
        self._make_move(action)
        
        # Check if game is over
        if self._is_game_over():
            self.done = True
        
        # Switch player turn
        self.player = 3 - self.player
        
        # Calculate reward
        reward = 0
        if self.done:
            if self.player == 2:
                reward = 1
            else:
                reward = -1
                
        return self.board, reward, self.done, {}
        
    def reset(self):
        # Clear game board
        self.board = np.zeros((8, 8))
        
        # Randomly generate subset of cells used in the game
        self.used_cells = self._generate_used_cells()
        
        # Set initial player turn and game status
        self.player = 1
        self.done = False
        
        return self.board
    
    def render(self, mode='console'):
        if mode == 'console':
            print(self.board)
        elif mode == 'human':
            pass
        
    def _make_move(self, action):
        # Find contiguous cells along a straight line
        for i in range(8):
            for j in range(8):
                if self.board[i, j] == self.player:
                    if action == 0 and j+2 < 8 and self.board[i, j+1] == self.player and self.board[i, j+2] == 0:
                        self.board[i, j] = 0
                        self.board[i, j+1] = 0
                        self.board[i, j+2] = self.player
                        return
                    elif action == 1 and i+2 < 8 and self.board[i+1, j] == self.player and self.board[i+2, j] == 0:
                        self.board[i, j] = 0
                        self.board[i+1, j] = 0
                        self.board[i+2, j] = self.player
                        return
                    elif action == 2 and i+2 < 8 and j+2 < 8 and self.board[i+1, j+1] == self.player and self.board[i+2, j+2] == 0:
                        self.board[i, j] = 0
                        self.board[i+1, j+1] = 0
                        self.board[i+2, j+2] = self.player
                        return
                        
    def _is_game_over(self):
        for i in range(8):
            for j in range(8):
                if self.board[i, j] == self.player