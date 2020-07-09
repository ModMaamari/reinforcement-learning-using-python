from random import randint, choice
from collections import deque
from time import sleep
import pygame, time
import numpy as np

pygame.init()

######################################################################################
class Field:
    def __init__(self, height=10, width=5):
        self.width      = width
        self.height     = height
        self.body = np.zeros(shape=(self.height, self.width))
    def update_field(self,walls, player):
        try:
            # Clear the field:
            self.body = np.zeros(shape=(self.height, self.width))
            # Put the walls on the field:
            for wall in walls:
                if not wall.out_of_range :
                    self.body[wall.y:min(wall.y+wall.height,self.height),:] = wall.body

            # Put the player on the field:
            self.body[player.y:player.y+player.height,
                      player.x:player.x+player.width] += player.body 
        except :
            pass
######################################################################################
class Wall:        
    def __init__(self, height = 5, width=100,  hole_width = 20,
                 y = 0, speed = 1, field = None):
        self.height       = height
        self.width        = width
        self.hole_width   = hole_width
        self.y            = y
        self.speed        = speed
        self.field        = field
        self.body_unit     = 1
        self.body         = np.ones(shape = (self.height, self.width))*self.body_unit
        self.out_of_range = False
        self.create_hole()
    def create_hole(self):
        hole = np.zeros(shape = (self.height, self.hole_width))
        hole_pos = randint(0,self.width-self.hole_width)
        self.body[ : , hole_pos:hole_pos+self.hole_width] = 0
    def move(self):
        self.y += self.speed
        self.out_of_range = True if ((self.y + self.height) > self.field.height) else False
######################################################################################
class Player:
    def __init__(self, height = 5, max_width = 10 , width=2,
                 x = 0, y = 0, speed = 2):
        self.height        = height
        self.max_width     = max_width
        self.width         = width
        self.x             = x
        self.y             = y
        self.speed         = speed
        self.body_unit     = 2
        self.body          = np.ones(shape = (self.height, self.width))*self.body_unit
        self.stamina       = 20
        self.max_stamina   = 20
    def move(self, field, direction = 0 ):
        '''
        Moves the player :
         - No change          = 0
         - left, if direction  = 1
         - right, if direction = 2
        '''
        val2dir   = {0:0 , 1:-1 , 2:1}
        direction = val2dir[direction]
        next_x = (self.x + self.speed*direction)
        if not (next_x + self.width > field.width or next_x < 0):
            self.x += self.speed*direction
            self.stamina -= 1 
    def change_width(self, action = 0):
        '''
        Change the player's width:
         - No change          = 0
         - narrow by one unit = 3
         - widen by one unit  = 4
        '''
        val2act   = {0:0 , 3:-1 , 4:1}
        action = val2act[action]
        new_width = self.width+action
        player_end = self.x + new_width
        if new_width <= self.max_width and new_width > 0 and player_end <= self.max_width:
            self.width = new_width
            self.body  = np.ones(shape = (self.height, self.width))*self.body_unit
######################################################################################
class Environment:
    P_HEIGHT      = 2  # Height of the player
    F_HEIGHT      = 20 # Height of the field
    W_HEIGHT      = 2  # Height of the walls
    WIDTH         = 10 # Width of the field and the walls
    MIN_H_WIDTH   = 2  # Minimum width of the holes
    MAX_H_WIDTH   = 6  # Maximum width of the holes
    MIN_P_WIDTH   = 2  # Minimum Width of the player
    MAX_P_WIDTH   = 6  # Maximum Width of the player
    HEIGHT_MUL    = 30 # Height Multiplier (used to draw np.array as blocks in pygame )
    WIDTH_MUL     = 40 # Width Multiplier (used to draw np.array as blocks in pygame )
    WINDOW_HEIGHT = (F_HEIGHT+1) * HEIGHT_MUL # Height of the pygame window
    WINDOW_WIDTH  = (WIDTH) * WIDTH_MUL       # Widh of the pygame window
    
    ENVIRONMENT_SHAPE = (F_HEIGHT,WIDTH,1)
    ACTION_SPACE      = [0,1,2,3,4]
    ACTION_SPACE_SIZE = len(ACTION_SPACE)
    PUNISHMENT        = -100  # Punishment increment
    REWARD            = 10    # Reward increment
    score             = 0     # Initial Score
    
    MOVE_WALL_EVERY   = 4     # Every how many frames the wall moves.
    MOVE_PLAYER_EVERY = 1     # Every how many frames the player moves.
    frames_counter    = 0

    def __init__(self):
        # Colors:
        self.BLACK      = (25,25,25)
        self.WHITE      = (255,255,255)
        self.RED        = (255, 80, 80)
        self.BLUE       = (80, 80, 255)
        self.field = self.walls = self.player = None
        self.current_state = self.reset()
        self.val2color  = {0:self.WHITE, self.walls[0].body_unit:self.BLACK,
                           self.player.body_unit:self.BLACK, self.MAX_VAL:self.RED}
    def reset(self):
        self.score          = 0
        self.frames_counter = 0
        self.game_over      = False
        
        self.field = Field(height=self.F_HEIGHT, width=self.WIDTH )
        w1    = Wall( height = self.W_HEIGHT, width=self.WIDTH,
                      hole_width = randint(self.MIN_H_WIDTH,self.MAX_H_WIDTH),
                     field = self.field)
        self.walls = deque([w1])
        p_width = randint(self.MIN_P_WIDTH,self.MAX_P_WIDTH)
        self.player    = Player( height = self.P_HEIGHT, max_width = self.WIDTH,
                                width = p_width,
                                x = randint(0,self.field.width-p_width),
                                y = int(self.field.height*0.7), speed = 1)
        self.MAX_VAL = self.player.body_unit + w1.body_unit
        # Update the field :
        self.field.update_field(self.walls, self.player)
        
        observation = self.field.body/self.MAX_VAL
        return observation
    def print_text(self, WINDOW = None, text_cords = (0,0), center = False,
                   text = "", color = (0,0,0), size = 32):
        pygame.init()
        font = pygame.font.Font('freesansbold.ttf', size) 
        text_to_print = font.render(text, True, color) 
        textRect = text_to_print.get_rect()
        if center:
            textRect.center = text_cords
        else:
            textRect.x = text_cords[0]
            textRect.y = text_cords[1]
        WINDOW.blit(text_to_print, textRect)
        
    def step(self, action):
        global score_increased

        self.frames_counter += 1
        reward = 0

        # If the performed action is (move) then player.move method is called:
        if action in [1,2]:
            self.player.move(direction = action, field = self.field)
        # If the performed action is (change_width) then player.change_width method is called:
        if action in [3,4]:
            self.player.change_width(action = action)                
        
        # Move the wall one step (one step every MOVE_WALL_EVERY frames):
        if self.frames_counter % self.MOVE_WALL_EVERY == 0:
            # move the wall one step
            self.walls[-1].move()
            # reset the frames counter
            self.frames_counter = 0
        
        # Update the field :
        self.field.update_field(self.walls, self.player)

        # If the player passed a wall successfully increase the reward +1
        if ((self.walls[-1].y) == (self.player.y + self.player.height)) and not score_increased :
            reward += self.REWARD
            self.score  += self.REWARD
            
            # Increase player's stamina every time it passed a wall successfully  
            self.player.stamina = min(self.player.max_stamina, self.player.stamina+10)
            # score_increased : a flag to make sure that reward increases once per wall 
            score_increased = True
            
        
        #  Lose Conditions : 
        # C1 : The player hits a wall
        # C2 : Player's width was far thinner than hole's width
        # C3 : Player fully consumed its stamina (energy)
        lose_conds = [self.MAX_VAL in self.field.body,
                      ((self.player.y == self.walls[-1].y) and (self.player.width < (self.walls[-1].hole_width-1))),
                      self.player.stamina <=0]
        

        # If one lose condition or more happend, the game ends:
        if True in lose_conds:
            self.game_over = True
            reward = self.PUNISHMENT
            return self.field.body/self.MAX_VAL, reward, self.game_over

        # Check if a wall moved out of the scene:
        if self.walls[-1].out_of_range:
            # Create a new wall
            self.walls[-1] = Wall( height = self.W_HEIGHT, width = self.WIDTH,
                                   hole_width = randint(self.MIN_H_WIDTH,self.MAX_H_WIDTH),
                                   field = self.field)

            score_increased = False

        
        # Return New Observation , reward, game_over(bool)
        return self.field.body/self.MAX_VAL, reward, self.game_over
    
    def render(self, WINDOW = None, human=False):
        if human:
            ################ Check Actions #####################
            action = 0
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.game_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 1
                    if event.key == pygame.K_RIGHT:
                        action = 2

                    if event.key == pygame.K_UP:
                        action = 4
                    if event.key == pygame.K_DOWN:
                        action = 3
            ################## Step ############################            
            _,reward, self.game_over = self.step(action)
        ################ Draw Environment ###################
        WINDOW.fill(self.WHITE)
        self.field.update_field(self.walls, self.player)
        for r in range(self.field.body.shape[0]):
            for c in range(self.field.body.shape[1]):
                pygame.draw.rect(WINDOW,
                                 self.val2color[self.field.body[r][c]],
                                 (c*self.WIDTH_MUL, r*self.HEIGHT_MUL, self.WIDTH_MUL, self.HEIGHT_MUL))

        self.print_text(WINDOW = WINDOW, text_cords = (self.WINDOW_WIDTH // 2, int(self.WINDOW_HEIGHT*0.1)),
                       text = str(self.score), color = self.RED, center = True)
        self.print_text(WINDOW = WINDOW, text_cords = (0, int(self.WINDOW_HEIGHT*0.9)),
                       text = str(self.player.stamina), color = self.RED)
        
        pygame.display.update()
######################################################################################
# Make an environment object
env            = Environment()
# Change wall speed to 3 (one step every 3 frames)
env.MOVE_WALL_EVERY = 3

# Initialize some variables 
WINDOW          = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
clock           = pygame.time.Clock()
win             = False
winning_score   = 100

# Repeaat the game untill the player win (got a score of winning_score) or quits the game.
while not win:
  score_increased = False
  game_over       = False
  _ = env.reset()
  pygame.display.set_caption("Game")
  while not game_over:
      clock.tick(27)
      env.render(WINDOW = WINDOW, human=True)
      game_over = env.game_over
  #####################################################
  sleep(0.5)
  WINDOW.fill(env.WHITE)
  if env.score >= winning_score:
    win = True
    env.print_text(WINDOW = WINDOW, text_cords = (env.WINDOW_WIDTH // 2, env.WINDOW_HEIGHT// 2),
                       text = f"You Win - Score : {env.score}", color = env.RED, center = True)
  else:
    env.print_text(WINDOW = WINDOW, text_cords = (env.WINDOW_WIDTH // 2, env.WINDOW_HEIGHT// 2),
                       text = f"Game Over - Score : {env.score}", color = env.RED, center = True)
  pygame.display.update()
######################################################################################
