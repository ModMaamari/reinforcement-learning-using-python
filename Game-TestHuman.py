from random import randint
from collections import deque
from time import sleep
import pygame

pygame.init()

class Field:
    def __init__(self, height=10, width=5):
        self.width = width
        self.height = height
        self.body = [[0] * width for _ in range(height)]

    def update_field(self, walls, player):
        self.body = [[0] * self.width for _ in range(self.height)]

        for wall in walls:
            if not wall.out_of_range:
                for i in range(wall.y, min(wall.y + wall.height, self.height)):
                    self.body[i][:] = wall.body[i - wall.y][:]

        for i in range(player.y, min(player.y + player.height, self.height)):
            for j in range(player.x, min(player.x + player.width, self.width)):
                self.body[i][j] = player.body[i - player.y][j - player.x]


class Wall:
    def __init__(self, height=5, width=100, hole_width=20, y=0, speed=1):
        self.height = height
        self.width = width
        self.hole_width = hole_width
        self.y = y
        self.speed = speed
        self.body_unit = 1
        self.body = [[self.body_unit] * width for _ in range(height)]
        self.out_of_range = False
        self.create_hole()

    def create_hole(self):
        hole_pos = randint(0, self.width - self.hole_width)
        for i in range(self.hole_width):
            self.body[self.height // 2][hole_pos + i] = 0

    def move(self):
        self.y += self.speed
        self.out_of_range = self.y + self.height > field.height


class Player:
    def __init__(self, height=5, max_width=10, width=2, x=0, y=0, speed=2):
        self.height = height
        self.max_width = max_width
        self.width = width
        self.x = x
        self.y = y
        self.speed = speed
        self.body_unit = 2
        self.body = [[self.body_unit] * width for _ in range(height)]

    def move(self, direction=0):
        if direction == 1 and self.x > 0:
            self.x -= self.speed
        elif direction == 2 and self.x + self.width < field.width:
            self.x += self.speed


class Environment:
    def __init__(self):
        self.BLACK = (25, 25, 25)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 80, 80)
        self.field = self.walls = self.player = None
        self.current_state = self.reset()

    def reset(self):
        self.field = Field()
        self.walls = deque([Wall()])
        self.player = Player(x=field.width // 2 - 1, y=field.height - 5)
        return self.field.body

    def step(self, action):
        reward = 0

        if action == 1 or action == 2:
            self.player.move(action)

        for wall in self.walls:
            wall.move()

        self.field.update_field(self.walls, self.player)

        if self.walls[-1].y == self.player.y + self.player.height:
            reward += 1

        return self.field.body, reward

    def render(self, window):
        window.fill(self.WHITE)

        for r in range(field.height):
            for c in range(field.width):
                color = self.WHITE if self.field.body[r][c] == 0 else self.BLACK
                pygame.draw.rect(window, color, (c * 40, r * 30, 40, 30))

        pygame.display.update()


env = Environment()
field = env.field

WINDOW_WIDTH = field.width * 40
WINDOW_HEIGHT = field.height * 30
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

clock = pygame.time.Clock()
game_over = False

while not game_over:
    clock.tick(27)
    env.render(WINDOW)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                env.step(1)
            elif event.key == pygame.K_RIGHT:
                env.step(2)

pygame.quit()
