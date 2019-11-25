#####################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.11.22
#   Intro: 五子棋人机游戏
#####################################

# 导入相关模块
import numpy as np
import pygame
import time
from gobang_ai import *


class Gobang(object):
    def __init__(self, board_size):
        '''
            初始化函数
            参数：
                board_size: 棋盘大小
            返回值：
                None
        '''
        super().__init__()
        self.board_size = board_size

    
    def initialize_game(self):
        '''
            初始化游戏
            参数：
                None
            返回值：
                None
        '''
        # 初始化棋盘
        self.board = np.zeros((self.board_size, self.board_size))


    def draw_board(self):
        '''
            绘制棋盘
            参数：
                None
            返回值：
                None
        '''
        # 绘制方框
        for i in range(self.board_size+1):
            pygame.draw.line(self.screen, self.LINE_COLOR, \
                            (self.top+i*self.CELL_SIZE, self.left), (self.top+i*self.CELL_SIZE, self.right))
            pygame.draw.line(self.screen, self.LINE_COLOR, \
                            (self.top, self.left+i*self.CELL_SIZE), (self.bottom, self.left+i*self.CELL_SIZE))
        
        # 绘制棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    # 玩家的棋子
                    position = (self.left+j*self.CELL_SIZE+self.CELL_SIZE//2, \
                                self.top+i*self.CELL_SIZE+self.CELL_SIZE//2)
                    pygame.draw.circle(self.screen, self.PLAYER_COLOR, position, int(self.CELL_SIZE*0.4))
                elif self.board[i, j] == -1:
                    # AI的棋子
                    position = (self.left+j*self.CELL_SIZE+self.CELL_SIZE//2, \
                                self.top+i*self.CELL_SIZE+self.CELL_SIZE//2)
                    pygame.draw.circle(self.screen, self.AI_COLOR, position, int(self.CELL_SIZE*0.4))


    def add_init_coins(self, turn):
        '''
            添加开局便存在的棋子
            参数：
                turn: 执棋方
            返回值：
                None:
        '''
        mid = self.board_size // 2
        self.board[mid, mid] = turn
        self.board[mid, mid+1] = turn
        self.board[mid, mid-1] = -turn
        self.board[mid+1, mid] = -turn


    def play(self):
        '''
            开始游戏
            参数：
                None
            返回值:
                None
        '''
        # 参数设置
        self.SCREEN_SIZE = 820
        self.BOARD_LEN = 780
        self.CELL_SIZE = self.BOARD_LEN // self.board_size
        self.BACKGROUND_COLOR = pygame.Color('peru')
        self.LINE_COLOR = pygame.Color('black')
        self.MAX_DEPTH = 3


        # 确定棋盘四条边的位置
        self.left = (self.SCREEN_SIZE - self.BOARD_LEN) // 2
        self.right = self.left + self.CELL_SIZE * self.board_size
        self.top = (self.SCREEN_SIZE - self.BOARD_LEN) // 2
        self.bottom = self.top + self.CELL_SIZE * self.board_size

        # 初始化
        pygame.init()
        self.initialize_game()
        # 屏幕设置
        self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
        # 标题设置
        pygame.display.set_caption('Gobang Game')
        # 背景颜色设置
        self.screen.fill(self.BACKGROUND_COLOR)

        # 读取先手方
        turn = input('First(1) or second(-1)?')
        while turn not in ('1', '-1'):
            turn = input('Error Input! Input Again: ')
        turn = int(turn)

        # 先手执黑
        if turn == 1:
            self.PLAYER_COLOR = pygame.Color('black')
            self.AI_COLOR = pygame.Color('white')
        else:
            self.PLAYER_COLOR = pygame.Color('white')
            self.AI_COLOR = pygame.Color('black')

        self.add_init_coins(turn)

        # 初始化邻居统计
        valid_board = np.zeros_like(self.board)
        for i in range(0, self.board_size):
            for j in range(0, self.board_size):
                    if self.board[i, j] == 0 \
                        and np.sum(self.board[max(0, i-1):min(self.board_size, i+2), \
                            max(0, j-1):min(self.board_size, j+2)]!=0)>0:
                        valid_board[i, j] += 1

        game_over = False
        print("#---------------------- Game Start ----------------------#")
        while not game_over:
            if turn == -1:
                # AI执棋
                cur_time = time.time()
                if np.sum(self.board!=0) == 0:
                    # 棋盘为空时
                    row, col = self.board_size // 2, self.board_size // 2
                else:
                    row, col, _ = alpha_beta(self.board, valid_board, turn, -np.inf, np.inf, self.MAX_DEPTH)
                self.board[row, col] = turn

                # 更新邻居统计
                for i in range(max(0, row-1), min(self.board_size, row+2)):
                    for j in range(max(0, col-1), min(self.board_size, col+2)):
                        if self.board[i, j] == 0:
                            valid_board[i, j] += 1
                valid_board[row, col] = 0

                turn = -turn
                # 输出记录
                score = evaluate(self.board, turn)
                print('AI turn: {}, score: {}, time cost: {}'.format((row, col), \
                    score, time.time()-cur_time))
            else:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # 关闭窗口
                        pygame.quit()
                        print("#----------------------- Game End -----------------------#")
                        return
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # 鼠标按键
                        # 计算落子位置
                        row = (event.pos[1] - self.top) // self.CELL_SIZE
                        col = (event.pos[0] - self.left) // self.CELL_SIZE
                        # 添加棋子
                        if (0<=row<self.board_size) and (0<=col<self.board_size) \
                            and (self.board[row, col]==0):
                            self.board[row, col] = turn
                            # 转换角色
                            turn = -turn
                            # 输出记录
                            score = evaluate(self.board, turn)
                            print('Player turn: {}, score: {}'.format((row, col), score))
                            
                            # 更新邻居统计
                            for i in range(max(0, row-1), min(self.board_size, row+2)):
                                for j in range(max(0, col-1), min(self.board_size, col+2)):
                                    if self.board[i, j] == 0:
                                        valid_board[i, j] += 1
                            valid_board[row, col] = 0

            # 绘制棋盘
            self.draw_board()
            # 刷新
            pygame.display.update()

            # 检查是否结束
            game_over = self.check_if_game_over()
            if game_over:
                if turn == 1:
                    print('AI Win!')
                else:
                    print('Player Win!')
                print('#----------------------- Game End -----------------------#')
            else:
                # 检查是否平局
                game_over = np.sum(self.board==0) == 0
                if game_over:
                    print('Draw!')
                    print('#----------------------- Game End -----------------------#')

        while True:
            # 等待关闭窗口
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return



    def check_if_game_over(self):
        '''
            检查游戏是否已经结束
            参数：
                None
            返回值：
                布尔值, 表示游戏是否结束
        '''
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0:
                    count = 1
                    step = 1

                    # 水平方向
                    while count < 5 and j < self.board_size-4 \
                        and self.board[i, j+step] == self.board[i, j]:
                        count += 1
                        step += 1
                    if count == 5:
                        return True
                    else:
                        count = 1
                        step = 1

                    #竖直方向
                    while count < 5 and i < self.board_size-4 \
                        and self.board[i+step, j] == self.board[i, j]:
                        count += 1
                        step += 1
                    if count == 5:
                        return True
                    else:
                        count = 1
                        step = 1

                    # 主对角线方向
                    while count < 5 and i < self.board_size-4 and j < self.board_size-4 \
                        and self.board[i+step, j+step] == self.board[i, j]:
                        count += 1
                        step += 1
                    if count == 5:
                        return True
                    else:
                        count = 1
                        step = 1

                    # 从对角线方向
                    while count < 5 and i < self.board_size-4 and j > 3 \
                        and self.board[i+step, j-step] == self.board[i, j]:
                        count += 1
                        step += 1
                    if count == 5:
                        return True
                    else:
                        count = 1
                        step = 1

        return False



if __name__ == "__main__":
    game = Gobang(11)
    game.play()