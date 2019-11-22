#########################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.11.22
#   Intro: 五子棋AI，采用Alpha-Beta剪枝搜索
#########################################

# 导入相关模块
import numpy as np


def alpha_beta(board):
    '''
        alpha-beta剪枝搜索算法
        参数：
            board: numpy.array棋盘，0为空格，-1为AI棋子，+1为玩家棋子
        返回值：
            (row, col): 下一步旗
            score: 分数
    '''
    rows, cols = np.where(board==0)
    pos = np.random.randint(len(rows))
    pos = 0
    return rows[pos], cols[pos], 1


def evaluate(board):
    '''
        评价函数，得出当前分数
        参数：
            board: numpy.array棋盘，0为空格，-1为AI棋子，+1为玩家棋子
        返回值：
            score: 当前分数
    '''
    score = 1

    return score