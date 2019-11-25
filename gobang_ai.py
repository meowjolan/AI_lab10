#########################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.11.22
#   Intro: 五子棋AI，采用Alpha-Beta剪枝搜索
#########################################

# 导入相关模块
import numpy as np
import time

# 常量定义
# 评分棋型设计
score_dict = [
    # 五子
    (np.array([1, 1, 1, 1, 1]) / 5, 1e7),
    # 四子
    (np.array([1000, 1, 1, 1, 1, 10000]) / 4, 1e5),
    (np.array([1000, 1, 1, 1, 1, -1]) / 5, 5000),
    (np.array([-1, 1, 1, 1, 1, 1000]) / 5, 5000),
    (np.array([1, 1000, 1, 1, 1, 10000]) / 4, 5000),
    (np.array([10000, 1, 1, 1, 1000, 1]) / 4, 5000),
    (np.array([1, 1000, 1, 1, 1, -1]) / 5, 5000),
    (np.array([1, -1, 1, 1, 1000, 1]) / 5, 5000),
    # 三子
    (np.array([1000, 1, 1, 1, 10000]) / 3, 1000),
    (np.array([-1, 1, 1, 1, 10000]) / 4, 500),
    (np.array([10000, 1, 1, 1, -1]) / 4, 500),
    # 二子
    (np.array([1000, 1, 1, 10000]) / 2, 100),
]


def alpha_beta(board, valid_board, turn, alpha, beta, depth):
    '''
        alpha-beta剪枝搜索算法
        参数：
            board: numpy.array棋盘，0为空格，-1为AI棋子，+1为玩家棋子
            valid_board: 可行步，>0表示可行
            turn: 极大(1) 或 极小(-1)
            alpha, beta: 极大极小值
            depth: 限制深度
        返回值：
            (best_row, best_col): 已知最好的下一步旗
            _: 效益值
    '''
    # 检查是否到达终止条件
    if depth==0 or if_game_over(board, turn):
        return None, None, evaluate(board, turn)

    # 获取可行步列表
    childList = getNextSteps(board, turn, valid_board)

    best_row, best_col = 0, 0

    if turn == 1:
        # max
        for row, col in childList:
            # 递归调用
            board[row, col] = turn
            for i in range(max(0, row-1), min(board.shape[0], row+2)):
                for j in range(max(0, col-1), min(board.shape[1], col+2)):
                    if board[i, j] == 0:
                        valid_board[i, j] += 1
            _, _, next_beta = alpha_beta(board, valid_board, -turn, alpha, beta, depth-1)
            for i in range(max(0, row-1), min(board.shape[0], row+2)):
                for j in range(max(0, col-1), min(board.shape[1], col+2)):
                    if board[i, j] == 0:
                        valid_board[i, j] -= 1
            board[row, col] = 0
            # 更新alpha
            if alpha < next_beta:
                alpha = next_beta
                best_row, best_col = row, col
            # 剪枝判断
            if alpha >= beta:
                break
        return best_row, best_col, alpha
    else:
        # min
        for row, col in childList:
            # 递归调用
            board[row, col] = turn
            for i in range(max(0, row-1), min(board.shape[0], row+2)):
                for j in range(max(0, col-1), min(board.shape[1], col+2)):
                    if board[i, j] == 0:
                        valid_board[i, j] += 1
            _, _, next_alpha = alpha_beta(board, valid_board, -turn, alpha, beta, depth-1)
            for i in range(max(0, row-1), min(board.shape[0], row+2)):
                for j in range(max(0, col-1), min(board.shape[1], col+2)):
                    if board[i, j] == 0:
                        valid_board[i, j] -= 1
            board[row, col] = 0
            # 更新beta
            if beta > next_alpha:
                beta = next_alpha
                best_row, best_col = row, col
            # 剪枝判断
            if beta <= alpha:
                break
        return best_row, best_col, beta


def getNextSteps(board, turn, valid_board):
    '''
        获取可行的下一步列表
        参数：
            board: numpy.array棋盘，0为空格，-1为AI棋子，+1为玩家棋子
            turn: 极大(1) 或 极小(-1)
            valid_board: 邻居统计
        返回值：
            next_steps: 下一步列表
    '''
    # 经过排序后，末尾的空位价值一般不高，可以忽略
    LIMITED_AMOUNT = 8
    # 只选择四邻域内有棋子的空位
    next_steps = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if valid_board[i, j] > 0:
                next_steps.append((i, j))

    # 根据评分进行排序
    next_steps.sort(key=lambda x: getPointScore(turn*board, x[0], x[1]), reverse=True)

    return next_steps[:LIMITED_AMOUNT]


def getPointScore(board, row, col):
    '''
        计算下一步的分数
        参数：
            board: numpy.array棋盘，0为空格，-1为对方棋子，+1为本方棋子
            row, col: 下一步
        返回值：
            score: 下一步之后的分数
    '''
    board[row, col] = 1
    score = evaluate(board, 1)
    return score


def evaluate(board, turn):
    '''
        评价函数，得出当前分数
        参数：
            board: numpy.array棋盘，0为空格，-1为AI棋子，+1为玩家棋子
            turn: 执棋方
        返回值：
            score: 当前分数
    '''
    # 计算两方评分
    max_score = 0
    min_score = 0

    # max
    temp_board = np.pad(board, ((1,1),(1,1)), 'constant', constant_values=(-1,-1))
    # 翻转
    board_fp = np.fliplr(temp_board)
    # 各个方向的向量
    board_max_vec = [i for i in temp_board] + [i for i in temp_board.T] \
        + [temp_board.diagonal(i) for i in range(-temp_board.shape[0]+5, temp_board.shape[1]-4)] \
        + [board_fp.diagonal(i) for i in range(-board_fp.shape[0]+5, board_fp.shape[1]-4)]

    # min
    temp_board = -board
    temp_board = np.pad(temp_board, ((1,1),(1,1)), 'constant', constant_values=(-1,-1))
    # 翻转
    board_fp = np.fliplr(temp_board)
    # 各个方向的向量
    board_min_vec = [i for i in temp_board] + [i for i in temp_board.T] \
        + [temp_board.diagonal(i) for i in range(-temp_board.shape[0]+5, temp_board.shape[1]-4)] \
        + [board_fp.diagonal(i) for i in range(-board_fp.shape[0]+5, board_fp.shape[1]-4)]

    # 遍历各种棋型
    for model, model_score in score_dict:
        # 统计频数并更新分数
        max_score += conv2d(board_max_vec, model) * model_score
        min_score += conv2d(board_min_vec, model) * model_score

    # 执棋方有额外加分
    if turn == 1:
        max_score += max_score // 10
    else:
        min_score += min_score // 10

    # 最终得分
    score = max_score - min_score
    
    return score


def conv2d(a, b):
    '''
        特殊设计的二维卷积
        参数：  
            a: 一维数组的列表
            b: 一维数组
        返回值：
            count: 两个计算结果
    '''
    count = 0
    for aa in a:
        if len(aa) >= len(b):
            count += np.sum(np.convolve(aa, b, mode='valid')==1)
    return count


def if_game_over(board, turn):
    '''
        判断游戏是否已经结束
        参数：
            board: numpy.array棋盘，0为空格，-1为AI棋子，+1为玩家棋子
            turn: 执棋方
        返回值：
            布尔值
    '''
    # 检查棋盘是否已满
    if np.sum(board==0) == 0:
        return True

    # 翻转
    board_fp = np.fliplr(board)
    # 各个方向的向量
    board_vec = [i for i in board] + [i for i in board.T] \
        + [board.diagonal(i) for i in range(-board.shape[0]+5, board.shape[1]-4)] \
        + [board_fp.diagonal(i) for i in range(-board_fp.shape[0]+5, board_fp.shape[1]-4)]

    model = np.array([1, 1, 1, 1, 1]) * turn / 5

    # 遍历每个向量
    for vec in board_vec:
        if np.sum(np.convolve(vec, model, mode='valid')==1) > 0:
            return True

    return False