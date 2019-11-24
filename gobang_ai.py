#########################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.11.22
#   Intro: 五子棋AI，采用Alpha-Beta剪枝搜索
#########################################

# 导入相关模块
import numpy as np

# 常量定义
# 评分棋型设计
score_dict = [
    (np.array([1, 1, 1, 1, 1]) / 5, 1e6),
    (np.array([1000, 1, 1, 1, 1, 10000]) / 4, 10000),
    (np.array([1000, 1, 1, 1, 1, -1]) / 5, 5000),
    (np.array([-1, 1, 1, 1, 1, 1000]) / 5, 5000),
    (np.array([1000, 1, 1, 1, 10000]) / 3, 1000),
    (np.array([-1, 1, 1, 1, 10000]) / 4, 500),
    (np.array([10000, 1, 1, 1, -1]) / 4, 500),
    (np.array([1000, 1, 1, 10000]) / 2, 100),
]


def alpha_beta(board, turn, alpha, beta, depth):
    '''
        alpha-beta剪枝搜索算法
        参数：
            board: numpy.array棋盘，0为空格，-1为AI棋子，+1为玩家棋子
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
    childList = getNextSteps(board)

    best_row, best_col = 0, 0

    if turn == 1:
        # max
        for row, col in childList:
            # 递归调用
            board[row, col] = turn
            _, _, next_beta = alpha_beta(board, -turn, alpha, beta, depth-1)
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
            _, _, next_alpha = alpha_beta(board, -turn, alpha, beta, depth-1)
            board[row, col] = 0
            # 更新beta
            if beta > next_alpha:
                beta = next_alpha
                best_row, best_col = row, col
            # 剪枝判断
            if beta <= alpha:
                break
        return best_row, best_col, beta


def getNextSteps(board):
    '''
        获取可行的下一步列表
        参数：
            board: numpy.array棋盘，0为空格，-1为AI棋子，+1为玩家棋子
        返回值：
            next_steps: 下一步列表
    '''
    # 确定棋子分布范围
    left = 0
    right = board.shape[1]-1
    sum_0 = np.sum(board!=0, axis=0)
    while sum_0[left] == 0:
        left += 1
    while sum_0[right] == 0:
        right -= 1
    
    top = 0
    bottom = board.shape[0]-1
    sum_1 = np.sum(board!=0, axis=1)
    while sum_1[top] == 0:
        top += 1
    while sum_1[bottom] == 0:
        bottom -= 1

    # 扩充一格的范围
    top = max(top-1, 0)
    bottom = min(bottom+1, board.shape[0]-1)
    left = max(left-1, 0)
    right = min(right+1, board.shape[1]-1)

    # 统计四邻域的棋子数
    board_count = np.zeros_like(board)
    next_steps = []

    for i in range(top, bottom+1):
        for j in range(left, right+1):
            if board[i, j] == 0:
                next_steps.append((i, j))
            else:
                board_count[max(0, i-1):min(board.shape[0]-1, i+2), \
                        max(0, j-1):min(board.shape[1], j+2)] += 1
    
    # 忽略四领域没有棋子的格子
    record = []
    for i in range(len(next_steps)):
        row, col = next_steps[i]
        if board_count[row, col] == 0:
            record.append(i)
    next_steps = [next_steps[i] for i in range(len(next_steps)) if i not in record]

    # 根据四领域的棋子数进行排序
    next_steps.sort(key=lambda x: board_count[x[0], x[1]], reverse=True)

    return next_steps

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

    # 翻转
    board_fp = np.fliplr(board)
    # 各个方向的向量
    board_vec = [i for i in board] + [i for i in board.T] \
        + [board.diagonal(i) for i in range(-board.shape[0]+1, board.shape[1])] \
        + [board_fp.diagonal(i) for i in range(-board_fp.shape[0]+1, board_fp.shape[1])]

    # 遍历各种棋型
    for model, model_score in score_dict:
        # 统计频数
        pos, neg = conv2d(board_vec, model)
        # 更新分数
        max_score += pos * model_score
        min_score += neg * model_score

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
            pos, neg: 两个计算结果
    '''
    pos = 0
    neg = 0
    for aa in a:
        if len(aa) >= len(b):
            result = np.convolve([-1]+aa.tolist()+[-1], b, mode='valid')
            pos += np.sum(result==1)
            result = np.convolve([1]+aa.tolist()+[1], b, mode='valid')
            neg += np.sum(result==-1)
    return pos, neg


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
        + [board.diagonal(i) for i in range(-board.shape[0]+1, board.shape[1])] \
        + [board_fp.diagonal(i) for i in range(-board_fp.shape[0]+1, board_fp.shape[1])]

    model = np.array([1, 1, 1, 1, 1]) * turn / 5

    # 遍历每个向量
    for vec in board_vec:
        if len(vec) >= len(model) and np.sum(np.convolve(vec, model, mode='valid')==1) > 0:
            return True

    return False