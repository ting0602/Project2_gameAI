import STcpClient_1 as STcpClient
import numpy as np
import random


'''
    input position (x,y) and direction
    output next node position on this direction
'''
def Next_Node(pos_x,pos_y,direction):
    if pos_y%2==1:
        if direction==1:
            return pos_x,pos_y-1
        elif direction==2:
            return pos_x+1,pos_y-1
        elif direction==3:
            return pos_x-1,pos_y
        elif direction==4:
            return pos_x+1,pos_y
        elif direction==5:
            return pos_x,pos_y+1
        elif direction==6:
            return pos_x+1,pos_y+1
    else:
        if direction==1:
            return pos_x-1,pos_y-1
        elif direction==2:
            return pos_x,pos_y-1
        elif direction==3:
            return pos_x-1,pos_y
        elif direction==4:
            return pos_x+1,pos_y
        elif direction==5:
            return pos_x-1,pos_y+1
        elif direction==6:
            return pos_x,pos_y+1


def checkRemainMove(mapStat):
    free_region = (mapStat == 0)
    temp = []
    for i in range(len(free_region)):
        for j in range(len(free_region[0])):
            if(free_region[i][j] == True):
                temp.append([i,j])
    return temp


'''
    輪到此程式移動棋子
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 0=可移動區域, -1=障礙, 1~2為玩家1~2佔領區域
    gameStat : 棋盤歷史順序
    return Step
    Step : 3 elements, [(x,y), l, dir]
            x, y 表示要畫線起始座標
            l = 線條長度(1~3)
            dir = 方向(1~6),對應方向如下圖所示
              1  2
            3  x  4
              5  6
'''

# TODO: implement the Getstep() function and necessary functions
import math
import random
def Getstep(mapStat, gameStat):
    # define necessary functions
    def get_available_cells(mapStat):
        return [(i, j) for i in range(len(mapStat)) for j in range(len(mapStat[i])) if mapStat[i][j] == 0]
    def get_actions(mapStat):
        cells = get_available_cells(mapStat=mapStat)
        actions = [ [pos, 1, 1] for pos in cells]
        # dir
        #   1  2
        # 3  x  4
        #   5  6
        b = len(mapStat)
        # dir:1 
        actions += [[(i,j), 2, 5] for i, j in cells if j//2==0 and i-1>=0 and j+1<b and mapStat[i-1][j+1]==0]
        actions += [[(i,j), 2, 5] for i, j in cells if j//2==1 and j+1<b and mapStat[i][j+1]==0]
        actions += [[(i,j), 3, 5] for i, j in cells if j//2==0 and i-1>=0 and j+2<b and mapStat[i-1][j+1]==0 and mapStat[i-1][j+2]==0]
        actions += [[(i,j), 3, 5] for i, j in cells if j//2==1 and i-1>=0 and j+2<b and mapStat[i][j+1]==0 and mapStat[i-1][j+2]==0]

        # dir: 2
        actions += [[(i,j), 2, 6] for i, j in cells if j//2==1 and i+1<b and j+1<b and mapStat[i+1][j+1]==0]
        actions += [[(i,j), 2, 6] for i, j in cells if j//2==0 and j+1<b and mapStat[i][j+1]==0]
        actions += [[(i,j), 3, 6] for i, j in cells if j//2==1 and i+1<b and j+2<b and mapStat[i+1][j+1]==0 and mapStat[i+1][j+2]==0]
        actions += [[(i,j), 3, 6] for i, j in cells if j//2==0 and i+1<b and j+2<b and mapStat[i][j+1]==0 and mapStat[i+1][j+2]==0]

        # dir: 3
        actions += [[(i,j), 2, 3] for i, j in cells if i-1>=0 and mapStat[i-1][j]==0]
        actions += [[(i,j), 3, 3] for i, j in cells if i-2>=0 and mapStat[i-1][j]==0 and mapStat[i-2][j]==0]

        # dir: 4
        actions += [[(i,j), 2, 4] for i, j in cells if i+1<b and mapStat[i+1][j]==0]
        actions += [[(i,j), 3, 4] for i, j in cells if i+2<b and mapStat[i+1][j]==0 and mapStat[i+2][j]==0]

        # dir:5
        actions += [[(i,j), 2, 1] for i, j in cells if j//2==0 and i-1>=0 and j-1>=0 and mapStat[i-1][j-1]==0]
        actions += [[(i,j), 2, 1] for i, j in cells if j//2==1 and j-1>=0 and mapStat[i][j-1]==0]
        actions += [[(i,j), 3, 1] for i, j in cells if j//2==0 and i-1>=0 and j-2>=0 and mapStat[i-1][j-1]==0 and mapStat[i-1][j-2]==0]
        actions += [[(i,j), 3, 1] for i, j in cells if j//2==1 and i-1>=0 and j-2>=0 and mapStat[i][j-1]==0 and mapStat[i-1][j-2]==0]

        # dir: 6
        actions += [[(i,j), 2, 2] for i, j in cells if j//2==1 and i+1<b and j-1>=0 and mapStat[i+1][j-1]==0]
        actions += [[(i,j), 2, 2] for i, j in cells if j//2==0 and j-1>=0 and mapStat[i][j-1]==0]
        actions += [[(i,j), 3, 2] for i, j in cells if j//2==1 and i+1<b and j-2>=0 and mapStat[i+1][j-1]==0 and mapStat[i+1][j-2]==0]
        actions += [[(i,j), 3, 2] for i, j in cells if j//2==0 and i+1<b and j-2>=0 and mapStat[i][j-1]==0 and mapStat[i+1][j-2]==0]

        return actions    
    
    def update_mapStat(mapStat, action, player):
        pos, l, d = action
        x, y = pos
        mapStat[x][y]==player
        if l == 1:
            return mapStat
        
        if d == 1:
            if y // 2 == 0:
                mapStat[x-1][y-1] = player
            else:
                mapStat[x][y-1] = player
            if l == 3:
                mapStat[x-1][y-2] = player
        elif d == 2:
            if y // 2 == 0:
                mapStat[x][y-1] = player
            else:
                mapStat[x+1][y-1] = player
            if l == 3:
                mapStat[x+1][y-2] = player
        elif d == 3:
            for i in range(1, l):
                mapStat[x-i][y] = player
        elif d == 4:
            for i in range(1, l):
                mapStat[x+i][y] = player  
        elif d == 5:
            if y // 2 == 0:
                mapStat[x-1][y+1] = player
            else:
                mapStat[x][y+1] = player
            if l == 3:
                mapStat[x-1][y+2] = player
        elif d == 6:
            if y // 2 == 0:
                mapStat[x][y+1] = player
            else:
                mapStat[x+1][y+1] = player
            if l == 3:
                mapStat[x+1][y+2] = player

        return mapStat


    # player1: me, player 2: other
    def get_score(player, mapStat):
        # let 3 cells be a group
        null_cells = len(get_available_cells(mapStat))
        if null_cells == 1:
            if player == 1:
                return float('-inf')
            else:
                return float('inf')
        if player == 1:
            return -null_cells
        else:
            return null_cells
    
    ############### MCTS ###############
    class Node:
        def __init__(self, parent=None, action=None):
            self.parent = parent
            self.children = []
            self.visits = 0
            self.reward = 0
            self.untried_actions = get_actions(mapStat)
            self.action = action

        def expand(self):
            action = self.untried_actions.pop()
            child = Node(parent=self, action=action)
            self.children.append(child)
            return child

        def select_child(self):
            child_scores = [
                (c.reward / c.visits) + math.sqrt(2 * math.log(self.visits) / c.visits)
                for c in self.children
            ]
            return self.children[child_scores.index(max(child_scores))]

        def update(self, reward):
            self.visits += 1
            self.reward += reward

        def fully_expanded(self):
            return not self.untried_actions

        def __repr__(self):
            return f"Node(visits={self.visits}, reward={self.reward}, action={self.action})"

    def mcts(root, player):
        for i in range(5):
            node = root
            mapStat = root.mapStat.copy()
            # gameStat = root.gameStat.copy()

            # Select
            while node.fully_expanded() and node.children:
                node = node.select_child()
                action = node.action
                mapStat = update_mapStat(mapStat, action, player)
                # gameStat.append((player, action))

                # Switch players
                player = 1 if player == 2 else 2

            # Expand
            if not node.fully_expanded():
                node = node.expand()
                action = node.action
                mapStat = update_mapStat(mapStat, action, player)
                # gameStat.append((player, action))

                # Switch players
                player = 1 if player == 2 else 2

            # Rollout
            for j in range(10):
                if len(get_actions(mapStat)) == 0:
                    break
                if player == 1:
                    score, action = min_max(mapStat, player, 0, float('-inf'), float('inf'))
                else:
                    score, action = min_max(mapStat, player, 1, float('-inf'), float('inf'))
                mapStat = update_mapStat(mapStat, action, player)
                # gameStat.append((player, action))

                # Switch players
                player = 1 if player == 2 else 2

            # Backpropagate
            reward = get_score(player, mapStat)
            while node:
                node.update(reward)
                node = node.parent
        return max(root.children, key=lambda c: c.visits)

    def get_best_action(mapStat, gameStat, player):
        root = Node()
        root.mapStat = mapStat
        # root.gameStat = gameStat

        for i in range(5):
            mcts(root, player)

        return max(root.children, key=lambda c: c.visits).action

    #########################################################
        
    def alpha_beta_pruning(mapStat, player, depth, alpha, beta):
        score = get_score(player, mapStat)
        if score == float('-inf') or score == float('inf') or depth >= 2 or len(get_actions(mapStat)) <= 1:
            return score
        else:
            best_score, best_action = min_max(mapStat, player, depth, alpha, beta)
            return best_score
        
    def min_max(mapStat, player, depth, alpha, beta):
        next_player = 2 if player == 1 else 1
        actions = get_actions(mapStat)
        if player == 1:
            scores = []
            maxVal = float("-inf")
            for action in actions:
                next_mapStat = update_mapStat(mapStat, action, player)
                val = alpha_beta_pruning(next_mapStat, next_player, depth, alpha, beta)
                scores.append((val, action))
                maxVal = max(maxVal, val)
                if maxVal >= beta:
                    return (maxVal, action)
                alpha = max(alpha, maxVal)
            best_score = max(scores)
            bestIndex = [index for index in range(len(scores)) if scores[index] == best_score] 
            chosenIndex = random.choice(bestIndex)
            return scores[chosenIndex]
        # player2
        else:
            scores = []
            minVal = float("inf")
            for action in actions:
                next_player = 1
                next_mapStat = update_mapStat(mapStat, action, player)
                val = alpha_beta_pruning(next_mapStat, next_player, depth+1, alpha, beta)
                scores.append((val, action))
                minVal = min(minVal, val)
                if minVal <= alpha:
                    return (minVal, action)
                beta = min(beta, minVal)

            best_score = min(scores)
            bestIndex = [index for index in range(len(scores)) if scores[index] == best_score] 
            chosenIndex = random.choice(bestIndex)
            return scores[chosenIndex]
    
    # alpha = float("-inf")
    # beta = float("inf")
    # best_score, best_step = min_max(mapStat, 1, 0, alpha, beta)
    # return best_step
    output = get_best_action(mapStat, gameStat, 1)
    print("output", output)
    # print("best_step, best_score:",best_step, best_score)
    return output



# start game
print('start game')
while (True):

    (end_program, id_package, mapStat, gameStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    
    decision_step = Getstep(mapStat, gameStat)
    
    STcpClient.SendStep(id_package, decision_step)
