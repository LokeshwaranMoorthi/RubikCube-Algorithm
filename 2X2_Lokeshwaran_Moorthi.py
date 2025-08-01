import json
import os
from random import randint, choice
from tqdm import tqdm
import unittest

# ----------------- Cube Definition -----------------

class RubiksCube:
    def __init__(self, n=3, colours=['w', 'o', 'g', 'r', 'b', 'y'], state=None):
        self.n = n
        if state is None:
            self.colours = colours
            self.reset()
        else:
            self.n = int((len(state) / 6) ** 0.5)
            self.colours = []
            self.cube = [[[]]]
            for i, s in enumerate(state):
                if s not in self.colours:
                    self.colours.append(s)
                self.cube[-1][-1].append(s)
                if len(self.cube[-1][-1]) == self.n and len(self.cube[-1]) < self.n:
                    self.cube[-1].append([])
                elif len(self.cube[-1][-1]) == self.n and len(self.cube[-1]) == self.n and i < len(state) - 1:
                    self.cube.append([[]])

    def reset(self):
        self.cube = [[[c for _ in range(self.n)] for _ in range(self.n)] for c in self.colours]

    def solved(self):
        for side in self.cube:
            hold = []
            for row in side:
                if len(set(row)) == 1:
                    hold.append(row[0])
                else:
                    return False
            if len(set(hold)) > 1:
                return False
        return True

    def stringify(self):
        return ''.join(i for r in self.cube for s in r for i in s)

    def shuffle(self, l_rot=5, u_rot=100):
        moves = randint(l_rot, u_rot)
        actions = [('h', 0), ('h', 1), ('v', 0), ('v', 1), ('s', 0), ('s', 1)]
        for _ in range(moves):
            a = choice(actions)
            j = randint(0, self.n - 1)
            if a[0] == 'h':
                self.horizontal_twist(j, a[1])
            elif a[0] == 'v':
                self.vertical_twist(j, a[1])
            elif a[0] == 's':
                self.side_twist(j, a[1])

    def show(self):
        spacing = f'{" " * (len(str(self.cube[0][0])) + 2)}'
        l1 = '\n'.join(spacing + str(c) for c in self.cube[0])
        l2 = '\n'.join('  '.join(str(self.cube[i][j]) for i in range(1, 5)) for j in range(len(self.cube[0])))
        l3 = '\n'.join(spacing + str(c) for c in self.cube[5])
        print(f'{l1}\n\n{l2}\n\n{l3}')

    def horizontal_twist(self, row, direction):
        if row >= len(self.cube[0]):
            return
        if direction == 0:
            self.cube[1][row], self.cube[2][row], self.cube[3][row], self.cube[4][row] = (
                self.cube[2][row], self.cube[3][row], self.cube[4][row], self.cube[1][row])
        elif direction == 1:
            self.cube[1][row], self.cube[2][row], self.cube[3][row], self.cube[4][row] = (
                self.cube[4][row], self.cube[1][row], self.cube[2][row], self.cube[3][row])
        if direction == 0:
            if row == 0:
                self.cube[0] = [list(x) for x in zip(*reversed(self.cube[0]))]
            elif row == len(self.cube[0]) - 1:
                self.cube[5] = [list(x) for x in zip(*reversed(self.cube[5]))]
        elif direction == 1:
            if row == 0:
                self.cube[0] = [list(x) for x in zip(*self.cube[0])][::-1]
            elif row == len(self.cube[0]) - 1:
                self.cube[5] = [list(x) for x in zip(*self.cube[5])][::-1]

    def vertical_twist(self, column, direction):
        if column >= len(self.cube[0]):
            return
        for i in range(self.n):
            if direction == 0:
                self.cube[0][i][column], self.cube[2][i][column], self.cube[4][-i-1][-column-1], self.cube[5][i][column] = (
                    self.cube[4][-i-1][-column-1], self.cube[0][i][column],
                    self.cube[5][i][column], self.cube[2][i][column])
            elif direction == 1:
                self.cube[0][i][column], self.cube[2][i][column], self.cube[4][-i-1][-column-1], self.cube[5][i][column] = (
                    self.cube[2][i][column], self.cube[5][i][column],
                    self.cube[0][i][column], self.cube[4][-i-1][-column-1])
        if direction == 0:
            if column == 0:
                self.cube[1] = [list(x) for x in zip(*self.cube[1])][::-1]
            elif column == self.n - 1:
                self.cube[3] = [list(x) for x in zip(*self.cube[3])][::-1]
        elif direction == 1:
            if column == 0:
                self.cube[1] = [list(x) for x in zip(*reversed(self.cube[1]))]
            elif column == self.n - 1:
                self.cube[3] = [list(x) for x in zip(*reversed(self.cube[3]))]

    def side_twist(self, column, direction):
        if column >= len(self.cube[0]):
            return
        for i in range(self.n):
            if direction == 0:
                self.cube[0][column][i], self.cube[1][-i-1][column], self.cube[3][i][-column-1], self.cube[5][-column-1][-1-i] = (
                    self.cube[3][i][-column-1], self.cube[0][column][i],
                    self.cube[5][-column-1][-1-i], self.cube[1][-i-1][column])
            elif direction == 1:
                self.cube[0][column][i], self.cube[1][-i-1][column], self.cube[3][i][-column-1], self.cube[5][-column-1][-1-i] = (
                    self.cube[1][-i-1][column], self.cube[5][-column-1][-1-i],
                    self.cube[0][column][i], self.cube[3][i][-column-1])
        if direction == 0:
            if column == 0:
                self.cube[4] = [list(x) for x in zip(*reversed(self.cube[4]))]
            elif column == self.n - 1:
                self.cube[2] = [list(x) for x in zip(*reversed(self.cube[2]))]
        elif direction == 1:
            if column == 0:
                self.cube[4] = [list(x) for x in zip(*self.cube[4])][::-1]
            elif column == self.n - 1:
                self.cube[2] = [list(x) for x in zip(*self.cube[2])][::-1]

# ----------------- Solver -----------------

class IDA_star:
    def __init__(self, heuristic, max_depth=20):
        self.max_depth = max_depth
        self.threshold = max_depth
        self.min_threshold = None
        self.heuristic = heuristic
        self.moves = []

    def run(self, state):
        while True:
            status = self.search(state, 1)
            if status:
                return self.moves
            self.moves = []
            self.threshold = self.min_threshold

    def search(self, state, g_score):
        cube = RubiksCube(state=state)
        if cube.solved():
            return True
        elif len(self.moves) >= self.threshold:
            return False
        min_val = float('inf')
        best_action = None
        for a in [(r, n, d) for r in ['h', 'v', 's'] for d in [0, 1] for n in range(cube.n)]:
            cube = RubiksCube(state=state)
            if a[0] == 'h':
                cube.horizontal_twist(a[1], a[2])
            elif a[0] == 'v':
                cube.vertical_twist(a[1], a[2])
            elif a[0] == 's':
                cube.side_twist(a[1], a[2])
            if cube.solved():
                self.moves.append(a)
                return True
            cube_str = cube.stringify()
            h_score = self.heuristic.get(cube_str, self.max_depth)
            f_score = g_score + h_score
            if f_score < min_val:
                min_val = f_score
                best_action = [(cube_str, a)]
            elif f_score == min_val:
                best_action.append((cube_str, a))
        if best_action:
            if self.min_threshold is None or min_val < self.min_threshold:
                self.min_threshold = min_val
            next_action = choice(best_action)
            self.moves.append(next_action[1])
            return self.search(next_action[0], g_score + min_val)
        return False

def build_heuristic_db(state, actions, max_moves=5, heuristic=None):
    if heuristic is None:
        heuristic = {state: 0}
    que = [(state, 0)]
    node_count = sum([len(actions) ** (x + 1) for x in range(max_moves + 1)])
    with tqdm(total=node_count, desc='Heuristic DB') as pbar:
        while que:
            s, d = que.pop()
            if d > max_moves:
                continue
            for a in actions:
                cube = RubiksCube(state=s)
                if a[0] == 'h':
                    cube.horizontal_twist(a[1], a[2])
                elif a[0] == 'v':
                    cube.vertical_twist(a[1], a[2])
                elif a[0] == 's':
                    cube.side_twist(a[1], a[2])
                a_str = cube.stringify()
                if a_str not in heuristic or heuristic[a_str] > d + 1:
                    heuristic[a_str] = d + 1
                    que.append((a_str, d + 1))
                pbar.update(1)
    return heuristic

# ----------------- Main Driver -----------------

if __name__ == '__main__':
    MAX_MOVES = 5
    HEURISTIC_FILE = 'heuristic.json'
    NEW_HEURISTICS = False

    cube = RubiksCube(n=2)
    cube.show()
    print('-----------')

    if os.path.exists(HEURISTIC_FILE):
        with open(HEURISTIC_FILE) as f:
            h_db = json.load(f)
    else:
        h_db = None

    if h_db is None or NEW_HEURISTICS:
        actions = [(r, n, d) for r in ['h', 'v', 's'] for d in [0, 1] for n in range(cube.n)]
        h_db = build_heuristic_db(cube.stringify(), actions, max_moves=MAX_MOVES)
        with open(HEURISTIC_FILE, 'w') as f:
            json.dump(h_db, f, indent=4)

    cube.shuffle(l_rot=MAX_MOVES, u_rot=MAX_MOVES)
    cube.show()
    print('----------- Solving -----------')

    solver = IDA_star(h_db)
    moves = solver.run(cube.stringify())
    print(f'Moves: {moves}')

    for m in moves:
        if m[0] == 'h':
            cube.horizontal_twist(m[1], m[2])
        elif m[0] == 'v':
            cube.vertical_twist(m[1], m[2])
        elif m[0] == 's':
            cube.side_twist(m[1], m[2])
    cube.show()

    # Toggle tests
    run_tests = False
    if run_tests:
        unittest.main()
