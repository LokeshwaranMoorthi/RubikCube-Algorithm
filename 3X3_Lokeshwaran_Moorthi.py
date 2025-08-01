import json
import os
from random import randint, choice
from tqdm import tqdm
import unittest
import sys

# Increase recursion limit for deep searches
sys.setrecursionlimit(2000)


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
                self.cube[0][i][column], self.cube[2][i][column], self.cube[4][-i - 1][-column - 1], self.cube[5][i][
                    column] = (
                    self.cube[4][-i - 1][-column - 1], self.cube[0][i][column],
                    self.cube[5][i][column], self.cube[2][i][column])
            elif direction == 1:
                self.cube[0][i][column], self.cube[2][i][column], self.cube[4][-i - 1][-column - 1], self.cube[5][i][
                    column] = (
                    self.cube[2][i][column], self.cube[5][i][column],
                    self.cube[0][i][column], self.cube[4][-i - 1][-column - 1])
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
                self.cube[0][column][i], self.cube[1][-i - 1][column], self.cube[3][i][-column - 1], \
                self.cube[5][-column - 1][-1 - i] = (
                    self.cube[3][i][-column - 1], self.cube[0][column][i],
                    self.cube[5][-column - 1][-1 - i], self.cube[1][-i - 1][column])
            elif direction == 1:
                self.cube[0][column][i], self.cube[1][-i - 1][column], self.cube[3][i][-column - 1], \
                self.cube[5][-column - 1][-1 - i] = (
                    self.cube[1][-i - 1][column], self.cube[5][-column - 1][-1 - i],
                    self.cube[0][column][i], self.cube[3][i][-column - 1])
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
        self.heuristic = heuristic
        self.threshold = 0  # Initialize, will be set in run()

    def run(self, initial_state):
        # Initialize threshold with the heuristic value of the start state
        self.threshold = self.heuristic.get(initial_state, self.max_depth)

        while True:
            # We don't reset self.moves here, as it's not used in the new IDA* search structure
            result = self.search(initial_state, 0, [])

            if isinstance(result, list):
                # Solution found!
                return result

            if result == float('inf'):
                # No solution found within practical limits or if heuristic is too high
                return None

            # Increase threshold for the next iteration
            self.threshold = result

    def search(self, state, g_score, path):
        cube = RubiksCube(state=state)
        h_score = self.heuristic.get(state, self.max_depth)
        f_score = g_score + h_score

        if f_score > self.threshold:
            return f_score

        if cube.solved():
            return path

        min_f_score = float('inf')

        # Determine actions for the current cube size
        n = cube.n
        actions = [(r, n_val, d) for r in ['h', 'v', 's'] for d in [0, 1] for n_val in range(n)]

        for a in actions:
            temp_cube = RubiksCube(state=state)
            if a[0] == 'h':
                temp_cube.horizontal_twist(a[1], a[2])
            elif a[0] == 'v':
                temp_cube.vertical_twist(a[1], a[2])
            elif a[0] == 's':
                temp_cube.side_twist(a[1], a[2])

            result = self.search(temp_cube.stringify(), g_score + 1, path + [a])

            if isinstance(result, list):
                return result

            min_f_score = min(min_f_score, result)

        return min_f_score


def build_heuristic_db(state, actions, max_moves=5, heuristic=None):
    if heuristic is None:
        heuristic = {state: 0}
    que = [(state, 0)]
    visited = {state}

    # Estimate total nodes for progress bar
    n_val_for_actions = RubiksCube(state=state).n
    num_actions = len([(r, n_val, d) for r in ['h', 'v', 's'] for d in [0, 1] for n_val in range(n_val_for_actions)])
    node_count_estimate = sum([num_actions ** x for x in range(max_moves + 1)])

    with tqdm(total=node_count_estimate, desc=f'Heuristic DB (N={n_val_for_actions}, Max_Moves={max_moves})') as pbar:
        head = 0
        while head < len(que):
            s, d = que[head]
            head += 1

            if d >= max_moves:
                # To prevent pbar from overshooting if many branches hit max_moves
                # This approximation might still be off but better than previous.
                # A more accurate way would be to count unique states added.
                # For now, this is okay for a visual indicator.
                pbar.update(num_actions if head < len(que) else 0)
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
                    if a_str not in visited:
                        que.append((a_str, d + 1))
                        visited.add(a_str)
                pbar.update(1)
    return heuristic


# ----------------- Main Driver -----------------

if __name__ == '__main__':
    CUBE_SIZE = 3
    # WARNING: Building the heuristic DB for a 3x3x3 with MAX_HEURISTIC_MOVES > 2-3
    # is extremely memory and computationally intensive and may take a very long time.
    MAX_HEURISTIC_MOVES = 2  # Keep this low for 3x3x3 as building larger DB is very slow
    MAX_SOLVER_DEPTH = 30  # Max depth for IDA* solver
    HEURISTIC_FILE = f'heuristic_n{CUBE_SIZE}_m{MAX_HEURISTIC_MOVES}.json'
    NEW_HEURISTICS = False  # Set to True to force rebuilding the heuristic DB

    cube = RubiksCube(n=CUBE_SIZE)
    print(f"Initial Solved {CUBE_SIZE}x{CUBE_SIZE}x{CUBE_SIZE} Cube:")
    cube.show()
    print('-----------')

    h_db = None
    if os.path.exists(HEURISTIC_FILE):
        try:
            with open(HEURISTIC_FILE) as f:
                h_db = json.load(f)
            print(f"Loaded heuristic database from {HEURISTIC_FILE}")
        except json.JSONDecodeError:
            print(f"Error decoding {HEURISTIC_FILE}. Rebuilding heuristic database.")
            NEW_HEURISTICS = True
    else:
        print(f"Heuristic file {HEURISTIC_FILE} not found. Building new heuristic database.")
        NEW_HEURISTICS = True

    if h_db is None or NEW_HEURISTICS:
        print("Building heuristic database...")
        actions = [(r, n_val, d) for r in ['h', 'v', 's'] for d in [0, 1] for n_val in range(cube.n)]
        h_db = build_heuristic_db(cube.stringify(), actions, max_moves=MAX_HEURISTIC_MOVES)
        with open(HEURISTIC_FILE, 'w') as f:
            json.dump(h_db, f, indent=4)
        print(f"Heuristic database built and saved to {HEURISTIC_FILE}. Size: {len(h_db)}")

    # Shuffle the cube
    # For a 3x3x3, a small number of shuffle moves is better for testing with limited heuristic depth.
    # Keep SHUFFLE_U_ROT low, especially with MAX_HEURISTIC_MOVES=2, otherwise it might not find a solution.
    SHUFFLE_L_ROT = 2
    SHUFFLE_U_ROT = 5
    cube.shuffle(l_rot=SHUFFLE_L_ROT, u_rot=SHUFFLE_U_ROT)
    print(f'\nShuffled {CUBE_SIZE}x{CUBE_SIZE}x{CUBE_SIZE} Cube:')
    cube.show()
    print('----------- Solving -----------')

    solver = IDA_star(h_db, max_depth=MAX_SOLVER_DEPTH)

    try:
        moves = solver.run(cube.stringify())
        if moves:
            print(f'Solution found! Total moves: {len(moves)}')
            print(f'Moves: {moves}')

            print('\nApplying solution moves:')
            # Create a new cube for verification to avoid state issues
            verif_cube = RubiksCube(state=cube.stringify())
            for i, m in enumerate(moves):
                if m[0] == 'h':
                    verif_cube.horizontal_twist(m[1], m[2])
                elif m[0] == 'v':
                    verif_cube.vertical_twist(m[1], m[2])
                elif m[0] == 's':
                    verif_cube.side_twist(m[1], m[2])

            print('\nCube after applying solution:')
            verif_cube.show()
            if verif_cube.solved():
                print("Cube is solved!!")
            else:
                print("Cube is NOT solved. There might be an issue with the solver or twists.")
        else:
            print("No solution found within the given max_depth or with the current heuristic.")
            print(
                "Consider increasing MAX_SOLVER_DEPTH or rebuilding the heuristic database with a larger MAX_HEURISTIC_MOVES (WARNING: This can take a very, very long time for 3x3x3).")
    except RecursionError:
        print("\nRecursion depth limit reached. The problem might be too complex for the current settings.")
        print(
            "Try increasing the Python recursion limit (sys.setrecursionlimit) or running with a larger MAX_HEURISTIC_MOVES to get a better heuristic.")

    # Toggle tests
    run_tests = False
    if run_tests:
        print("\nRunning unit tests...")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)