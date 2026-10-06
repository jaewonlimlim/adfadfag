"""15-puzzle solver using a discrete Artificial Bee Colony (ABC) search.

This is a compact, runnable adaptation of Tuncer (2021):
"15-Puzzle Problem Solving with the Artificial Bee Colony Algorithm
Based on Pattern Database".  Each food source is a sequence of blank-tile
moves.  Its quality uses Manhattan distance plus linear conflicts.

This is a stochastic, satisficing solver, not an optimal solver.  It works
best for moderately scrambled puzzles; run it more than once for hard cases.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

SIZE = 4
GOAL = tuple(range(1, 16)) + (0,)

# Actions move the blank.  Opposites are used only to remove immediately
# cancelling moves from a food source before it is evaluated.
MOVES = ("L", "R", "U", "D")
OPPOSITE = {"L": "R", "R": "L", "U": "D", "D": "U"}


def print_board(state: tuple[int, ...]) -> None:
    """Print a puzzle state in a readable 4 x 4 layout."""
    for r in range(SIZE):
        row = state[r * SIZE : (r + 1) * SIZE]
        print(" ".join(f"{tile:2}" if tile else " _" for tile in row))
    print()


def is_solvable(state: tuple[int, ...]) -> bool:
    """Return whether a 4 x 4 puzzle can reach GOAL."""
    values = [tile for tile in state if tile]
    inversions = sum(
        values[i] > values[j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    blank_row_from_bottom = SIZE - (state.index(0) // SIZE)
    # For an even-width board, the parity must differ.
    return (inversions + blank_row_from_bottom) % 2 == 1


def apply_move(state: tuple[int, ...], move: str) -> tuple[int, ...] | None:
    """Apply one legal blank move, or return None when it leaves the board."""
    blank = state.index(0)
    row, col = divmod(blank, SIZE)
    dr, dc = {"L": (0, -1), "R": (0, 1), "U": (-1, 0), "D": (1, 0)}[move]
    nr, nc = row + dr, col + dc

    if not (0 <= nr < SIZE and 0 <= nc < SIZE):
        return None

    next_blank = nr * SIZE + nc
    result = list(state)
    result[blank], result[next_blank] = result[next_blank], result[blank]
    return tuple(result)


def normalize_moves(moves: Iterable[str]) -> list[str]:
    """Drop immediate opposite actions: L,R or U,D causes no net movement."""
    result: list[str] = []
    for move in moves:
        if result and OPPOSITE[move] == result[-1]:
            result.pop()
        else:
            result.append(move)
    return result


def execute_moves(
    state: tuple[int, ...], moves: Iterable[str]
) -> tuple[tuple[int, ...], list[str]]:
    """Execute a food source; ignore invalid moves as in the paper."""
    current = state
    executed: list[str] = []
    for move in normalize_moves(moves):
        nxt = apply_move(current, move)
        if nxt is not None:
            current = nxt
            executed.append(move)
    return current, executed


def solution_states(
    initial: tuple[int, ...], moves: Iterable[str]
) -> list[tuple[int, ...]]:
    """Return every board state from the initial board through the final move."""
    states = [initial]
    current = initial
    for move in moves:
        next_state = apply_move(current, move)
        if next_state is None:
            raise ValueError(f"{move!r} is not a legal move from the current board")
        states.append(next_state)
        current = next_state
    return states


def save_solution_gif(
    initial: tuple[int, ...],
    moves: Iterable[str],
    output_path: str | Path = "15_puzzle_solution.gif",
    *,
    interval_ms: int = 700,
) -> Path:
    """Save every solution step as a GIF animation.

    The initial board is frame 0.  Later frames show the action that moved the
    blank to reach that board.  Requires matplotlib and Pillow.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle

    moves = list(moves)
    states = solution_states(initial, moves)
    output_path = Path(output_path)

    figure, axis = plt.subplots(figsize=(5.8, 6.5))

    def draw(frame: int) -> None:
        axis.clear()
        axis.set_xlim(0, SIZE)
        axis.set_ylim(SIZE, 0)
        axis.set_aspect("equal")
        axis.axis("off")

        state = states[frame]
        for index, tile in enumerate(state):
            row, col = divmod(index, SIZE)
            if tile == 0:
                fill, label = "#FFF4CE", ""
            elif tile == index + 1:
                fill, label = "#CFE7D4", str(tile)
            else:
                fill, label = "#D9E9F8", str(tile)

            axis.add_patch(
                Rectangle(
                    (col + 0.04, row + 0.04), 0.92, 0.92,
                    facecolor=fill, edgecolor="#29445E", linewidth=1.8,
                )
            )
            if label:
                axis.text(
                    col + 0.5, row + 0.56, label,
                    ha="center", va="center", fontsize=22,
                    fontweight="bold", color="#172B3A",
                )

        move_text = "initial board" if frame == 0 else f"move {frame}: {moves[frame - 1]}"
        status = "SOLVED" if state == GOAL else f"heuristic: {heuristic(state)}"
        axis.set_title(f"15-Puzzle ABC solution\n{move_text}   |   {status}", fontsize=15, pad=20)

    animation = FuncAnimation(
        figure, draw, frames=len(states), interval=interval_ms, repeat=True
    )
    fps = max(1, round(1000 / interval_ms))
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return output_path


def manhattan_distance(state: tuple[int, ...]) -> int:
    """Sum tile distances from their goal cells, excluding the blank."""
    distance = 0
    for index, tile in enumerate(state):
        if tile == 0:
            continue
        goal_index = tile - 1
        row, col = divmod(index, SIZE)
        goal_row, goal_col = divmod(goal_index, SIZE)
        distance += abs(row - goal_row) + abs(col - goal_col)
    return distance


def linear_conflicts(state: tuple[int, ...]) -> int:
    """Count row/column tile pairs that must cross before reaching GOAL."""
    conflicts = 0

    # Row conflicts
    for row in range(SIZE):
        tiles = [state[row * SIZE + col] for col in range(SIZE)]
        for i in range(SIZE):
            first = tiles[i]
            if first == 0 or (first - 1) // SIZE != row:
                continue
            for j in range(i + 1, SIZE):
                second = tiles[j]
                if second == 0 or (second - 1) // SIZE != row:
                    continue
                if (first - 1) % SIZE > (second - 1) % SIZE:
                    conflicts += 1

    # Column conflicts
    for col in range(SIZE):
        tiles = [state[row * SIZE + col] for row in range(SIZE)]
        for i in range(SIZE):
            first = tiles[i]
            if first == 0 or (first - 1) % SIZE != col:
                continue
            for j in range(i + 1, SIZE):
                second = tiles[j]
                if second == 0 or (second - 1) % SIZE != col:
                    continue
                if (first - 1) // SIZE > (second - 1) // SIZE:
                    conflicts += 1
    return conflicts


def heuristic(state: tuple[int, ...]) -> int:
    """The paper's objective: Manhattan distance + 2 * linear conflicts."""
    return manhattan_distance(state) + 2 * linear_conflicts(state)


def make_food(length: int) -> list[str]:
    return [random.choice(MOVES) for _ in range(length)]


def mutate_food(food: list[str], change_rate: float) -> list[str]:
    """Discrete version of the paper's modified ABC neighbour operator."""
    candidate = food[:]
    changed = False
    for i in range(len(candidate)):
        if random.random() < change_rate:
            candidate[i] = random.choice(MOVES)
            changed = True
    if not changed:
        i = random.randrange(len(candidate))
        candidate[i] = random.choice(MOVES)
    return candidate


def evaluate(base: tuple[int, ...], food: list[str]) -> tuple[int, tuple[int, ...], list[str]]:
    state, actual_moves = execute_moves(base, food)
    # A very small length cost breaks equal-heuristic ties in favour of shorter paths.
    score = heuristic(state) * 1000 + len(actual_moves)
    return score, state, actual_moves


def roulette_select(scores: list[int]) -> int:
    """Choose a food source; smaller objective scores receive larger weights."""
    worst = max(scores)
    weights = [worst - score + 1 for score in scores]
    chosen = random.uniform(0, sum(weights))
    cumulative = 0.0
    for i, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= chosen:
            return i
    return len(scores) - 1


def abc_at_depth(
    base: tuple[int, ...],
    food_length: int,
    food_count: int,
    limit: int,
    cycles: int,
    change_rate: float,
) -> tuple[tuple[int, ...], list[str]]:
    """Run employed/onlooker/scout phases for one iterative-deepening depth."""
    foods = [make_food(food_length) for _ in range(food_count)]
    results = [evaluate(base, food) for food in foods]
    scores = [result[0] for result in results]
    trials = [0] * food_count

    best_index = min(range(food_count), key=lambda i: scores[i])
    best_state, best_moves = results[best_index][1], results[best_index][2]

    for _ in range(cycles):
        # Employed bees: every food source receives one local-search attempt.
        for i in range(food_count):
            candidate = mutate_food(foods[i], change_rate)
            candidate_result = evaluate(base, candidate)
            if candidate_result[0] < scores[i]:
                foods[i], results[i], scores[i] = candidate, candidate_result, candidate_result[0]
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlookers: promising sources receive additional local-search attempts.
        for _ in range(food_count):
            i = roulette_select(scores)
            candidate = mutate_food(foods[i], change_rate)
            candidate_result = evaluate(base, candidate)
            if candidate_result[0] < scores[i]:
                foods[i], results[i], scores[i] = candidate, candidate_result, candidate_result[0]
                trials[i] = 0
            else:
                trials[i] += 1

        # Scouts: replace only sources that have not improved for `limit` trials.
        for i in range(food_count):
            if trials[i] >= limit:
                foods[i] = make_food(food_length)
                results[i] = evaluate(base, foods[i])
                scores[i] = results[i][0]
                trials[i] = 0

        best_index = min(range(food_count), key=lambda i: scores[i])
        if scores[best_index] < heuristic(best_state) * 1000 + len(best_moves):
            best_state, best_moves = results[best_index][1], results[best_index][2]
        if best_state == GOAL:
            break

    return best_state, best_moves


def solve_15_puzzle_abc(
    initial: Iterable[int],
    *,
    food_count: int = 80,
    limit: int = 100,
    cycles_per_depth: int = 100,
    initial_food_length: int = 30,
    max_depths: int = 30,
    change_rate: float = 0.25,
    seed: int | None = None,
) -> tuple[list[str], tuple[int, ...]]:
    """Attempt a solution using paper-style ABC plus iterative deepening.

    A depth here is an ABC restart.  The best state obtained at a depth becomes
    the base state for the next depth, following the iterative improvement
    described in the paper.  Returned moves are legal blank moves: L/R/U/D.
    """
    if seed is not None:
        random.seed(seed)
    state = tuple(initial)
    if len(state) != 16 or set(state) != set(range(16)):
        raise ValueError("initial must contain every integer from 0 to 15 exactly once")
    if not is_solvable(state):
        raise ValueError("This 15-puzzle is unsolvable because its permutation parity is invalid")
    if state == GOAL:
        return [], state

    all_moves: list[str] = []
    best_h = heuristic(state)

    for depth in range(1, max_depths + 1):
        # The paper begins with a long food source and reduces it as the goal nears.
        food_length = max(6, initial_food_length - (depth - 1) // 2)
        next_state, moves = abc_at_depth(
            state, food_length, food_count, limit, cycles_per_depth, change_rate
        )

        if heuristic(next_state) <= best_h:
            state = next_state
            all_moves.extend(moves)
            best_h = heuristic(state)

        print(f"depth={depth:2}, heuristic={best_h:2}, total_moves={len(all_moves):3}")
        if state == GOAL:
            return all_moves, state

    return all_moves, state


if __name__ == "__main__":
    # A solvable board scrambled by a short sequence from the goal state.
    initial_board = (
        9, 1, 7, 8,
        6, 15, 5, 12,
        13, 14, 4, 3,
        10, 0, 11, 2,
    )

    print("Initial board")
    print_board(initial_board)

    moves, final_board = solve_15_puzzle_abc(
        initial_board,
        food_count=160,
        limit=200,
        cycles_per_depth=400,
        initial_food_length=35,
        max_depths=60,
        change_rate=0.35,
        seed=7,
    )

    print("Moves:", " ".join(moves))
    print("Final board")
    print_board(final_board)
    print("Solved:", final_board == GOAL)

    if final_board == GOAL:
        gif_path = save_solution_gif(initial_board, moves)
        print(f"Solution animation saved to: {gif_path.resolve()}")