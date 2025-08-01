# Heuristic-Driven IDA\* Rubik's Cube Algorithm

> Fast and efficient Rubik’s Cube solver built with Iterative Deepening A\* (IDA\*) using custom heuristics for both 3×3 and 2×2 configurations. Designed to be scalable, lightweight, and competition-ready.

---

## 🧰 Project Overview

This repository contains optimized Rubik’s Cube solvers (3×3 and 2×2) powered by a tailored **Heuristic-Driven IDA\*** algorithm. Rather than relying on hardcoded AI models or brute-force approaches, this solver uses deep domain knowledge, pruning strategies, and intelligent move ordering.

- 🔁 **Search Algorithm:** Iterative Deepening A\* (IDA\*)
- 🎯 **Heuristic:** Pattern Databases (PDB), cost-based depth
- ⚙️ **Scalability:** Modular design for extension to 4×4 and beyond

---

## 🧠 Heuristic Design

The heart of the solver is its **cross-state pattern database** heuristic:

- Precomputes all possible cross patterns and minimal move depths
- Stores only legal cube states for efficiency
- Enables IDA\* to skip vast invalid branches early
- Separate heuristics tailored for 3×3 and 2×2 logic

---

## 🔍 How It Works

1. **Start State** → Encoded into a flattened cube representation
2. **IDA\* Loop** → Iteratively increases depth while pruning using heuristic
3. **Move Ordering** → Prioritizes axes and face turns intelligently
4. **Goal Check** → Verifies solved state and returns shortest path

---

## 🚀 Performance

| Cube | Avg. Moves | Solve Time | Memory |
|------|------------|------------|--------|
| 2×2  | 7–12       | < 1 sec    | ~5 MB  |
| 3×3  | 20–25      | < 3 sec    | ~10 MB |

- Tested on: Intel i5, 8GB RAM
- No external ML libraries required
---

## 🧮 Time & Space Complexity

- **Time:** Exponential worst-case \( O(b^d) \), but greatly reduced via PDB pruning
- **Space:** Linear in depth, with heuristic lookup kept <20 KB in memory

---

## 🧱 Future Extensions

- 4×4 & 5×5 support using center/edge pairing heuristics
- Visual GUI using PyGame / OpenGL
- Web version with React + WASM

---

## 🧩 Run Instructions

# Run 3x3 Solver
python src/solver_3x3.py

# Run 2x2 Solver
python src/solver_2x2.py


