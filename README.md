# SOW-BKI258-Reinforcement-Learning

A reinforcement learning environment where an agent controls a mouse to push cheese into a hole.

We're reinforcing the learning I think.

## Goal

The goal is to push the cheese (c) into the hole (h) at the top-right corner of the board. The mouse (m) must navigate the 3×4 grid and push the cheese toward the goal.

## Board Layout

```
      ╭───╮
╭─────╯ h │  ← Goal: Hole at position [-1, 3]
│ 0 0 0 0 │
│ m c 0 0 │  ← Example: mouse and cheese on row 1
│ 0 0 0 0 │
╰─────────╯
```

**Legend:**
- **m**: Mouse (controlled agent)
- **c**: Cheese (object to push)
- **h**: Hole (goal position)
- **0**: Empty cells

## Environment Specifications

### Action Space
- **0**: Move/Push Up
- **1**: Move/Push Right
- **2**: Move/Push Down
- **3**: Move/Push Left

Total Actions $= 4$

### State Space
- Mouse can be in $12$ positions
- Cheese can be in $11$ remaining positions
- Plus $1$ terminal state (cheese in hole)

Total States $= 12 \cdot 11 + 1 = 133$

### Rewards
- **Win reward**: +100 (cheese pushed into hole)
- **Step punishment**: -1 (per step, incentivizes shortest path)
- **Impossible action**: -2 (moving into wall)
- **Lose punishment**: -50 (cheese reaches column 0 or row 2)

### Terminal Conditions
1. **Win**: Cheese is pushed to the hole at position [-1, 3]
2. **Lose**: Cheese reaches column 0 (left column) or row 2 (bottom row)

### Initial Random Position
- **Valid spawn positions**: Rows 0-1, Columns 1-3
- **Mouse spawn**: Any position that doesn't overlap with cheese