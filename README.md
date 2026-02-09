# SOW-BKI258-Reinforcement-Learning
We're reinforcing the learning i think


# Playing field
Below there is an example state of the game board.
0 0 0 0 0 0 0 <br>
0 0 0 0 0 0 0 <br>
0 0 0 0 0 0 0 <br>
0 1 2 3 0 d 0 <br>
0 0 0 0 0 0 0 <br>
0 0 0 0 0 0 0 <br>
0 0 0 0 0 0 0 <br>

0 indicates an empty space
d indicates the target (dirt) the worm needs to reach to grow
All the other numbers are the worm. The highest number is the head of the worm (front) and the 1 is the tail (last part of the worm and is 0 (empty space) in the next turn (unless it reaches the target))

# Basic Worm Step
1. Get all 4 neighbours of the highest number.
2. Keep every option that is not (highest_number - 1).
3. **Action:** Choose one of the 3 options.
4. **Result:**
    - if chosen_state == d: chosen_state = (highest_number + 1)
    - if chosen_state == 0: chosen_state = (highest_number + 1) <br>
                            AND THEN for all values > 0 => value -= 1      
    - else: Game Over