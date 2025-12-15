import tkinter as tk
from tkinter import messagebox
import random
import pickle
import os
from PIL import ImageGrab  # Used for saving game screenshots

# --- Tic Tac Toe game class with basic Q-learning reinforcement logic ---
class TicTacToe:
    def __init__(self, master):
        self.master = master
        self.master.title("Tic Tac Toe")

        # --- Reinforcement learning parameters ---
        self.alpha = 0.1       # Learning rate (how much new info overrides old)
        self.epsilon = 0.2     # Exploration rate (chance of random action)

        # Q-table (state-action values) stored between sessions
        self.q_table_file = "ttt_q_table.pkl"
        self.q_table = self.load_q_table()  # Load if available, else empty dict

        # History of computer moves in the current game for updating Q-values
        self.game_history = []

        # Screenshot tracking variables
        self.game_number = 1   # For naming screenshots by game
        self.step_number = 0   # For naming screenshots by step

        # --- Board setup ---
        self.buttons = {}  
        self.board = [[None for _ in range(3)] for _ in range(3)]  # 3x3 matrix for board state
        self.game_over = False

        # Create 3x3 grid of clickable buttons
        for i in range(3):
            for j in range(3):
                button = tk.Button(
                    master, text="", font=("Helvetica", 24), width=5, height=2,
                    command=lambda i=i, j=j: self.player_move(i, j)
                )
                button.grid(row=i, column=j)
                self.buttons[(i, j)] = button

        # Randomly decide who starts the game
        self.current_turn = random.choice(["player", "computer"])
        if self.current_turn == "computer":
            # Delay slightly before computer moves first
            self.master.after(500, self.computer_move)

        # Save a screenshot of the empty board at start
        self.master.after(500, self.save_screenshot)

        # Save Q-table when window is closed
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Helper to convert the board to a string for Q-table indexing ---
    def get_state_string(self):
        """Return a string representation of the board ('-' for empty)."""
        state = ""
        for row in self.board:
            for cell in row:
                state += cell if cell is not None else '-'
        return state

    # --- Player move handling ---
    def player_move(self, i, j):
        # Ignore invalid moves (already filled, game over, or not player’s turn)
        if self.game_over or self.board[i][j] is not None or self.current_turn != "player":
            return

        # Mark the player’s move
        self.board[i][j] = "O"
        self.buttons[(i, j)].config(text="O", state="disabled")
        self.save_screenshot()  # Save screenshot after player move

        # Check for game-ending conditions
        if self.check_winner("O"):
            self.game_over = True
            self.update_q_table(-1)  # Computer penalized for losing
            self.show_result("You win!")
            return
        elif self.is_draw():
            self.game_over = True
            self.update_q_table(0)   # Draw = neutral reward
            self.show_result("Draw!")
            return

        # Switch to computer turn
        self.current_turn = "computer"
        self.master.after(500, self.computer_move)

    # --- Computer move logic (using Q-learning) ---
    def computer_move(self):
        """Make a move for the computer using the Q-learning policy."""
        if self.game_over:
            return

        state = self.get_state_string()

        # List of available cells (actions)
        free_cells = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] is None]
        if not free_cells:
            return

        # Ensure the state exists in Q-table; initialize all unseen actions with 0.0
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in free_cells}
        else:
            for action in free_cells:
                if action not in self.q_table[state]:
                    self.q_table[state][action] = 0.0

        # Epsilon-greedy policy: explore (random move) or exploit (best known move)
        if random.random() < self.epsilon:
            action = random.choice(free_cells)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [act for act, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)

        # Record state and chosen action for later update
        self.game_history.append((state, action))

        # Execute the move (computer plays "X")
        i, j = action
        self.board[i][j] = "X"
        self.buttons[(i, j)].config(text="X", state="disabled")
        self.save_screenshot()

        # Check if the computer wins or the game ends
        if self.check_winner("X"):
            self.game_over = True
            self.update_q_table(1)  # Reward +1 for winning
            self.show_result("Computer wins!")
            return
        elif self.is_draw():
            self.game_over = True
            self.update_q_table(0)
            self.show_result("Draw!")
            return

        # Switch turn to player
        self.current_turn = "player"

    # --- Win condition checking ---
    def check_winner(self, mark):
        # Check rows, columns, and diagonals for three identical marks
        for i in range(3):
            if all(self.board[i][j] == mark for j in range(3)):
                return True
            if all(self.board[j][i] == mark for j in range(3)):
                return True
        if all(self.board[i][i] == mark for i in range(3)):
            return True
        if all(self.board[i][2 - i] == mark for i in range(3)):
            return True
        return False

    # --- Draw checking ---
    def is_draw(self):
        return all(self.board[i][j] is not None for i in range(3) for j in range(3))

    # --- Q-learning table update ---
    def update_q_table(self, reward):
        """
        Update the Q-table after a game.
        Q(s,a) ‹ Q(s,a) + ? * (reward - Q(s,a))
        """
        for state, action in self.game_history:
            if state in self.q_table and action in self.q_table[state]:
                current_q = self.q_table[state][action]
                self.q_table[state][action] = current_q + self.alpha * (reward - current_q)
        self.game_history = []  # Clear history for next game

    # --- Game end message and restart handling ---
    def show_result(self, message):
        # Ask player whether to restart
        answer = messagebox.askyesno("Game Over", f"{message}\nDo you want to restart?")
        if answer:
            self.restart_game()
        else:
            for btn in self.buttons.values():
                btn.config(state="disabled")

    # --- Reset the game for a new round ---
    def restart_game(self):
        # Clear board state and reset buttons
        for i in range(3):
            for j in range(3):
                self.board[i][j] = None
                self.buttons[(i, j)].config(text="", state="normal")
        self.game_over = False
        self.game_history = []

        # Increment counters for screenshots
        self.game_number += 1
        self.step_number = 0

        # Randomly select who starts next game
        self.current_turn = random.choice(["player", "computer"])
        if self.current_turn == "computer":
            self.master.after(500, self.computer_move)

        # Save screenshot of the new empty board
        self.master.after(500, self.save_screenshot)

    # --- Q-table file operations ---
    def load_q_table(self):
        """Load Q-table from file, or return empty dict if not found."""
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, "rb") as f:
                    q_table = pickle.load(f)
                    print("Q-table loaded.")
                    return q_table
            except Exception as e:
                print("Error loading Q-table:", e)
        print("Starting with an empty Q-table.")
        return {}

    def save_q_table(self):
        """Save Q-table to disk for persistence between runs."""
        try:
            with open(self.q_table_file, "wb") as f:
                pickle.dump(self.q_table, f)
                print("Q-table saved.")
        except Exception as e:
            print("Error saving Q-table:", e)

    # --- Screenshot capturing ---
    def save_screenshot(self):
        """Take and save a screenshot of the game window."""
        self.master.update_idletasks()  # Refresh UI before capture
        x = self.master.winfo_rootx()
        y = self.master.winfo_rooty()
        w = self.master.winfo_width()
        h = self.master.winfo_height()
        bbox = (x, y, x + w, y + h)
        filename = f"screenshot_{self.game_number}_{self.step_number}.png"
        try:
            img = ImageGrab.grab(bbox)
            img.save(filename)
            print(f"Saved screenshot: {filename}")
        except Exception as e:
            print("Error capturing screenshot:", e)
        self.step_number += 1  # Increment step counter

    # --- When closing the game window ---
    def on_closing(self):
        """Save Q-table and close the app safely."""
        self.save_q_table()
        self.master.destroy()


# --- Run the GUI application ---
root = tk.Tk()
game = TicTacToe(root)
root.mainloop()
