"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Penalize moves which are closer to the corners of the board.

    Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # if it is earlier in the game, it is not as bad to be in a corner
    gameState = 1

    if len(game.get_blank_spaces()) < game.width * game.height / 4.:
        gameState = 4

    myMoves = game.get_legal_moves(player)
    selfInCorner = getNumberMovesInCorners(game, myMoves)
    theirMoves = game.get_legal_moves(game.get_opponent(player))
    themInCorner = getNumberMovesInCorners(game, theirMoves)

    return float(len(myMoves) - (gameState * len(selfInCorner))
                 - len(theirMoves) + (gameState * len(themInCorner)))

def getNumberMovesInCorners(game, moves):
    """ Returns the number of moves which are "ours" within the game that are
    in a corner

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    moves : list of (int,int)
        List of all moves in the game which are to be considered "ours"

    Returns
    -------
    int
        number of moves which are in a corner
    """

    # Four corners
    corners = [(0, 0),
               (0, (game.width - 1)),
               ((game.height - 1), 0),
               ((game.height - 1), (game.width - 1))]

    movesinCorners = [move for move in moves if move in corners]

    return movesinCorners

def custom_score_2(game, player):
    """Penalize moves which are closer to the edge of the board.

    Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    myMoves = game.get_legal_moves(player)
    theirMoves = game.get_legal_moves(game.get_opponent(player))

    selfAgainstWall = getNumberMovesAgainstWall(game, myMoves)
    themAgainstWall = getNumberMovesAgainstWall(game, theirMoves)

    return float(len(myMoves) - len(selfAgainstWall)
                 - len(theirMoves) + len(themAgainstWall))

def getNumberMovesAgainstWall(game, moves):
    """ Returns the number of moves which are "ours" within the game that are
    against a wall

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    moves : list of (int,int)
        List of all moves in the game which are to be considered "ours"

    Returns
    -------
    int
        number of moves which are against a wall
    """
    movesAgainstWall = [move for move in moves if move[0] == 0
                       or move[0] == (game.height - 1)
                       or move[1] == 0
                       or move[1] == (game.width - 1)]
    return movesAgainstWall

def custom_score_3(game, player):
    """Run toward the opponent.

    Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    theirLocation = game.get_player_location(game.get_opponent(player))
    myLocation = game.get_player_location(player)

    if myLocation == None or theirLocation == None:
        return 0.

    return float(-abs(sum(theirLocation) - sum(myLocation)))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def terminal_test(self, game, depth):
        """Determine if the game is in a terminal state

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        bool
            True if the game is in a terminal state (i.e. no more legal moves
            or at a depth of 0 or less).  False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth <= 0 or len(game.get_legal_moves()) == 0:
            return True
        else:
            return False


    def min_value(self, game, depth):
        """Get the minimum score of all potential moves at the depth of the
        game which is passed into the function

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        int
            Minimum score of all potential moves at the depth of the
            game which is passed into the function
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game, depth):
            return self.score(game, self)
        score = float("inf")
        for move in game.get_legal_moves():
            score = min(score, self.max_value(game.forecast_move(move), depth - 1))
        return score


    def max_value(self, game, depth):
        """Get the maximum score of all potential moves at the depth of the
        game which is passed into the function

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        int
            Maximum score of all potential moves at the depth of the
            game which is passed into the function
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game, depth):
            return self.score(game, self)
        score = float("-inf")
        for move in game.get_legal_moves():
            score = max(score, self.min_value(game.forecast_move(move), depth - 1))
        return score

    def minimax(self, game, depth):
        """Iteratively search the game tree using the minimax algorithm to find the
        optimal move at the depth passed in

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            Location of the best move on the game board at the current depth.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        potentialMoves = game.get_legal_moves()
        bestMove = None
        bestScore = float("-inf")

        for move in potentialMoves:
            current = self.min_value(game.forecast_move(move), depth-1)
            if current > bestScore:
                bestScore = current
                bestMove = move

        return bestMove

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        if len(game.get_legal_moves()) == 0:
            return (-1, -1)

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            iterative_depth = 1
            while True:
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                best_move = self.alphabeta(game, iterative_depth)
                iterative_depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        return best_move

    def terminal_test(self, game, depth):
        """Determine if the game is in a terminal state

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        bool
            True if the game is in a terminal state (i.e. no more legal moves
            or at a depth of 0 or less).  False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth <= 0 or len(game.get_legal_moves()) == 0:
            return True
        else:
            return False


    def min_value(self, game, depth, alpha, beta):
        """Get the minimum score of all potential moves at the depth of the
        game which is passed into the function

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        int
            Minimum score of all potential moves at the depth of the
            game which is passed into the function
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game, depth):
            return self.score(game, self)
        score = float("inf")
        for move in game.get_legal_moves():
            score = min(score, self.max_value(game.forecast_move(move), depth - 1, alpha, beta))
            if score <= alpha:
                return score
            beta = min(beta, score)
        return score


    def max_value(self, game, depth, alpha, beta):
        """Get the maximum score of all potential moves at the depth of the
        game which is passed into the function

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        int
            Maximum score of all potential moves at the depth of the
            game which is passed into the function
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game, depth):
            return self.score(game, self)
        score = float("-inf")
        for move in game.get_legal_moves():
            score = max(score, self.min_value(game.forecast_move(move), depth - 1, alpha, beta))
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return score

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        potentialMoves = game.get_legal_moves()
        bestMove = None
        alpha = float("-inf")

        for move in potentialMoves:
            current = self.min_value(game.forecast_move(move), depth-1, alpha, beta)
            if current > alpha:
                alpha = current
                bestMove = move

        return bestMove