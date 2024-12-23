import logging
from pprint import pprint
from typing import Optional

import chess

from utils.chess_stuff import ChessboardDifference, ChessboardDifferenceType
from utils.logger import create_logger

logger = create_logger(name=__name__, level=logging.DEBUG)


class ChessbotMoveHeuristics:
    def __init__(self, board: chess.Board):
        self._board = board

        logger.debug("ChessbotMoveHeuristics created")

    def try_update_with_move(self, differences: list[ChessboardDifference]) -> Optional[
        chess.Move]:
        """
        If the differences represent a move, update the board with the move.

        :param differences: A list of differences.
        :return: The move if the differences represent a move, otherwise None.
        """
        logger.debug("Trying to update as move")

        # Single move must be two differences
        if len(differences) != 2:
            logger.debug(f"Expected 2 differences to update as move, got "
                         f"{len(differences)}")
            return None

        try:
            # Must be one add and one remove, throws IndexError otherwise
            removal: ChessboardDifference = \
                list(filter(lambda x: x.type == ChessboardDifferenceType.REMOVE,
                            differences))[0]
            addition: ChessboardDifference = \
                list(filter(lambda x: x.type == ChessboardDifferenceType.ADD,
                            differences))[0]

            # Must be the same piece
            if removal.piece != addition.piece:
                logger.debug("Expected same piece to update as move")
                return None

            # TODO: If promotion bail, other function should handle it
            # Must be a legal move
            move = self._board.find_move(removal.square, addition.square)
            logger.debug(f"Found move {move}")
            self._board.push(move)

            return move
        except IndexError:
            logger.debug("Expected one add and one remove to update as move")
        except chess.IllegalMoveError:
            logger.debug("Expected legal move to update as move")

        return None

    def try_update_with_capture(self, differences: list[ChessboardDifference]) -> \
            Optional[
                chess.Move]:
        """
        If the differences represent a capture, update the board with the capture.

        :param differences: A list of differences.
        :return: The move if the differences represent a capture, otherwise None.
        """
        logger.debug("Trying to update as capture")

        # Single capture must be three differences
        if len(differences) != 3:
            logger.debug(f"Expected 3 differences to update as capture, got "
                         f"{len(differences)}")
            return None

        try:
            # Must be two removes and one add, throws IndexError otherwise
            removals = list(filter(lambda x: x.type == ChessboardDifferenceType.REMOVE,
                                   differences))

            addition = list(filter(lambda x: x.type == ChessboardDifferenceType.ADD,
                                   differences))[0]
            removal_captured = \
                list(filter(lambda x: x.square == addition.square, removals))[0]
            removal_capturing = \
                list(filter(lambda x: x.square != addition.square, removals))[0]

            # The addition and the capturing piece must be the same
            if addition.piece != removal_capturing.piece:
                logger.debug("Expected same piece to update as capture")
                return None

            # The captured piece must be a different color
            if addition.piece.color == removal_captured.piece.color:
                logger.debug("Expected different color to update as capture")
                return None

            # TODO: If promotion bail, other function should handle it
            # Must be a legal move
            move = self._board.find_move(removal_capturing.square, addition.square)
            logger.debug(f"Found move {move}")
            self._board.push(move)

            return move
        except IndexError:
            logger.debug("Expected two removes and one add to update as capture")
        except chess.IllegalMoveError:
            logger.debug("Expected legal move to update as capture")

        return None

    def try_update_board(self, differences: list[ChessboardDifference]):
        """
        Try to update the board with the given differences.

        :param differences: A list of differences.
        """
        if len(differences) == 0:
            return
        pprint(differences)
        if self.try_update_with_move(differences) is not None:
            return
        if self.try_update_with_capture(differences) is not None:
            return
        # TODO: Handle promotion, castling, en passant
