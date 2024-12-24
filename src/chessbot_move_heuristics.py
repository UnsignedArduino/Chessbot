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

    def try_update_with_castle(self, differences: list[ChessboardDifference]) -> \
            Optional[
                chess.Move]:
        """
        If the differences represent a castle, update the board with the castle.

        :param differences: A list of differences.
        :return: The move if the differences represent a castle, otherwise None.
        """
        logger.debug("Trying to update as castle")

        # Single castle must be four differences
        if len(differences) != 4:
            logger.debug(f"Expected 4 differences to update as castle, got "
                         f"{len(differences)}")
            return None

        try:
            # Must be two removes and two adds, throws IndexError otherwise
            king_removal = list(filter(lambda
                                           x: x.type == ChessboardDifferenceType.REMOVE and x.piece.piece_type == chess.KING,
                                       differences))[0]
            king_addition = list(filter(lambda
                                            x: x.type == ChessboardDifferenceType.ADD and x.piece.piece_type == chess.KING,
                                        differences))[0]
            rook_removal = list(filter(lambda
                                           x: x.type == ChessboardDifferenceType.REMOVE and x.piece.piece_type == chess.ROOK,
                                       differences))[0]
            rook_addition = list(filter(lambda
                                            x: x.type == ChessboardDifferenceType.ADD and x.piece.piece_type == chess.ROOK,
                                        differences))[0]

            # All differences must be the same color
            def all_colors_same(ds: list[ChessboardDifference]) -> bool:
                return all(x.piece.color == ds[0].piece.color for x in ds)

            if not all_colors_same(
                    [king_removal, king_addition, rook_removal, rook_addition]):
                logger.debug("Expected same color to update as castle")
                return None

            # All differences must be on the same rank
            def all_ranks_same(ds: list[ChessboardDifference]) -> bool:
                return all(
                    chess.square_rank(x.square) == chess.square_rank(ds[0].square) for x
                    in ds)

            if not all_ranks_same(
                    [king_removal, king_addition, rook_removal, rook_addition]):
                logger.debug("Expected same rank to update as castle")
                return None

            # Must be a legal move
            move = self._board.find_move(king_removal.square, king_addition.square)
            logger.debug(f"Found move {move}")
            self._board.push(move)

            return move
        except IndexError:
            logger.debug("Expected two removes and two adds to update as castle")
        except chess.IllegalMoveError:
            logger.debug("Expected legal move to update as castle")

        return None

    def try_update_with_promotion(self, differences: list[ChessboardDifference]) -> \
            Optional[
                chess.Move]:
        """
        If the differences represent a promotion, update the board with the promotion.

        :param differences: A list of differences.
        :return: The move if the differences represent a promotion, otherwise None.
        """
        logger.debug("Trying to update as promotion")

        # Single promotion must be two differences
        if len(differences) != 2:
            logger.debug(f"Expected 2 differences to update as promotion, got "
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

            # Removed piece must be a pawn
            if removal.piece.piece_type != chess.PAWN:
                logger.debug("Expected pawn to update as promotion")
                return None

            # Added piece must be a promotable piece
            if addition.piece.piece_type not in [chess.QUEEN, chess.ROOK, chess.BISHOP,
                                                 chess.KNIGHT]:
                logger.debug("Expected promotable piece to update as promotion")
                return None

            # Both must be the same color
            if removal.piece.color != addition.piece.color:
                logger.debug("Expected same color to update as promotion")
                return None

            # If the pieces are white, the addition must be on the 8th rank
            if removal.piece.color == chess.WHITE and chess.square_rank(
                    addition.square) != 7:
                logger.debug("Expected 8th rank to update as promotion")
                return None
            # and the removal must be on the 7th rank
            if removal.piece.color == chess.WHITE and chess.square_rank(
                    removal.square) != 6:
                logger.debug("Expected 7th rank to update as promotion")
                return None

            # If the pieces are black, the addition must be on the 1st rank
            if removal.piece.color == chess.BLACK and chess.square_rank(
                    addition.square) != 0:
                logger.debug("Expected 1st rank to update as promotion")
                return None
            # and the removal must be on the 2nd rank
            if removal.piece.color == chess.BLACK and chess.square_rank(
                    removal.square) != 1:
                logger.debug("Expected 2nd rank to update as promotion")
                return None

            # Since this promotion isn't also a capture, the addition and removal must
            # be on the same file
            if chess.square_file(removal.square) != chess.square_file(addition.square):
                logger.debug("Expected same file to update as promotion")
                return None

            # Must be a legal move
            move = self._board.find_move(removal.square, addition.square,
                                         addition.piece.piece_type)
            logger.debug(f"Found move {move}")
            self._board.push(move)

            return move
        except IndexError:
            logger.debug("Expected one add and one remove to update as promotion")
        except chess.IllegalMoveError:
            logger.debug("Expected legal move to update as promotion")

        return None

    def try_update_board(self, differences: list[ChessboardDifference]) -> Optional[
        chess.Move]:
        """
        Try to update the board with the given differences.

        :param differences: A list of differences.
        :return: The move if the board was updated, otherwise None.
        """
        if len(differences) == 0:
            return
        pprint(differences)
        if (move := self.try_update_with_move(differences)) is not None:
            return move
        if (move := self.try_update_with_capture(differences)) is not None:
            return move
        if (move := self.try_update_with_castle(differences)) is not None:
            return move
        if (move := self.try_update_with_promotion(differences)) is not None:
            return move
        # TODO: Handle capturing promotion, en passant
        return None
