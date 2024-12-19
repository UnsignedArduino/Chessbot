from collections import namedtuple
from dataclasses import dataclass
from enum import Enum

import chess
import chess.svg


def board_sync_from_chessboard_arrangement(board: chess.Board,
                                           arrangement: str) -> chess.SquareSet:
    """
    Given a string representation of a chess board, reset the board to match the
    arrangement.

    :param board: The board to set.
    :param arrangement: A string representation of the board. NOT a FEN or PGN.
    :return: A set of squares that are unknown.
    """
    ss = chess.SquareSet()
    board.clear()
    rows = arrangement.strip().split('\n')
    for row_index, row in enumerate(rows):
        for col_index, piece in enumerate(row.split()):
            if piece not in (".", "?"):
                square = chess.square(col_index, 7 - row_index)
                board.set_piece_at(square, chess.Piece.from_symbol(piece))
            else:
                if piece == "?":
                    ss.add(chess.square(col_index, 7 - row_index))
                board.remove_piece_at(chess.square(col_index, 7 - row_index))
    return ss


class ChessboardDifferenceType(Enum):
    ADD = "ADD"
    REMOVE = "REMOVE"


@dataclass
class ChessboardDifference:
    type: ChessboardDifferenceType
    square: chess.Square
    piece: chess.PieceType


def find_chessboard_differences(old: str, new: str) -> list[ChessboardDifference]:
    pass
