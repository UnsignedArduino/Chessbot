import logging
from dataclasses import dataclass
from enum import Enum

import chess
import chess.svg

from utils.logger import create_logger

logger = create_logger(name=__name__, level=logging.DEBUG)


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
        for col_index, piece in enumerate(row.split(" ")):
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
    piece: chess.Piece


def find_chessboard_differences(old: str, new: str) -> list[ChessboardDifference]:
    """
    Given two string representations of a chess board, find the differences between them.

    :param old: Old board arrangement.
    :param new: New board arrangement.
    :return: A list of differences.
    """
    differences = []
    old_rows = old.strip().split('\n')
    new_rows = new.strip().split('\n')
    for row_index, (old_row, new_row) in enumerate(zip(old_rows, new_rows)):
        for col_index, (old_piece, new_piece) in enumerate(
                zip(old_row.split(" "), new_row.split(" "))):
            if old_piece != new_piece:
                square = chess.square(col_index, 7 - row_index)
                if old_piece == ".":
                    differences.append(
                        ChessboardDifference(type=ChessboardDifferenceType.ADD,
                                             square=square,
                                             piece=chess.Piece.from_symbol(
                                                 new_piece)))
                elif new_piece == ".":
                    differences.append(
                        ChessboardDifference(type=ChessboardDifferenceType.REMOVE,
                                             square=square,
                                             piece=chess.Piece.from_symbol(
                                                 old_piece)))
                else:
                    differences.append(
                        ChessboardDifference(type=ChessboardDifferenceType.REMOVE,
                                             square=square,
                                             piece=chess.Piece.from_symbol(
                                                 old_piece)))
                    differences.append(
                        ChessboardDifference(type=ChessboardDifferenceType.ADD,
                                             square=square,
                                             piece=chess.Piece.from_symbol(
                                                 new_piece)))
    return differences
