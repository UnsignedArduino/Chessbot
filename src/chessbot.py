import logging
from enum import Enum

import chess
import chess.pgn
import chess.svg
import numpy as np

from chessbot_move_heuristics import ChessbotMoveHeuristics
from cv.board import GetChessboardOnlyResultType, get_chessboard_only
from cv.pieces import get_piece_matrix
from utils.chess_stuff import board_sync_from_chessboard_arrangement, \
    find_chessboard_differences
# from utils.chess_stuff import board_sync_from_chessboard_arrangement
from utils.cv2_stuff import svg_to_numpy, write_text
from utils.logger import create_logger

logger = create_logger(name=__name__, level=logging.DEBUG)


class ChessbotFrameUpdateResult(Enum):
    OK = "OK"
    NO_CHESSBOARD_FOUND = "NO_CHESSBOARD_FOUND"
    NOT_QUADRILATERAL = "NOT_QUADRILATERAL"
    NOT_RECTANGULAR_ENOUGH = "NOT_RECTANGULAR_ENOUGH"
    OBSTRUCTED_SQUARES = "OBSTRUCTED_SQUARES"
    NO_CHANGE = "NO_CHANGE"


class Chessbot:
    def __init__(self):
        self._board = chess.Board()
        self._move_heuristics = ChessbotMoveHeuristics(self._board)

        self._camera_preview = None
        self._chessboard_preview = None

        logger.debug("Chessbot created")

    def _get_game_pgn_preview(self) -> str:
        pgn_game = chess.pgn.Game.from_board(self._board)
        exporter = chess.pgn.StringExporter(headers=False)
        return pgn_game.accept(exporter)

    def update(self, frame: np.ndarray,
               force_board_sync: bool = False) -> ChessbotFrameUpdateResult:
        """
        Update the chessbot with a new frame.

        :param frame: The frame to update the chessbot with, typically from a camera.
        :param force_board_sync: Force a board sync from the frame, even if the frame
         does not represent a valid move. Useful for starting in the middle of a game.
        :return: The result of the update.
        """
        update_result = ChessbotFrameUpdateResult.OK
        self._camera_preview = frame.copy()

        # Use ML model to segment the board
        result = get_chessboard_only(frame)
        cb_only = None
        if result.result_type == GetChessboardOnlyResultType.CHESSBOARD_FOUND:
            cb_only = result.chessboard
            self._camera_preview = result.chessboard.copy()
        elif result.result_type == GetChessboardOnlyResultType.NO_CHESSBOARD_FOUND:
            write_text(self._camera_preview, "No chessboard found", 10, 10)
            update_result = ChessbotFrameUpdateResult.NO_CHESSBOARD_FOUND
        elif result.result_type == GetChessboardOnlyResultType.NOT_QUADRILATERAL:
            write_text(self._camera_preview, "Chessboard not quadrilateral", 10, 10)
            update_result = ChessbotFrameUpdateResult.NOT_QUADRILATERAL
        elif result.result_type == GetChessboardOnlyResultType.NOT_RECTANGULAR_ENOUGH:
            write_text(self._camera_preview, "Chessboard not rectangular enough", 10,
                       10)
            update_result = ChessbotFrameUpdateResult.NOT_RECTANGULAR_ENOUGH

        # Use ML model to classify each square and get a chessboard arrangement
        if cb_only is not None:
            results = get_piece_matrix(cb_only, return_annotations=True)
            # for i, r in enumerate(results):
            #     print(
            #         f"Chessboard detection result {i}: ({r.confidence})\n{r.pieces}\n")
            if force_board_sync:
                logger.info("Forcing board sync from chessboard arrangement")
                board_sync_from_chessboard_arrangement(self._board, results[0].pieces)
            else:
                for i, result in enumerate(results):
                    logger.debug(f"Trying update with possible result {i}")
                    self._camera_preview = result.annotation
                    write_text(self._camera_preview, f"{result.confidence:.4f}", 10, 10)
                    try:
                        diffs = find_chessboard_differences(str(self._board),
                                                            result.pieces)
                        if len(diffs) == 0:
                            logger.debug("No differences in board found, skipping")
                            update_result = ChessbotFrameUpdateResult.NO_CHANGE
                            break
                        m = self._move_heuristics.try_update_board(diffs)
                    except ValueError:
                        logger.debug(
                            "Unknown square, assuming obstructed/bad camera angle")
                    else:
                        if m is not None:
                            break
                        else:
                            logger.debug(
                                "Could not find legal move, trying next result")
                else:
                    if update_result != ChessbotFrameUpdateResult.NO_CHANGE:
                        logger.warning(
                            "No valid chessboard arrangement found for possible moves")
                        update_result = ChessbotFrameUpdateResult.OBSTRUCTED_SQUARES

        pgn = self._get_game_pgn_preview()
        print(pgn)
        self._chessboard_preview = svg_to_numpy(chess.svg.board(self._board, size=512))

        return update_result

    @property
    def camera_preview(self) -> np.ndarray:
        """
        Get the camera preview image.

        :return: The camera preview image.
        """
        return self._camera_preview

    @property
    def chessboard_preview(self) -> np.ndarray:
        """
        Get the chessboard preview image.

        :return: The chessboard preview image.
        """
        return self._chessboard_preview
