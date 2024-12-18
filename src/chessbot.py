import logging

import chess
import chess.svg
import numpy as np

from cv.board import GetChessboardOnlyResultType, get_chessboard_only
from cv.pieces import get_piece_matrix
from utils.chess_stuff import board_sync_from_chessboard_arrangement
from utils.cv2_stuff import svg_to_numpy, write_text_tl
from utils.logger import create_logger

logger = create_logger(name=__name__, level=logging.DEBUG)


class Chessbot:
    def __init__(self):
        self.board = chess.Board()

        self.camera_preview = None
        self.chessboard_preview = None

        logger.debug("Chessbot created")

    def update(self, frame: np.ndarray) -> None:
        """
        Update the chessbot with a new frame.

        :param frame: The frame to update the chessbot with, typically from a camera.
        """
        self.camera_preview = frame.copy()
        result = get_chessboard_only(frame)
        cb_only = None
        if result.result_type == GetChessboardOnlyResultType.CHESSBOARD_FOUND:
            cb_only = result.chessboard
            self.camera_preview = result.chessboard.copy()
        elif result.result_type == GetChessboardOnlyResultType.NO_CHESSBOARD_FOUND:
            write_text_tl(self.camera_preview, "No chessboard found")
        elif result.result_type == GetChessboardOnlyResultType.NOT_QUADRILATERAL:
            write_text_tl(self.camera_preview, "Chessboard not quadrilateral")
        elif result.result_type == GetChessboardOnlyResultType.NOT_RECTANGULAR_ENOUGH:
            write_text_tl(self.camera_preview, "Chessboard not rectangular enough")

        unknown_squares = None

        if cb_only is not None:
            result = get_piece_matrix(cb_only, return_annotations=True)[0]
            self.camera_preview = result.annotation
            write_text_tl(self.camera_preview, f"{result.confidence:.4f}")
            unknown_squares = board_sync_from_chessboard_arrangement(self.board,
                                                                     result.pieces)
            # print(result.pieces)
            # print(str(board) == str(result.pieces))
        else:
            self.board.clear()

        self.chessboard_preview = svg_to_numpy(
            chess.svg.board(self.board, squares=unknown_squares, size=512))

    def get_camera_preview(self) -> np.ndarray:
        """
        Get the camera preview image.

        :return: The camera preview image.
        """
        return self.camera_preview

    def get_chessboard_preview(self) -> np.ndarray:
        """
        Get the chessboard preview image.

        :return: The chessboard preview image.
        """
        return self.chessboard_preview
