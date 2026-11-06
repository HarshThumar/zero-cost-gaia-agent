from agent.tools.code import chess_best_move, chess_image_to_fen, run_python
from agent.tools.files import describe_image, read_file, transcribe_audio
from agent.tools.video import youtube_frames, youtube_transcript
from agent.tools.web import fetch_page, find_in_page, web_search, wikipedia_page, wikipedia_search

ALL_TOOLS = [
    web_search, fetch_page, find_in_page, wikipedia_search, wikipedia_page,
    read_file, describe_image, transcribe_audio,
    youtube_transcript, youtube_frames,
    run_python, chess_image_to_fen, chess_best_move,
]
