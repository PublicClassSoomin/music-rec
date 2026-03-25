from pydantic import BaseModel

class InteractionRequest(BaseModel):
    username:    str
    song_id:     str
    action:      str        # "play" | "like" | "skip" | "unlike"
    play_seconds: int = 0