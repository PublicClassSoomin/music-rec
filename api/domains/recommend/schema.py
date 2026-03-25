from pydantic import BaseModel

class RecommendRequest(BaseModel):
    song_id:   str
    algorithm: str = "default"
    top_k:     int = 10

class UserRecommendRequest(BaseModel):
    username:  str
    algorithm: str = "default"
    top_k:     int = 10

class SearchRequest(BaseModel):
    query:     str
    algorithm: str = "default"
    top_k:     int = 10