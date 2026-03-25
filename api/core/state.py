import pandas as pd
from typing import Dict, Any

class AppState:
    def __init__(self):
        self.song_df: pd.DataFrame | None = None
        self.recommenders: Dict[str, Any] = {}

# 싱글톤 인스턴스로 생성하여 앱 전체에서 공유
state = AppState()