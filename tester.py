import requests
import json
import time

# FastAPI 서버 주소
BASE_URL = "http://127.0.0.1:8000"

def print_log(step, message):
    print(f"\n[{step}] {'='*40}\n-> {message}")

def print_results(recommendations):
    if not recommendations:
        print("추천 결과가 없습니다.")
        return

    for i, song in enumerate(recommendations, 1):
        print(f"    {i}. [{song['genre']}] {song['title']} - {song['artist']} (Score: {song['score']})")

def run_tests():
    print("Vibe_Sync 알고리즘 성능 및 흐름 테스트 시작\n")

    print_log("STEP 1", "서버에서 곡 리스트를 가져와 기준 곡을 선택합니다.")

    try:
        res = requests.get(f"{BASE_URL}/api/songs?limit=10")
        res.raise_for_status()
        songs = res.json()
        base_song = songs[0]
        print(f"기준 곡 선택 완료 : {base_song['title']} : {base_song['artist']}")
    
    except Exception as e:
        print(f'연결 실패 {e}')
        return
    
    print_log("STEP 2", "나만의 믹싱 OFF (원본 기준 곡 벡터만 사용한 추천)")
    payload_default = {
        "song_id": base_song['song_id'],
        "algorithm": "vibe_sync",
        "top_k": 10,
        "params": {
            "manual_mode": False # 조절바 무시
        }
    }

    start_time = time.time()
    res_default = requests.post(f"{BASE_URL}/api/recommend", json=payload_default)
    print(f"응답 속도: {time.time() - start_time:.4f}초")
    print_results(res_default.json().get("recommendations", []))

    # 3. 조절바 개입 테스트 (안전 모드)
    print_log("STEP 3", "조절바 ON - 안전 모드 (비슷한 분위기와 장르 위주)")
    payload_safe = {
        "song_id": base_song['song_id'],
        "algorithm": "vibe_sync",
        "top_k": 10,
        "params": {
            "manual_mode": True,
            "artist": 50,
            "mood": 50,
            "genre": 50,
            "similarity": 20, # 음색 일치
            "emotion": 50,
            "adventure": 10   # 안전 (같은 장르 위주)
        }
    }
    res_safe = requests.post(f"{BASE_URL}/api/recommend", json=payload_safe)
    print_results(res_safe.json().get("recommendations", []))

    # 4. 조절바 개입 테스트 (극한의 모험 모드)
    print_log("STEP 4", "조절바 ON - 🔥극한 모드 (장르 파괴, 강렬한 에너지, 템포 흔들기)")
    payload_extreme = {
        "song_id": base_song['song_id'],
        "algorithm": "vibe_sync",
        "top_k": 10,
        "params": {
            "manual_mode": True,
            "artist": 80,      # 새로운 가수
            "mood": 100,       # 엄청난 강렬함 (에너지 폭발)
            "genre": 100,      # 장르 파괴
            "similarity": 90,  # 느낌만 일치 (노이즈 최대)
            "emotion": 90,     # 지금 기분 (BPM 변형)
            "adventure": 95    # 극한의 개척 (완전 다른 장르 섞임)
        }
    }
    res_extreme = requests.post(f"{BASE_URL}/api/recommend", json=payload_extreme)
    print_results(res_extreme.json().get("recommendations", []))
    
    print("\n🎉 모든 테스트가 순조롭게 완료되었습니다!")

if __name__ == "__main__":
    run_tests()
