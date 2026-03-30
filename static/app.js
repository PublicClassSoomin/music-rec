const API = "";
let currentSong   = null;
let currentAlgo   = () => document.getElementById("algo-select").value;
let authUser      = null;
let authToken     = null;
let currentUser   = () => authUser?.username || "guest";
let likedSongs    = new Set();
let songCache     = new Map(); // song_id -> full song object

/** 추천(/recommend)용. 검색은 검색창 입력 + hybridSearchPayload()만 사용 */
let hybridOptions = {
  mode: "advanced",
  weights: { content: 0.55, cooc: 0.45 },
  threshold: 0.0,
  use_llm_search: false,
};

const EVAL_SETTINGS_STORAGE_KEY = "music_rec_eval_settings_v1";

/** 평가 옵션(검색어/기준곡/범위). 하이브리드 가중치는 hybridOptions와 별도 */
let evalSettings = {
  eval_mode: "search",
  search_eval_query: "",
  eval_seed_song_id: "",
  eval_only_current_user: true,
};

function loadEvalSettingsFromStorage() {
  try {
    const raw = localStorage.getItem(EVAL_SETTINGS_STORAGE_KEY);
    if (!raw) return;
    const o = JSON.parse(raw);
    if (o.eval_mode === "search" || o.eval_mode === "recommend") {
      evalSettings.eval_mode = o.eval_mode;
    }
    if (typeof o.search_eval_query === "string") {
      evalSettings.search_eval_query = o.search_eval_query;
    }
    if (typeof o.eval_seed_song_id === "string") {
      evalSettings.eval_seed_song_id = o.eval_seed_song_id;
    }
    if (typeof o.eval_only_current_user === "boolean") {
      evalSettings.eval_only_current_user = o.eval_only_current_user;
    }
  } catch (_) {
    /* ignore */
  }
}

function persistEvalSettings() {
  try {
    localStorage.setItem(EVAL_SETTINGS_STORAGE_KEY, JSON.stringify(evalSettings));
  } catch (_) {
    /* ignore */
  }
}

/** 홈 상단 그리드: 곡/유저 추천 시 표시 개수 */
const HOME_RECO_TOP_K = 16;
/** 검색 API top_k (유사도 순, 아래 검색 결과 블록에만 표시) */
const SEARCH_TOP_K = 24;
/** 장르 클릭 시 상단 미리보기 개수 */
const GENRE_PREVIEW_K = 6;

/** score 필드가 있으면 내림차순(유사도·점수 높은 순). 없으면 원래 순서 유지 */
function sortByScoreDesc(songs) {
  if (!songs?.length) return [];
  const any = songs.some(s => s && s.score != null && !Number.isNaN(Number(s.score)));
  if (!any) return [...songs];
  return [...songs].sort((a, b) => Number(b.score) - Number(a.score));
}

/** 검색 시 상단 #song-grid는 이전 추천·장르 미리보기라 아래와 겹쳐 보일 수 있어 안내 */
function setHomePreviewSearchHint() {
  document.getElementById("song-grid").innerHTML =
    "<p class=\"placeholder\">이 칸은 <strong>검색과 별개</strong>로, 이전에 고른 장르 미리보기나 곡·취향 추천이 그대로 남아 있을 수 있어요. 방금 검색한 곡은 <strong>아래 검색 결과</strong>에만 유사도 높은 순으로 모여 있습니다.</p>";
}

/** POST /api/search 에만 넣을 옵션 (질의 문장은 payload.query) */
function hybridSearchPayload() {
  return {
    use_llm_search: !!hybridOptions.use_llm_search,
  };
}

let ytApiReady = false; // YouTube IFrame API 준비 여부
let ytPlayer = null; // YouTube IFrame API 인스턴스
let ytCurrentSongId = null; // 현재 재생 중인 곡 ID
let ytPlayStartAt = null; // 재생 시작 시간
let ytAccumulatedSec = 0; // 누적 재생 시간
let ytHeartbeat = null; // 하트비트 타이머, 10초마다 재생 진행률을 서버에 전송
let ytPendingLoadId = null; // 플레이어 생성 전·onReady 전에 로드할 video id
let ytEmbedLoadedId = null; // iframe에 실제 loadVideoById 된 video id
/** 사용자가 닫기로 숨김. 푸터 「내장 플레이어에서 듣기」로 다시 열면 해제 */
let embedPanelDismissed = false;
/** 축소 시 사이드바 열에 붙은 좁은 패널 */
let embedPanelMinimized = false;

// ── 초기화 ────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", async () => {
  loadEvalSettingsFromStorage();
  bindAuthEvents();
  await bootstrapAuth();
  setupNav();
  await loadAlgorithms();
  await loadGenres();
  await loadSongs();

  document.getElementById("search-btn").addEventListener("click", onSearch);
  document.getElementById("search-input").addEventListener("keydown", e => {
    if (e.key === "Enter") onSearch();
  });
  document.getElementById("btn-youtube").addEventListener("click", openYoutube);
  document.getElementById("btn-embed-close").addEventListener("click", closeEmbedPanel);
  document.getElementById("btn-embed-minimize").addEventListener("click", toggleEmbedPanelMinimize);
  document.getElementById("btn-like-player").addEventListener("click", onLikePlayer);
  document.getElementById("user-reco-btn").addEventListener("click", fetchUserRecommendations);
  document.getElementById("algo-select").addEventListener("change", () => {
    syncHybridEmbedChrome();
    if (currentSong) {
      fetchRecommendations(currentSong.song_id);
      updatePlayer(currentSong);
    } else if (authUser) {
      fetchUserRecommendations();
    }
  });

  // 하이브리드 설정 모달
  document.getElementById("hybrid-config-btn").addEventListener("click", openHybridModal);
  document.getElementById("hyb-cancel-btn").addEventListener("click", closeHybridModal);
  document.getElementById("hyb-save-btn").addEventListener("click", saveHybridOptions);
  document.getElementById("hyb-close-x").addEventListener("click", closeHybridModal);
  document.getElementById("hyb-balance").addEventListener("input", updateHybridBalanceUI);
  document.getElementById("hyb-threshold").addEventListener("input", updateHybridThresholdUI);
  document.getElementById("hyb-mode-seg").addEventListener("click", onHybridModeSegClick);

  document.querySelectorAll("[data-modal-dismiss]").forEach((el) => {
    el.addEventListener("click", () => {
      const id = el.getAttribute("data-modal-dismiss");
      if (id === "hybrid-modal") closeHybridModal();
      if (id === "eval-modal") closeEvalModal();
      if (id === "eval-settings-modal") closeEvalSettingsModal();
    });
  });

  document.getElementById("eval-settings-btn").addEventListener("click", () => {
    void openEvalSettingsModal();
  });
  document.getElementById("eval-open-btn").addEventListener("click", () => {
    void openEvalModal();
  });
  document.getElementById("eval-close-btn").addEventListener("click", closeEvalModal);
  document.getElementById("eval-close-x").addEventListener("click", closeEvalModal);
  document.getElementById("eval-run-btn").addEventListener("click", runEvaluation);
  document.getElementById("eval-edit-settings-btn").addEventListener("click", () => {
    void openEvalSettingsModal();
  });
  document.getElementById("eval-settings-save-btn").addEventListener("click", saveEvalSettingsAndClose);
  document.getElementById("eval-settings-cancel-btn").addEventListener("click", closeEvalSettingsModal);
  document.getElementById("eval-settings-close-x").addEventListener("click", closeEvalSettingsModal);
  document.querySelectorAll('input[name="eval-settings-eval-mode"]').forEach((el) => {
    el.addEventListener("change", syncEvalSettingsModePanels);
  });

  document.addEventListener("keydown", onModalEscape);

  loadYoutubeIframeAPI();
});

function bindAuthEvents() {
  document.getElementById("gate-signup-btn").addEventListener("click", onSignup);
  document.getElementById("gate-login-btn").addEventListener("click", onLogin);
  document.getElementById("logout-btn").addEventListener("click", onLogout);
}

async function bootstrapAuth() {
  const token = localStorage.getItem("auth_token");
  if (!token) {
    showAuthGate(true);
    refreshAuthUI();
    return;
  }
  authToken = token;
  const me = await apiFetch("/api/me");
  if (!me?.username) {
    localStorage.removeItem("auth_token");
    authToken = null;
    authUser = null;
    showAuthGate(true);
    refreshAuthUI();
    return;
  }
  authUser = { username: me.username, user_id: me.user_id };
  showAuthGate(false);
  refreshAuthUI();
  await loadLikedFromServer();
}

function refreshAuthUI() {
  const status = document.getElementById("auth-status");
  const gateStatus = document.getElementById("auth-gate-status");
  if (authUser) {
    status.textContent = `로그인됨: ${authUser.username}`;
    if (gateStatus) gateStatus.textContent = `로그인됨: ${authUser.username}`;
  } else {
    status.textContent = "비로그인 상태";
    if (gateStatus) gateStatus.textContent = "";
  }
  const ou = document.getElementById("eval-settings-only-current-user");
  if (ou) {
    ou.disabled = !authUser;
    if (!authUser) ou.checked = false;
  }
  if (!document.getElementById("eval-modal")?.classList.contains("hidden")) {
    updateEvalSummaryInEvalModal();
  }
  if (!document.getElementById("eval-settings-modal")?.classList.contains("hidden")) {
    loadEvalSettingsUI();
    void refreshEvalSettingsSeedSelect();
  }
}

function showAuthGate(visible) {
  document.getElementById("auth-gate").classList.toggle("hidden", !visible);
  document.getElementById("app-shell").classList.toggle("hidden", visible);
}

async function onSignup() {
  const username = document.getElementById("auth-username").value.trim();
  const password = document.getElementById("auth-password").value;
  if (!username || !password) return alert("아이디/비밀번호를 입력해주세요.");
  const res = await apiFetch("/api/signup", "POST", { username, password });
  if (!res) return alert("회원가입 실패: 중복 아이디 또는 형식 오류");
  authUser = { username: res.username, user_id: res.user_id };
  authToken = res.access_token;
  localStorage.setItem("auth_token", authToken);
  showAuthGate(false);
  refreshAuthUI();
  await loadLikedFromServer();
  alert("회원가입 완료");
}

async function onLogin() {
  const username = document.getElementById("auth-username").value.trim();
  const password = document.getElementById("auth-password").value;
  if (!username || !password) return alert("아이디/비밀번호를 입력해주세요.");
  const res = await apiFetch("/api/login", "POST", { username, password });
  if (!res) return alert("로그인 실패: 아이디/비밀번호를 확인해주세요.");
  authUser = { username: res.username, user_id: res.user_id };
  authToken = res.access_token;
  localStorage.setItem("auth_token", authToken);
  showAuthGate(false);
  refreshAuthUI();
  await loadLikedFromServer();
  fetchUserRecommendations();
}

function onLogout() {
  authUser = null;
  authToken = null;
  likedSongs.clear();
  localStorage.removeItem("auth_token");
  if (evalSettings.eval_only_current_user) {
    evalSettings.eval_only_current_user = false;
    persistEvalSettings();
  }
  showAuthGate(true);
  refreshAuthUI();
}

// ── 네비게이션 ────────────────────────────────────────────

function setupNav() {
  document.querySelectorAll(".nav-item").forEach(el => {
    el.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
      document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
      el.classList.add("active");
      document.getElementById(`view-${el.dataset.view}`).classList.add("active");
      if (el.dataset.view === "liked") renderLiked();
    });
  });
}

// ── 데이터 로드 ───────────────────────────────────────────

async function loadAlgorithms() {
  const res = await apiFetch("/api/algorithms");
  const select = document.getElementById("algo-select");
  const hint = document.getElementById("algo-hint");
  select.innerHTML = '<option value="">선택 안 됨</option>';
  const algos = res?.algorithms || [];
  algos.forEach(algo => {
    const opt = document.createElement("option");
    opt.value = algo;
    opt.textContent = algo;
    select.appendChild(opt);
  });
  if (algos.length === 0) {
    hint.textContent =
      res?.notice ||
      "등록된 알고리즘이 없습니다. 서버 로그에서 FAISS·DB를 확인하세요.";
    hint.classList.remove("hidden");
  } else {
    hint.textContent = "";
    hint.classList.add("hidden");
  }
  syncHybridEmbedChrome();
}

async function loadGenres() {
  const songs = await apiFetch("/api/songs?limit=200");
  if (!songs) return;
  const genres = [...new Set(songs.map(s => s.genre).filter(Boolean))];
  const list = document.getElementById("genre-list");
  list.innerHTML = "";
  genres.forEach(genre => {
    const el = document.createElement("div");
    el.className = "genre-item";
    el.textContent = genre;
    el.addEventListener("click", () => {
      document.querySelectorAll(".genre-item").forEach(g => g.classList.remove("active"));
      el.classList.add("active");
      loadSongs(genre);
    });
    list.appendChild(el);
  });
}

async function loadSongs(genre = null) {
  const url = genre ? `/api/songs?limit=30&genre=${encodeURIComponent(genre)}` : "/api/songs?limit=30";
  const songs = await apiFetch(url);
  if (songs) renderGrid(songs.slice(0, GENRE_PREVIEW_K), "song-grid", false);
}

// ── 추천 ──────────────────────────────────────────────────

async function fetchRecommendations(songId) {
  const algo = currentAlgo();
  if (!algo) return;

  const payload = {
    song_id:   songId,
    algorithm: algo,
    top_k:     HOME_RECO_TOP_K,
  };
  if (isHybridAlgo(algo)) payload.options = hybridOptions;

  const res = await apiFetch("/api/recommend", "POST", payload);

  if (res?.recommendations) {
    renderGrid(res.recommendations, "song-grid", true);
    document.getElementById("algo-badge").textContent = algo;
  }
}

async function fetchUserRecommendations() {
  const algo = currentAlgo();
  if (!algo) return alert("알고리즘을 먼저 선택해주세요.");
  if (!authUser) return alert("로그인 후 사용해주세요.");

  const payload = {
    username: authUser.username,
    algorithm: algo,
    top_k: HOME_RECO_TOP_K,
  };
  if (isHybridAlgo(algo)) payload.options = hybridOptions;

  const res = await apiFetch("/api/recommend/user", "POST", payload);
  if (res?.recommendations) {
    renderGrid(res.recommendations, "song-grid", true);
    document.getElementById("algo-badge").textContent = `${algo} · ${authUser.username}`;
  }
}

// ── 검색 ──────────────────────────────────────────────────

async function onSearch() {
  const query = document.getElementById("search-input").value.trim();
  if (!query) return;

  const algo = currentAlgo();
  if (!algo) {
    alert("알고리즘을 먼저 선택해주세요.");
    return;
  }

  const payload = { query, algorithm: algo, top_k: SEARCH_TOP_K };
  if (isHybridAlgo(algo)) payload.options = hybridSearchPayload();

  const res = await apiFetch("/api/search", "POST", payload);

  const block = document.getElementById("search-results-block");
  const container = document.getElementById("search-result");
  block.removeAttribute("hidden");

  if (res?.recommendations?.length) {
    setHomePreviewSearchHint();
    renderGrid(res.recommendations, "search-result", true);
  } else {
    container.innerHTML =
      "<p class='placeholder'>검색 결과가 없거나 이 알고리즘은 자연어 검색을 지원하지 않습니다. (API 오류면 콘솔·네트워크 탭을 확인해 주세요.)</p>";
  }

  block.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── 좋아요 (DB 동기화) ────────────────────────────────────

async function loadLikedFromServer() {
  if (!authToken) {
    likedSongs.clear();
    return;
  }
  const res = await apiFetch("/api/likes");
  if (!res?.likes) return;
  likedSongs.clear();
  res.likes.forEach(song => {
    if (song?.song_id) {
      likedSongs.add(song.song_id);
      songCache.set(song.song_id, song);
    }
  });
  if (currentSong) updatePlayer(currentSong);
  const likedView = document.getElementById("view-liked");
  if (likedView?.classList.contains("active")) renderLiked();
}

// ── 인터랙션 ──────────────────────────────────────────────

async function logInteraction(songId, action, playSeconds = 0) {
  await apiFetch("/api/interact", "POST", {
    username:     currentUser(),
    song_id:      songId,
    action,
    play_seconds: playSeconds,
  });
}

function onLikePlayer() {
  if (!currentSong) return;
  const sid = currentSong.song_id;
  songCache.set(sid, currentSong);
  const btn = document.getElementById("btn-like-player");
  if (likedSongs.has(sid)) {
    likedSongs.delete(sid);
    btn.textContent = "🤍";
    btn.classList.remove("liked");
    logInteraction(sid, "unlike");
  } else {
    likedSongs.add(sid);
    btn.textContent = "💚";
    btn.classList.add("liked");
    logInteraction(sid, "like");
  }
}

function setEmbedPanelMinimized(on) {
  embedPanelMinimized = !!on;
  const wrap = document.getElementById("player-embed-wrap");
  const minBtn = document.getElementById("btn-embed-minimize");
  wrap?.classList.toggle("player-embed-panel--minimized", embedPanelMinimized);
  if (minBtn) {
    minBtn.textContent = embedPanelMinimized ? "확대" : "축소";
    minBtn.setAttribute("aria-pressed", embedPanelMinimized ? "true" : "false");
    minBtn.title = embedPanelMinimized
      ? "메인 영역에서 크게 보기"
      : "사이드바 쪽 작은 패널로 보기";
  }
}

function toggleEmbedPanelMinimize() {
  setEmbedPanelMinimized(!embedPanelMinimized);
}

function closeEmbedPanel() {
  embedPanelDismissed = true;
  setEmbedPanelMinimized(false);
  document.getElementById("player-embed-wrap")?.setAttribute("hidden", "");
  try {
    ytPlayer?.pauseVideo?.();
  } catch (_) {
    /* noop */
  }
}

function syncHybridEmbedChrome() {
  const wrap = document.getElementById("player-embed-wrap");
  const btn = document.getElementById("btn-youtube");
  if (!btn) return;
  if (!isHybridAlgo(currentAlgo())) {
    embedPanelDismissed = false;
    setEmbedPanelMinimized(false);
    wrap?.setAttribute("hidden", "");
    ytPendingLoadId = null;
    ytEmbedLoadedId = null;
    try {
      ytPlayer?.pauseVideo?.();
    } catch (_) {
      /* noop */
    }
    btn.textContent = "▶ YouTube에서 듣기";
    btn.title = "";
    return;
  }
  btn.textContent = "▶ 내장 플레이어에서 듣기";
  btn.title =
    "내장 플레이어 패널을 열고 YouTube iframe에서 재생합니다 (재생 시간 기록). 패널을 닫았다면 이 버튼으로 다시 열 수 있어요.";
}

function openYoutube() {
  if (!currentSong) return;
  const algo = currentAlgo();
  if (isHybridAlgo(algo)) {
    playInsideIframe(currentSong);
  } else {
    logInteraction(currentSong.song_id, "play", 30);
    window.open(currentSong.youtube_url, "_blank");
  }
}

// ── 렌더링 ────────────────────────────────────────────────

function renderGrid(songs, containerId, showScore) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  let list = Array.isArray(songs) ? songs : [];
  if (showScore) list = sortByScoreDesc(list);

  if (!list.length) {
    container.innerHTML = "<p class='placeholder'>곡이 없습니다.</p>";
    return;
  }

  list.forEach(song => {
    if (song?.song_id) songCache.set(song.song_id, song);
    const tpl = document.getElementById("song-card-tpl").content.cloneNode(true);
    const card = tpl.querySelector(".song-card");

    tpl.querySelector(".card-thumb").src = song.thumbnail_url || "";
    tpl.querySelector(".card-thumb").alt = song.title;
    tpl.querySelector(".card-title").textContent = song.title;
    tpl.querySelector(".card-artist").textContent = song.artist || "";

    const badge = tpl.querySelector(".score-badge");
    if (showScore && song.score != null) {
      badge.textContent = `${(song.score * 100).toFixed(0)}%`;
    } else {
      badge.style.display = "none";
    }

    const likeBtn = tpl.querySelector(".btn-like");
    if (likedSongs.has(song.song_id)) {
      likeBtn.textContent = "💚";
      likeBtn.style.color = "var(--accent)";
    }

    // 재생 클릭
    const playFn = async (e) => {
      if (e.target.closest(".card-actions")) return;
      currentSong = song;
      updatePlayer(song);
      await logInteraction(song.song_id, "play");
      await fetchRecommendations(song.song_id);
    };
    card.addEventListener("click", playFn);
    tpl.querySelector(".btn-play-overlay").addEventListener("click", playFn);

    // 좋아요
    likeBtn.addEventListener("click", async e => {
      e.stopPropagation();
      const btn = e.currentTarget;
      if (likedSongs.has(song.song_id)) {
        likedSongs.delete(song.song_id);
        btn.textContent = "🤍";
        btn.style.color = "";
        await logInteraction(song.song_id, "unlike");
      } else {
        likedSongs.add(song.song_id);
        btn.textContent = "💚";
        btn.style.color = "var(--accent)";
        await logInteraction(song.song_id, "like");
      }
    });

    // 스킵
    tpl.querySelector(".btn-skip").addEventListener("click", async e => {
      e.stopPropagation();
      await logInteraction(song.song_id, "skip");
      card.style.opacity = "0.3";
    });

    container.appendChild(tpl);
  });
}

function renderLiked() {
  const liked = [...likedSongs].map(sid => {
    const cached = songCache.get(sid);
    if (cached) return cached;
    return {
      song_id: sid,
      title: sid,
      artist: "정보 없음",
      youtube_url: "",
      thumbnail_url: "",
    };
  });
  renderGrid(liked, "liked-list", false);
}

function updatePlayer(song) {
  const thumb = document.getElementById("player-thumb");
  const fallback = document.getElementById("player-thumb-fallback");
  thumb.src = song.thumbnail_url || "";
  thumb.onerror = () => fallback.classList.remove("hidden");
  if (song.thumbnail_url) fallback.classList.add("hidden");
  else fallback.classList.remove("hidden");
  document.getElementById("player-title").textContent  = song.title;
  document.getElementById("player-artist").textContent = song.artist || "";
  document.getElementById("btn-youtube").disabled = false;
  document.getElementById("btn-like-player").textContent =
    likedSongs.has(song.song_id) ? "💚" : "🤍";
  document.getElementById("btn-like-player").classList.toggle(
    "liked", likedSongs.has(song.song_id)
  );

  syncHybridEmbedChrome();
  if (song && isHybridAlgo(currentAlgo()) && !embedPanelDismissed) {
    document.getElementById("player-embed-wrap")?.removeAttribute("hidden");
    if (song.song_id !== ytEmbedLoadedId) playInsideIframe(song);
  }
}

// ── 유틸 ──────────────────────────────────────────────────

async function apiFetch(url, method = "GET", body = null) {
  try {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (authToken) {
      opts.headers.Authorization = `Bearer ${authToken}`;
    }
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(API + url, opts);
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    console.error("API 오류:", e);
    return null;
  }
}

function isHybridAlgo(algo) {
  return !!algo && algo.startsWith("hybrid");
}

function loadYoutubeIframeAPI() {
  if (window.YT && window.YT.Player) {
    ytApiReady = true;
    ensureYtPlayer();
    return;
  }
  const tag = document.createElement("script"); 
  tag.src = "https://www.youtube.com/iframe_api";
  document.head.appendChild(tag);

  window.onYouTubeIframeAPIReady = () => {
    ytApiReady = true;
    ensureYtPlayer();
  };
}

function ensureYtPlayer() {
  if (!ytApiReady || ytPlayer) return;
  const host = document.getElementById("yt-player");
  if (!host) return;
  ytPlayer = new YT.Player("yt-player", {
    height: "360",
    width: "640",
    playerVars: {
      rel: 0,
      modestbranding: 1,
      playsinline: 1,
      origin: window.location.origin,
    },
    events: {
      onStateChange: onYtStateChange,
      onReady: () => {
        if (ytPendingLoadId && ytPlayer?.loadVideoById) {
          const id = ytPendingLoadId;
          ytPlayer.loadVideoById(id);
          ytPendingLoadId = null;
          ytEmbedLoadedId = id;
        }
      },
    },
  });
}

function onYtStateChange(e) {
  if (!ytCurrentSongId) return;
  const PS = window.YT?.PlayerState;
  if (!PS) return;

  if (e.data === PS.PLAYING) {
    ytPlayStartAt = Date.now();
    if (ytHeartbeat) clearInterval(ytHeartbeat);
    ytHeartbeat = setInterval(() => flushPlayProgress(false), 10000);
  } else if (e.data === PS.PAUSED || e.data === PS.ENDED) {
    flushPlayProgress(true);
    if (ytHeartbeat) {
      clearInterval(ytHeartbeat);
      ytHeartbeat = null;
    }
  }
}

async function flushPlayProgress(forceSend) {
  if (!ytCurrentSongId || !ytPlayStartAt) return;
  const delta = Math.max(0, Math.floor((Date.now() - ytPlayStartAt) / 1000));
  ytPlayStartAt = Date.now();
  ytAccumulatedSec += delta;

  if (forceSend || ytAccumulatedSec >= 10) {
    await logInteraction(ytCurrentSongId, "play", ytAccumulatedSec);
    ytAccumulatedSec = 0;
  }
}

function playInsideIframe(song) {
  if (!song?.song_id) return;
  embedPanelDismissed = false;
  document.getElementById("player-embed-wrap")?.removeAttribute("hidden");
  ytCurrentSongId = song.song_id;
  ytAccumulatedSec = 0;
  ytPlayStartAt = null;

  loadYoutubeIframeAPI();
  if (!ytApiReady) {
    ytPendingLoadId = song.song_id;
    return;
  }

  ensureYtPlayer();
  if (ytPlayer && typeof ytPlayer.loadVideoById === "function") {
    try {
      ytPlayer.loadVideoById(song.song_id);
      ytPendingLoadId = null;
      ytEmbedLoadedId = song.song_id;
    } catch (_) {
      ytPendingLoadId = song.song_id;
    }
  } else {
    ytPendingLoadId = song.song_id;
  }
}

// ── 하이브리드 모달 ───────────────────────────────────────

function getSelectedHybridMode() {
  const chip = document.querySelector("#hyb-mode-seg .hyb-mode-chip.is-active");
  return chip?.dataset.hybMode || "advanced";
}

function setHybridModeUI(mode) {
  document.querySelectorAll("#hyb-mode-seg .hyb-mode-chip").forEach((btn) => {
    const on = btn.dataset.hybMode === mode;
    btn.classList.toggle("is-active", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

function onHybridModeSegClick(ev) {
  const btn = ev.target.closest(".hyb-mode-chip");
  if (!btn || !document.getElementById("hyb-mode-seg").contains(btn)) return;
  setHybridModeUI(btn.dataset.hybMode);
}

function updateHybridBalanceUI() {
  const el = document.getElementById("hyb-balance");
  const c = Math.min(100, Math.max(0, Number(el.value)));
  const co = 100 - c;
  document.getElementById("hyb-balance-pct-content").textContent = String(c);
  document.getElementById("hyb-balance-pct-cooc").textContent = String(co);
  el.setAttribute("aria-valuetext", `오디오 ${c}%, 협업 ${co}%`);
  const track = el.closest(".hyb-range-track--balance");
  if (track) track.style.setProperty("--balance", `${c}%`);
}

function updateHybridThresholdUI() {
  const el = document.getElementById("hyb-threshold");
  const raw = Math.min(100, Math.max(0, Number(el.value)));
  const v = raw / 100;
  document.getElementById("hyb-threshold-readout").textContent = v.toFixed(2);
}

function openHybridModal() {
  const w = hybridOptions.weights || { content: 0.55, cooc: 0.45 };
  const bc = Math.round((Number(w.content) || 0) * 100);
  setHybridModeUI(hybridOptions.mode || "advanced");

  const bal = document.getElementById("hyb-balance");
  bal.value = String(Math.min(100, Math.max(0, bc)));
  updateHybridBalanceUI();

  const th = Math.round((Number(hybridOptions.threshold) || 0) * 100);
  document.getElementById("hyb-threshold").value = String(Math.min(100, Math.max(0, th)));
  updateHybridThresholdUI();

  document.getElementById("hyb-use-llm-search").checked = !!hybridOptions.use_llm_search;

  document.getElementById("hybrid-modal").classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeHybridModal() {
  document.getElementById("hybrid-modal").classList.add("hidden");
  if (!document.getElementById("eval-modal")?.classList.contains("hidden")) return;
  if (!document.getElementById("eval-settings-modal")?.classList.contains("hidden")) return;
  document.body.style.overflow = "";
}

function saveHybridOptions() {
  const balance = Math.min(100, Math.max(0, Number(document.getElementById("hyb-balance").value)));
  hybridOptions = {
    mode: getSelectedHybridMode(),
    weights: {
      content: balance / 100,
      cooc: (100 - balance) / 100,
    },
    threshold: Math.min(1, Math.max(0, Number(document.getElementById("hyb-threshold").value) / 100)),
    use_llm_search: document.getElementById("hyb-use-llm-search").checked,
  };
  closeHybridModal();
}

function onModalEscape(ev) {
  if (ev.key !== "Escape") return;
  const es = document.getElementById("eval-settings-modal");
  if (es && !es.classList.contains("hidden")) {
    closeEvalSettingsModal();
    return;
  }
  const hyb = document.getElementById("hybrid-modal");
  const eva = document.getElementById("eval-modal");
  if (hyb && !hyb.classList.contains("hidden")) {
    closeHybridModal();
    return;
  }
  if (eva && !eva.classList.contains("hidden")) {
    closeEvalModal();
  }
}

// 평가 옵션 모달 + 결과 모달 + 표/차트
function syncEvalSettingsModePanels() {
  const mode =
    document.querySelector('input[name="eval-settings-eval-mode"]:checked')?.value ||
    "search";
  const ps = document.getElementById("eval-settings-panel-search");
  const pr = document.getElementById("eval-settings-panel-recommend");
  if (ps) ps.toggleAttribute("hidden", mode !== "search");
  if (pr) pr.toggleAttribute("hidden", mode !== "recommend");
  const seedHint = document.getElementById("eval-settings-seed-hint-login");
  if (seedHint) {
    const show = mode === "recommend" && !authToken;
    seedHint.toggleAttribute("hidden", !show);
  }
}

async function refreshEvalSettingsSeedSelect() {
  const sel = document.getElementById("eval-settings-seed-song-id");
  if (!sel) return;
  const keep = evalSettings.eval_seed_song_id || sel.value;
  sel.innerHTML =
    '<option value="">선택 안 함 — 모든 시드·유저 조합으로 평가</option>';
  if (!authToken) return;
  try {
    const res = await apiFetch("/api/likes");
    if (!res?.likes?.length) return;
    for (const song of res.likes) {
      if (!song?.song_id) continue;
      const opt = document.createElement("option");
      opt.value = song.song_id;
      const t = (song.title || "").trim();
      const a = (song.artist || "").trim();
      opt.textContent = t && a ? `${t} — ${a}` : t || song.song_id;
      sel.appendChild(opt);
    }
    if (keep && [...sel.options].some((o) => o.value === keep)) sel.value = keep;
  } catch (_) {
    /* ignore */
  }
}

function loadEvalSettingsUI() {
  const mode = evalSettings.eval_mode === "recommend" ? "recommend" : "search";
  const rSearch = document.getElementById("eval-settings-mode-search");
  const rRec = document.getElementById("eval-settings-mode-recommend");
  if (rSearch && rRec) {
    if (mode === "recommend") rRec.checked = true;
    else rSearch.checked = true;
  }
  const sq = document.getElementById("eval-settings-search-query");
  if (sq) sq.value = evalSettings.search_eval_query || "";
  const ou = document.getElementById("eval-settings-only-current-user");
  if (ou) {
    ou.disabled = !authUser;
    ou.checked = !!(authUser && evalSettings.eval_only_current_user);
  }
  syncEvalSettingsModePanels();
}

function readEvalSettingsFromForm() {
  evalSettings.eval_mode =
    document.querySelector('input[name="eval-settings-eval-mode"]:checked')?.value ||
    "search";
  evalSettings.search_eval_query =
    document.getElementById("eval-settings-search-query")?.value?.trim() || "";
  evalSettings.eval_seed_song_id =
    document.getElementById("eval-settings-seed-song-id")?.value?.trim() || "";
  const ou = document.getElementById("eval-settings-only-current-user");
  evalSettings.eval_only_current_user = !!(authUser && ou?.checked);
}

function saveEvalSettingsAndClose() {
  readEvalSettingsFromForm();
  persistEvalSettings();
  updateEvalSummaryInEvalModal();
  closeEvalSettingsModal();
}

function closeEvalSettingsModal() {
  document.getElementById("eval-settings-modal")?.classList.add("hidden");
  if (!document.getElementById("eval-modal")?.classList.contains("hidden")) return;
  if (!document.getElementById("hybrid-modal")?.classList.contains("hidden")) return;
  document.body.style.overflow = "";
}

async function openEvalSettingsModal() {
  loadEvalSettingsUI();
  document.getElementById("eval-settings-modal")?.classList.remove("hidden");
  document.body.style.overflow = "hidden";
  await refreshEvalSettingsSeedSelect();
  syncEvalSettingsModePanels();
}

function updateEvalSummaryInEvalModal() {
  const el = document.getElementById("eval-current-settings-summary");
  if (!el) return;
  const modeLabel =
    evalSettings.eval_mode === "recommend" ? "기준곡(추천)" : "검색어";
  const si = (document.getElementById("search-input")?.value || "").trim();
  const savedQ = (evalSettings.search_eval_query || "").trim();
  let detail = "";
  if (evalSettings.eval_mode === "search") {
    const q = savedQ || si;
    detail = q
      ? `문장: 「${escapeHtmlEval(q)}」`
      : "문장: (비어 있으면 실행 시 홈 검색창과 동일)";
  } else {
    detail = evalSettings.eval_seed_song_id
      ? `기준곡 ID: ${escapeHtmlEval(evalSettings.eval_seed_song_id)}`
      : "기준곡: 선택 안 함 · 전체 조합";
  }
  const scope =
    authUser && evalSettings.eval_only_current_user
      ? "범위: 현재 계정만"
      : "범위: DB 전체(좋아요 있는 계정)";
  el.innerHTML = `<strong>${modeLabel}</strong> · ${detail} · ${scope}<br /><span class="eval-run-info__muted">가중치·threshold·LLM은 하이브리드 설정과 동일</span>`;
}

async function openEvalModal() {
  document.getElementById("eval-modal").classList.remove("hidden");
  document.body.style.overflow = "hidden";
  updateEvalSummaryInEvalModal();
}

function closeEvalModal() {
  document.getElementById("eval-modal").classList.add("hidden");
  renderEvalRunInfo(null);
  const capRec = document.getElementById("eval-chart-caption-recommend");
  if (capRec) capRec.textContent = "오프라인 평가";
  if (!document.getElementById("hybrid-modal")?.classList.contains("hidden")) return;
  if (!document.getElementById("eval-settings-modal")?.classList.contains("hidden")) return;
  document.body.style.overflow = "";
}

const EVAL_COL_LABELS = {
  weight_audio_pct: "설정·오디오%",
  weight_cooc_pct: "설정·협업%",
  threshold: "설정·임계값",
  variant_mode: "조합·공동출현",
  variant_use_llm_search: "조합·LLM검색",
  variant: "평가방식",
  eval_kind: "평가종류",
  search_query: "검색어",
  n_cases: "케이스수",
  user_id: "user_id",
  query_song_id: "기준곡 ID",
  query_title: "기준곡 제목",
  query_artist: "기준곡 아티스트",
  n_relevant: "정답곡수",
};

const EVAL_TABLE_COL_PRIORITY = [
  "eval_kind",
  "search_query",
  "query_title",
  "query_artist",
  "query_song_id",
  "user_id",
  "n_relevant",
  "variant_mode",
  "variant_use_llm_search",
  "weight_audio_pct",
  "weight_cooc_pct",
  "threshold",
];

function evalTableColumnOrder(cols) {
  const out = [];
  for (const p of EVAL_TABLE_COL_PRIORITY) {
    if (cols.includes(p)) out.push(p);
  }
  for (const c of cols) {
    if (!out.includes(c)) out.push(c);
  }
  return out;
}

function evalColTitle(key) {
  return EVAL_COL_LABELS[key] || key;
}

function escapeHtmlEval(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatEvalWeightsLine(w) {
  if (!w || typeof w !== "object") return "";
  const a = Math.round(Number(w.content ?? 0) * 100);
  const b = Math.round(Number(w.cooc ?? 0) * 100);
  return `오디오 ${a}% · 협업 ${b}%`;
}

function renderEvalRunInfo(res) {
  const el = document.getElementById("eval-run-info");
  if (!el) return;
  if (!res) {
    el.innerHTML = "";
    el.setAttribute("hidden", "");
    return;
  }
  const algo = res.algorithm || "";
  const kList = (res.k_list || []).join(", ");
  const snap = res.hybrid_sidebar_snapshot;
  const blocks = [];

  if (snap && isHybridAlgo(algo)) {
    const modeLabel =
      snap.mode === "simple"
        ? "Simple"
        : snap.mode === "advanced"
          ? "Advanced"
          : String(snap.mode || "—");
    const wline = formatEvalWeightsLine(snap.weights);
    const th = snap.threshold != null ? Number(snap.threshold) : 0;
    const llm =
      snap.use_llm_search === true
        ? "자연어 검색 시 LLM: 켜짐"
        : "자연어 검색 시 LLM: 꺼짐";
    blocks.push(
      `<p class="eval-run-info__strong">하이브리드 설정 <span class="eval-run-info__muted">(사이드바·모달에 저장된 값)</span></p>`,
      `<p class="eval-run-info__line">공동출현 모드: <strong>${escapeHtmlEval(modeLabel)}</strong> · 가중치: <strong>${escapeHtmlEval(wline || "—")}</strong> · threshold: <strong>${escapeHtmlEval(String(th))}</strong></p>`,
      `<p class="eval-run-info__line">${escapeHtmlEval(llm)}</p>`
    );
    const evm = res.eval_eval_mode || res.options_used?.eval_mode || "search";
    if (res.mode === "variants") {
      if (evm === "search") {
        blocks.push(
          `<p class="eval-run-info__note">표·차트는 <strong>네 조합</strong>(Simple/Advanced × LLM)마다 <strong>동일 검색어</strong>로 <code>search_by_query_with_options</code>를 돌린 결과입니다. 가중치·threshold는 추천용이며 검색 점수에는 직접 쓰이지 않을 수 있습니다.</p>`
        );
      } else {
        blocks.push(
          `<p class="eval-run-info__note">표·차트는 <strong>네 조합</strong>(Simple/Advanced × LLM on/off)마다 계산했고, 가중치·threshold·LLM 스위치는 <strong>요청 시점 저장값</strong>이 <code>recommend_with_options</code>에 넘어갑니다. 시드 곡 메타로 텍스트 검색이 합성되므로 LLM on/off에 따라 수치가 달라질 수 있습니다.</p>`
        );
      }
    } else {
      blocks.push(
        `<p class="eval-run-info__note">단일 평가(run_variants=false)입니다. 표의 조합 열은 위 저장 설정과 같습니다.</p>`
      );
    }
  } else {
    blocks.push(
      `<p class="eval-run-info__line">알고리즘: <strong>${escapeHtmlEval(algo)}</strong> · K: ${escapeHtmlEval(kList)}</p>`
    );
  }

  const evMode = res.eval_eval_mode || res.options_used?.eval_mode || "search";
  const usedQ = (res.options_used && res.options_used.search_eval_query) || "";
  if (evMode === "search" && String(usedQ).trim()) {
    blocks.push(
      `<p class="eval-run-info__line">이번 검색어 평가 문장: <strong>${escapeHtmlEval(String(usedQ).trim())}</strong></p>`
    );
  }
  const usedSeed = (res.options_used && res.options_used.eval_seed_song_id) || "";
  if (evMode === "recommend" && String(usedSeed).trim()) {
    blocks.push(
      `<p class="eval-run-info__line">선택한 기준곡 ID: <strong>${escapeHtmlEval(String(usedSeed).trim())}</strong></p>`
    );
  }

  if (res.eval_only_current_user === true) {
    blocks.push(
      `<p class="eval-run-info__line">평가 범위: <strong>현재 로그인 계정만</strong> (user_id=${escapeHtmlEval(String(res.eval_filter_user_id ?? ""))})</p>`
    );
  } else if (res.eval_only_current_user === false) {
    blocks.push(`<p class="eval-run-info__line">평가 범위: <strong>DB 전체 유저</strong></p>`);
  }

  const pa = res.eval_recommend_pool_all_users;
  const pf = res.eval_recommend_pool_filtered_user;
  const ne = res.eval_recommend_cases_evaluated;
  if (pa != null && ne != null) {
    const samePool = pf != null && Number(pa) === Number(pf) && res.eval_only_current_user;
    const kindLabel = evMode === "search" ? "검색어 평가 케이스" : "기준곡(추천) 평가 케이스";
    blocks.push(
      `<p class="eval-run-info__line">${escapeHtmlEval(kindLabel)} — DB 전체 풀 <strong>${escapeHtmlEval(String(pa))}</strong>건 · 필터 적용 후 대상 풀 <strong>${escapeHtmlEval(String(pf ?? "—"))}</strong>건 · 이번에 실제 계산 <strong>${escapeHtmlEval(String(ne))}</strong>건${
        samePool ? " <span class=\"eval-run-info__muted\">(전체와 내 계정 풀이 같아 지표가 동일할 수 있음)</span>" : ""
      }</p>`
    );
  }

  if (res.eval_notice) {
    blocks.push(
      `<p class="eval-run-info__note">${escapeHtmlEval(String(res.eval_notice))}</p>`
    );
  }

  el.innerHTML = blocks.join("");
  el.removeAttribute("hidden");
}

async function runEvaluation() {
  const algo = currentAlgo();
  const opts = isHybridAlgo(algo) ? { ...hybridOptions } : {};
  const onlyMe = !!(authUser && evalSettings.eval_only_current_user);
  opts.eval_only_current_user = onlyMe;

  opts.eval_mode = evalSettings.eval_mode;

  if (evalSettings.eval_mode === "search") {
    const q = (evalSettings.search_eval_query || "").trim();
    const si = document.getElementById("search-input")?.value?.trim() || "";
    opts.search_eval_query = q || si;
  } else if (evalSettings.eval_seed_song_id) {
    opts.eval_seed_song_id = evalSettings.eval_seed_song_id;
  }

  const body = {
    algorithm: algo,
    options: opts,
    eval_only_current_user: onlyMe,
  };
  let res;
  try {
    const fetchOpts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    };
    if (authToken) fetchOpts.headers.Authorization = `Bearer ${authToken}`;
    const r = await fetch(API + "/api/eval/run", fetchOpts);
    if (r.status === 401) {
      let msg = "로그인이 필요합니다. (「현재 계정만」 평가)";
      try {
        const j = await r.json();
        if (typeof j.detail === "string") msg = j.detail;
      } catch (_) {}
      alert(msg);
      return;
    }
    if (!r.ok) {
      let msg = "평가 실패 (서버 오류 또는 알고리즘 없음)";
      try {
        const j = await r.json();
        if (typeof j.detail === "string") msg = j.detail;
      } catch (_) {}
      return alert(msg);
    }
    res = await r.json();
  } catch (e) {
    console.error(e);
    return alert("평가 실패 (네트워크 오류)");
  }

  renderEvalRunInfo(res);
  const capRec = document.getElementById("eval-chart-caption-recommend");
  if (capRec) {
    const only = res.eval_only_current_user === true;
    const ne = res.eval_recommend_cases_evaluated;
    const em = res.eval_eval_mode || evalSettings.eval_mode;
    if (em === "search") {
      const q = (res.options_used && res.options_used.search_eval_query) || opts.search_eval_query || "";
      capRec.textContent =
        only && ne != null
          ? `검색어 평가 · 「${String(q).slice(0, 40)}${String(q).length > 40 ? "…" : ""}」 · 내 계정만 · ${ne}건`
          : only
            ? `검색어 평가 · 내 계정만`
            : `검색어 평가 · DB 전체 유저`;
    } else {
      capRec.textContent =
        only && ne != null
          ? `기준곡 추천 평가 · 내 계정만 · ${ne}건`
          : only
            ? `기준곡 추천 평가 · 내 계정만`
            : `기준곡 추천 평가 · DB 전체 유저`;
    }
  }
  renderEvalTable(res.rows || [], res.summary_rows || [], res);
  renderEvalChart(res.summary || {});
}

function _fmtEvalCell(v, col) {
  if (col === "eval_kind") {
    if (v === "recommend") return "기준곡";
    if (v === "search") return "검색어";
  }
  if (col === "variant_use_llm_search") {
    if (v === true || v === "true") return "반영(on)";
    if (v === false || v === "false") return "미반영(off)";
  }
  if (col === "variant_mode") {
    if (v === "simple") return "Simple";
    if (v === "advanced") return "Advanced";
  }
  if (v === null || v === undefined) return "";
  if (typeof v === "number" && Number.isFinite(v)) return String(Math.round(v * 10000) / 10000);
  return String(v);
}

function renderEvalTable(rows, summaryRows, res) {
  const wrap = document.getElementById("eval-table-wrap");
  let zeroBanner = "";
  const evMode = res?.eval_eval_mode || res?.options_used?.eval_mode || "search";
  if (res && Number(res.eval_recommend_cases_evaluated) === 0) {
    const only = res.eval_only_current_user === true;
    if (evMode === "search") {
      zeroBanner = `<p class="eval-banner eval-banner--warn">검색어 평가 <strong>케이스 0건</strong>입니다. Precision/Recall/NDCG는 데이터 없이 <strong>0</strong>으로 보일 수 있습니다.${
        only
          ? " <strong>현재 계정</strong>에 좋아요가 1곡 이상 DB에 있는지, 검색어를 입력했는지 확인하세요."
          : " 좋아요 1곡 이상인 유저가 DB에 있는지 확인하세요."
      }</p>`;
    } else {
      zeroBanner = `<p class="eval-banner eval-banner--warn">기준곡(추천) 평가 <strong>케이스 0건</strong>입니다. Precision/Recall/NDCG는 <strong>데이터 없이 0</strong>으로 보일 수 있습니다.${
        only
          ? " <strong>현재 계정</strong>에 좋아요 2곡 이상(또는 재생 20초 이상 곡 2곡 이상)이 있는지, 아래 <strong>기준곡 선택</strong>에 쓸 좋아요가 있는지 확인하세요."
          : " DB에 해당 조건을 만족하는 유저·로그가 있는지 확인하세요."
      }</p>`;
    }
  }

  if (rows.length) {
    const cols = evalTableColumnOrder(Object.keys(rows[0]));
    const head = `<tr>${cols.map((c) => `<th>${escapeHtmlEval(evalColTitle(c))}</th>`).join("")}</tr>`;
    const body = rows
      .map(
        (r) =>
          `<tr>${cols.map((c) => `<td>${escapeHtmlEval(_fmtEvalCell(r[c], c))}</td>`).join("")}</tr>`
      )
      .join("");
    const sub =
      evMode === "search"
        ? "검색어 평가 · 행마다 검색어·user_id · 하이브리드면 조합(공동출현·LLM) 열 참고"
        : "기준곡 평가 · 표의 기준곡 제목·아티스트가 시드 곡 · 하이브리드면 조합 열 참고";
    wrap.innerHTML = `${zeroBanner}<p class="eval-subcap">${escapeHtmlEval(sub)}</p><table class="eval-table"><thead>${head}</thead><tbody>${body}</tbody></table>`;
    return;
  }

  if (summaryRows && summaryRows.length) {
    const cols = evalTableColumnOrder(Object.keys(summaryRows[0]));
    const head = `<tr>${cols.map((c) => `<th>${escapeHtmlEval(evalColTitle(c))}</th>`).join("")}</tr>`;
    const body = summaryRows
      .map(
        (r) =>
          `<tr>${cols.map((c) => `<td>${escapeHtmlEval(_fmtEvalCell(r[c], c))}</td>`).join("")}</tr>`
      )
      .join("");
    const zero = summaryRows.every((r) => !r.n_cases);
    const cap = zero
      ? "<p class=\"eval-subcap\">요약: 케이스가 없어 지표는 0입니다. 각 행 = 공동출현 모드 × LLM · n_cases 확인</p>"
      : "<p class=\"eval-subcap\">요약: 각 행 = 공동출현 모드 × LLM 옵션 · 앞 열은 설정·오디오%/협업%/임계값</p>";
    wrap.innerHTML = `${zeroBanner}${cap}<table class="eval-table"><thead>${head}</thead><tbody>${body}</tbody></table>`;
    return;
  }

  wrap.innerHTML =
    zeroBanner +
    "<p class=\"eval-empty\">표 데이터가 없습니다. 알고리즘을 선택한 뒤 다시 실행하거나 서버 오류를 확인하세요.</p>";
}

/** 평가 막대그래프 Y축(0~1) 눈금 옆 해석 — 대략적인 가이드 */
function formatEvalYAxisTick(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return String(value);
  if (v <= 0.02) return "0\n(맞춤 거의 없음)";
  if (Math.abs(v - 0.25) < 0.04) return "0.25\n(미흡)";
  if (Math.abs(v - 0.5) < 0.04) return "0.5\n(보통)";
  if (Math.abs(v - 0.75) < 0.04) return "0.75\n(양호)";
  if (v >= 0.97) return "1.0\n(매우 좋음)";
  return v.toFixed(2);
}

const evalChartScales = {
  x: {
    ticks: { maxRotation: 45, minRotation: 45, font: { size: 9 } },
  },
  y: {
    min: 0,
    max: 1,
    ticks: {
      stepSize: 0.25,
      callback: (value) => formatEvalYAxisTick(value),
      font: { size: 10, lineHeight: 1.25 },
      maxRotation: 0,
      autoSkip: false,
    },
    title: {
      display: true,
      text: "지표 값 (0 = 최악, 1 = 이상적)",
      color: "#9a9a9a",
      font: { size: 11 },
    },
    grid: { color: "rgba(255,255,255,0.06)" },
  },
};

const evalChartLayout = {
  padding: { left: 4, right: 8, top: 4, bottom: 4 },
};

let evalChart;

function renderEvalChart(summary) {
  const ctx = document.getElementById("eval-chart");
  if (evalChart) {
    evalChart.destroy();
    evalChart = null;
  }
  const labels = Object.keys(summary || {});
  const data = Object.values(summary || {});
  if (!labels.length) {
    if (typeof Chart === "undefined") return;
    evalChart = new Chart(ctx, {
      type: "bar",
      data: { labels: ["(데이터 없음)"], datasets: [{ label: "지표", data: [0], backgroundColor: "#333" }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: evalChartLayout,
        scales: evalChartScales,
      },
    });
    return;
  }
  evalChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "지표",
          data,
          backgroundColor: "rgba(29, 185, 84, 0.55)",
          borderColor: "rgba(29, 185, 84, 0.9)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: evalChartLayout,
      scales: evalChartScales,
      plugins: {
        tooltip: {
          callbacks: {
            label(ctx) {
              const raw = ctx.raw;
              const n = typeof raw === "number" ? raw : parseFloat(raw);
              if (!Number.isFinite(n)) return String(raw);
              let hint = "";
              if (n < 0.2) hint = " — 맞춤 거의 없음";
              else if (n < 0.4) hint = " — 다소 미흡";
              else if (n < 0.6) hint = " — 보통 수준";
              else if (n < 0.8) hint = " — 양호";
              else hint = " — 매우 좋음";
              return ` ${n.toFixed(4)}${hint}`;
            },
          },
        },
      },
    },
  });
}