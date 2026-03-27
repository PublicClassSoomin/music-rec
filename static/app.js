const API = "";
let currentSong   = null;
let currentAlgo   = () => document.getElementById("algo-select").value;
let authUser      = null;
let authToken     = null;
let currentUser   = () => authUser?.username || "guest";
let likedSongs    = new Set();
let songCache     = new Map(); // song_id -> full song object
let currentGenre  = null;
let currentFeed   = [];
let skippedSongs  = new Set();
let currentFeedContext = { type: "catalog" };

// ── 초기화 ────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", async () => {
  bindAuthEvents();
  await bootstrapAuth();
  setupNav();
  await loadAlgorithms();
  await loadInitialCatalog();

  document.getElementById("search-btn").addEventListener("click", onSearch);
  document.getElementById("search-input").addEventListener("keydown", e => {
    if (e.key === "Enter") onSearch();
  });
  document.getElementById("btn-youtube").addEventListener("click", openYoutube);
  document.getElementById("btn-like-player").addEventListener("click", onLikePlayer);
  document.getElementById("user-reco-btn").addEventListener("click", fetchUserRecommendations);
  document.getElementById("algo-select").addEventListener("change", () => {
    if (currentSong) {
      fetchRecommendations(currentSong.song_id);
    } else if (authUser) {
      fetchUserRecommendations();
    }
  });
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
  authUser = { username: me.username };
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
  authUser = { username: res.username };
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
  authUser = { username: res.username };
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
  skippedSongs.clear();
  localStorage.removeItem("auth_token");
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
}

async function loadInitialCatalog(retries = 5, delayMs = 800) {
  for (let attempt = 0; attempt < retries; attempt += 1) {
    const songs = await apiFetch("/api/songs?limit=200");
    if (songs?.length) {
      renderGenresFromSongs(songs);
      currentFeedContext = { type: "catalog", genre: null };
      updateMainGrid(filterVisibleSongs(songs).slice(0, 3), false);
      return;
    }
    await sleep(delayMs);
  }

  document.getElementById("song-grid").innerHTML =
    "<p class='placeholder'>초기 곡 목록을 불러오지 못했습니다. 잠시 후 새로고침해 주세요.</p>";
}

function renderGenresFromSongs(songs) {
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
      currentGenre = genre;
      loadSongs(genre);
    });
    list.appendChild(el);
  });
}

async function loadSongs(genre = null) {
  currentGenre = genre;
  const url = genre ? `/api/songs?limit=30&genre=${encodeURIComponent(genre)}` : "/api/songs?limit=30";
  const songs = await apiFetch(url);
  if (songs) {
    const visibleSongs = filterVisibleSongs(songs).slice(0, 3);
    currentFeedContext = { type: "catalog", genre };
    updateMainGrid(visibleSongs, false);
  }
}

// ── 추천 ──────────────────────────────────────────────────

async function fetchRecommendations(songId) {
  const algo = currentAlgo();
  if (!algo) return;

  const res = await apiFetch("/api/recommend", "POST", {
    song_id:   songId,
    algorithm: algo,
    top_k:     12,
    exclude_song_ids: [...skippedSongs],
  });

  if (res?.recommendations) {
    currentFeedContext = { type: "song", songId, algorithm: algo };
    updateMainGrid(filterVisibleSongs(res.recommendations).slice(0, 3), true);
    document.getElementById("algo-badge").textContent = algo;
  }
}

async function fetchUserRecommendations() {
  const algo = currentAlgo();
  if (!algo) return alert("알고리즘을 먼저 선택해주세요.");
  if (!authUser) return alert("로그인 후 사용해주세요.");

  const res = await apiFetch("/api/recommend/user", "POST", {
    username: authUser.username,
    algorithm: algo,
    top_k: 12,
    exclude_song_ids: [...skippedSongs],
  });
  if (res?.recommendations) {
    currentFeedContext = { type: "user", username: authUser.username, algorithm: algo };
    updateMainGrid(filterVisibleSongs(res.recommendations).slice(0, 3), true);
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

  const res = await apiFetch("/api/search", "POST", {
    query, algorithm: algo, top_k: 12, exclude_song_ids: [...skippedSongs],
  });

  const block = document.getElementById("search-results-block");
  const container = document.getElementById("search-result");
  block.removeAttribute("hidden");

  if (res?.recommendations?.length) {
    renderGrid(res.recommendations.slice(0, 3), "search-result", true);
  } else {
    container.innerHTML = "<p class='placeholder'>검색 결과가 없거나 이 알고리즘은 자연어 검색을 지원하지 않습니다.</p>";
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

function openYoutube() {
  if (!currentSong) return;
  logInteraction(currentSong.song_id, "play", 30);
  window.open(currentSong.youtube_url, "_blank");
}

// ── 렌더링 ────────────────────────────────────────────────

function updateMainGrid(songs, showScore) {
  currentFeed = songs.slice();
  renderGrid(currentFeed, "song-grid", showScore);
}

function filterVisibleSongs(songs) {
  return (songs || []).filter(song => song?.song_id && !skippedSongs.has(song.song_id));
}

function renderGrid(songs, containerId, showScore) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  if (!songs.length) {
    container.innerHTML = "<p class='placeholder'>곡이 없습니다.</p>";
    return;
  }

  songs.forEach(song => {
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
      skippedSongs.add(song.song_id);
      await handleSkip(song.song_id);
    });

    container.appendChild(tpl);
  });
}

async function handleSkip(songId) {
  currentFeed = currentFeed.filter(song => song.song_id !== songId);

  if (currentFeed.length < 3) {
    await refillRecommendations();
  } else {
    renderGrid(currentFeed.slice(0, 3), "song-grid", hasScoredSongs(currentFeed));
  }
}

async function refillRecommendations() {
  const ctx = currentFeedContext;
  if (ctx.type === "song" && ctx.songId) {
    const res = await apiFetch("/api/recommend", "POST", {
      song_id: ctx.songId,
      algorithm: ctx.algorithm,
      top_k: 20,
      exclude_song_ids: [...skippedSongs],
    });
    if (res?.recommendations) {
      updateMainGrid(filterVisibleSongs(res.recommendations).slice(0, 3), true);
      return;
    }
  }

  if (ctx.type === "user" && ctx.username) {
    const res = await apiFetch("/api/recommend/user", "POST", {
      username: ctx.username,
      algorithm: ctx.algorithm,
      top_k: 20,
      exclude_song_ids: [...skippedSongs],
    });
    if (res?.recommendations) {
      updateMainGrid(filterVisibleSongs(res.recommendations).slice(0, 3), true);
      return;
    }
  }

  if (ctx.type === "catalog") {
    await loadSongs(ctx.genre || null);
    return;
  }

  renderGrid(currentFeed.slice(0, 3), "song-grid", hasScoredSongs(currentFeed));
}

function hasScoredSongs(songs) {
  return (songs || []).some(song => song?.score != null);
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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
