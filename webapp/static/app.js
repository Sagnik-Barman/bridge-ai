const SEATS = ["N", "E", "S", "W"];
const STRAINS = ["C", "D", "H", "S", "NT"];
const RED_SUITS = ["H", "D"];

let lastFeedback = null;

async function apiGet(path) {
  const res = await fetch(path);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

async function apiDelete(path) {
  const res = await fetch(path, { method: "DELETE" });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function suitOf(cardStr) {
  return cardStr[cardStr.length - 1] === "N" ? "NT" : cardStr.slice(-1);
}

function rankOf(cardStr) {
  return cardStr.slice(0, -1);
}

function renderCardChip(cardStr, { clickable = false, onClick = null } = {}) {
  const span = document.createElement("span");
  const suit = suitOf(cardStr);
  span.className = "card-chip" + (RED_SUITS.includes(suit) ? " red" : "");
  span.textContent = cardStr;
  if (clickable) {
    span.classList.add("legal");
    span.addEventListener("click", onClick);
  }
  return span;
}

function renderHands(state) {
  for (const seat of SEATS) {
    const container = document.querySelector(`#seat-${seat} .cards`);
    container.innerHTML = "";
    const cards = state.hands[seat];
    if (!cards) continue;
    const isMyTurnHand = state.phase === "play" && state.next_to_play === seat && state.controlled_seats.includes(seat);
    for (const cardStr of cards) {
      const legalHere = isMyTurnHand && state.legal_cards.includes(cardStr);
      container.appendChild(
        renderCardChip(cardStr, {
          clickable: legalHere,
          onClick: () => playCard(seat, cardStr),
        })
      );
    }
  }
}

function renderTrick(state) {
  const el = document.getElementById("trick-cards");
  el.innerHTML = "";
  if (!state.current_trick) return;
  const bySeat = {};
  for (const play of state.current_trick) bySeat[play.seat] = play.card;
  // Layout: N top-left cell? Just show seat: card pairs in a simple grid, order N,W,E,S positions matching table.
  const order = ["N", "W", "E", "S"];
  for (const seat of order) {
    const cell = document.createElement("div");
    if (bySeat[seat]) {
      cell.appendChild(renderCardChip(bySeat[seat]));
    } else {
      cell.textContent = "";
    }
    el.appendChild(cell);
  }
}

function renderAuction(state) {
  const tbody = document.querySelector("#auction-table tbody");
  tbody.innerHTML = "";
  const dealer = state.dealer;
  const startCol = SEATS.indexOf(dealer);
  const rows = [];
  state.auction.forEach((entry, i) => {
    const rowIdx = Math.floor((startCol + i) / 4);
    const colIdx = (startCol + i) % 4;
    if (!rows[rowIdx]) rows[rowIdx] = ["", "", "", ""];
    rows[rowIdx][colIdx] = entry;
  });
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (let c = 0; c < 4; c++) {
      const td = document.createElement("td");
      const entry = row ? row[c] : null;
      if (entry) {
        td.textContent = entry.call;
        td.classList.add("has-explanation");
        td.title = entry.explanation;
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function renderSuggestion(state) {
  const panel = document.getElementById("suggestion-panel");
  if (state.phase === "bidding" && state.suggestion) {
    panel.hidden = false;
    document.getElementById("suggestion-call").textContent = state.suggestion.call;
    document.getElementById("suggestion-explanation").textContent = state.suggestion.explanation;
  } else {
    panel.hidden = true;
  }
}

function renderFeedback() {
  const panel = document.getElementById("feedback-panel");
  if (!lastFeedback) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  panel.classList.remove("match", "mismatch");
  const matched = lastFeedback.matched_suggestion;
  panel.classList.add(matched ? "match" : "mismatch");
  if (matched) {
    panel.innerHTML = `<div class="fb-title">&#10003; You bid ${lastFeedback.call} — matches the suggestion.</div>`;
  } else {
    panel.innerHTML = `<div class="fb-title">You bid ${lastFeedback.call}.</div>
      <div>Suggested: <strong>${lastFeedback.suggested_call}</strong> — ${lastFeedback.suggestion_explanation}</div>`;
  }
}

function renderBiddingBox(state) {
  const box = document.getElementById("bidding-box");
  const myTurn = state.phase === "bidding" && state.whose_turn_to_call === state.user_seat;
  box.hidden = !myTurn;
  if (!myTurn) return;

  for (let level = 1; level <= 7; level++) {
    const row = document.getElementById(`bid-row-${level}`);
    row.innerHTML = "";
    for (const strain of STRAINS) {
      const call = `${level}${strain}`;
      const btn = document.createElement("button");
      btn.textContent = call;
      btn.dataset.call = call;
      btn.disabled = !state.legal_calls.includes(call);
      btn.addEventListener("click", () => makeCall(call));
      row.appendChild(btn);
    }
  }
  const specialRow = document.getElementById("bid-row-special");
  for (const btn of specialRow.querySelectorAll("button")) {
    btn.disabled = !state.legal_calls.includes(btn.dataset.call);
    btn.onclick = () => makeCall(btn.dataset.call);
  }
}

function renderContract(state) {
  const panel = document.getElementById("contract-panel");
  if (state.contract) {
    panel.hidden = false;
    const c = state.contract;
    document.getElementById("contract-text").textContent =
      `${c.level}${c.strain}${c.double} by ${c.declarer}`;
    if (state.tricks_by_side) {
      document.getElementById("tricks-text").textContent =
        `Tricks — NS: ${state.tricks_by_side.NS}, EW: ${state.tricks_by_side.EW}`;
    }
  } else {
    panel.hidden = true;
  }
}

function renderControls(state) {
  document.getElementById("undo-call-btn").hidden = !(state.phase === "bidding" && state.can_undo_call);
  document.getElementById("undo-card-btn").hidden = !(state.phase === "play" && state.can_undo_card);
  document.getElementById("analyze-btn").hidden = state.phase !== "play";
  document.getElementById("next-board-btn").hidden = state.phase !== "board_complete";
}

const SOURCE_LABELS = {
  heuristic: "Heuristic bot",
  dds: "DDS / PIMC",
  rl: "RL network",
  exact_endgame: "Exact endgame solver",
};

function renderAnalysis(analysis) {
  const panel = document.getElementById("analysis-panel");
  if (!analysis) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  document.getElementById("analysis-seat").textContent = `(${analysis.seat} to play, ${analysis.cards_remaining} cards left)`;

  const consensusEl = document.getElementById("analysis-consensus");
  if (analysis.sources_agree && analysis.consensus) {
    consensusEl.textContent = `All available sources agree: play ${analysis.consensus}.`;
    consensusEl.className = "analysis-agree";
  } else if (analysis.consensus) {
    consensusEl.textContent = "";
    consensusEl.className = "";
  } else {
    consensusEl.textContent = "Sources disagree — see the breakdown below.";
    consensusEl.className = "analysis-disagree";
  }

  const tbody = document.querySelector("#analysis-table tbody");
  tbody.innerHTML = "";
  for (const src of ["heuristic", "dds", "rl", "exact_endgame"]) {
    const entry = analysis[src];
    if (!entry) continue;
    const tr = document.createElement("tr");
    const nameTd = document.createElement("td");
    nameTd.textContent = SOURCE_LABELS[src] || src;
    const cardTd = document.createElement("td");
    cardTd.textContent = entry.available ? entry.suggested_card : "—";
    const noteTd = document.createElement("td");
    noteTd.textContent = entry.note;
    noteTd.className = "analysis-note";
    tr.appendChild(nameTd);
    tr.appendChild(cardTd);
    tr.appendChild(noteTd);
    tbody.appendChild(tr);
  }
}

async function analyzePosition() {
  try {
    const data = await apiPost("/api/analyze");
    renderAnalysis(data.analysis);
  } catch (e) {
    alert(e.message);
  }
}

function renderBoardResult(state) {
  const panel = document.getElementById("board-result-panel");
  if (state.phase === "board_complete" && state.last_board_result) {
    panel.hidden = false;
    const r = state.last_board_result;
    let text = `<div><strong>${r.contract_summary}</strong></div><div>${r.score_summary}</div>`;
    if (r.user_imps !== null && r.user_imps !== undefined) {
      const sign = r.user_imps > 0 ? "+" : "";
      text += `<div>Your side: ${sign}${r.user_imps} IMPs vs. double-dummy par (${r.par_tricks} tricks in ${r.par_strain})</div>`;
    } else {
      text += `<div style="color:#888">(Install <code>endplay</code> to see an IMP comparison against double-dummy par.)</div>`;
    }
    document.getElementById("board-result-text").innerHTML = text;
  } else {
    panel.hidden = true;
  }
}

const PLAY_MODE_LABELS = {
  heuristic: "vs. Standard bot",
  dds: "vs. DDS",
  rl: "vs. RL",
};

function renderTopBar(state) {
  document.getElementById("board-number").textContent = `Board ${state.board_number}`;
  document.getElementById("dealer-vul").textContent = `Dealer ${state.dealer} · Vul ${state.vulnerable}`;
  document.getElementById("your-seat").textContent = `You are ${state.user_seat}`;
  const modeLabel = document.getElementById("play-mode-label");
  modeLabel.textContent = PLAY_MODE_LABELS[state.play_mode] || state.play_mode;
  modeLabel.title = state.play_mode_status || "";
  const imps = state.total_imps;
  document.getElementById("total-imps").textContent =
    imps !== 0 || state.boards_played > 0 ? `${imps >= 0 ? "+" : ""}${imps} IMPs` : "";
  document.getElementById("boards-played").textContent = `${state.boards_played} board(s) played`;
}

function render(state) {
  renderTopBar(state);
  renderHands(state);
  renderTrick(state);
  renderAuction(state);
  renderSuggestion(state);
  renderFeedback();
  renderBiddingBox(state);
  renderContract(state);
  renderControls(state);
  renderBoardResult(state);
  renderAnalysis(null); // the position just changed — any prior analysis is stale
}

async function refresh() {
  const state = await apiGet("/api/state");
  render(state);
}

async function makeCall(call) {
  try {
    const state = await apiPost("/api/call", { call });
    lastFeedback = state.feedback;
    render(state);
  } catch (e) {
    alert(e.message);
  }
}

async function playCard(seat, cardStr) {
  try {
    const state = await apiPost("/api/card", { seat, card: cardStr });
    render(state);
  } catch (e) {
    alert(e.message);
  }
}

async function undoCall() {
  const state = await apiPost("/api/undo_call");
  lastFeedback = null;
  render(state);
}

async function undoCard() {
  const state = await apiPost("/api/undo_card");
  render(state);
}

async function restartBoard() {
  const state = await apiPost("/api/restart_board");
  lastFeedback = null;
  render(state);
}

async function nextBoard() {
  const state = await apiPost("/api/next_board");
  lastFeedback = null;
  render(state);
}

// -- saved games -----------------------------------------------------

function formatSavedAt(epochSeconds) {
  if (!epochSeconds) return "";
  const d = new Date(epochSeconds * 1000);
  return d.toLocaleString();
}

function renderSavedGamesList(saves) {
  const list = document.getElementById("saved-games-list");
  list.innerHTML = "";
  if (!saves.length) {
    const msg = document.createElement("div");
    msg.id = "no-saves-msg";
    msg.textContent = "No saved games yet — save your current game below to come back to it later.";
    list.appendChild(msg);
    return;
  }
  for (const save of saves) {
    const row = document.createElement("div");
    row.className = "saved-game-row";
    const imps = save.total_imps;
    const impsText = imps !== null && imps !== undefined ? ` · ${imps >= 0 ? "+" : ""}${imps} IMPs` : "";
    row.innerHTML = `
      <div class="sg-info">
        <span class="sg-name">${save.name}</span>
        <span>Board ${save.board_number} · ${save.phase} · ${save.boards_played} played${impsText}</span>
        <span>${formatSavedAt(save.saved_at)}</span>
      </div>
      <div class="sg-actions">
        <button class="load-btn">Load</button>
        <button class="delete-btn">Delete</button>
      </div>
    `;
    row.querySelector(".load-btn").addEventListener("click", () => loadSavedGame(save.name));
    row.querySelector(".delete-btn").addEventListener("click", () => deleteSavedGame(save.name));
    list.appendChild(row);
  }
}

async function openSavedGamesModal() {
  document.getElementById("saved-games-modal").hidden = false;
  document.getElementById("save-name-input").value = "";
  const data = await apiGet("/api/saves");
  renderSavedGamesList(data.saves);
}

function closeSavedGamesModal() {
  document.getElementById("saved-games-modal").hidden = true;
}

async function saveCurrentGame() {
  const name = document.getElementById("save-name-input").value.trim();
  if (!name) {
    alert("Give this game a name first.");
    return;
  }
  try {
    const data = await apiPost("/api/save", { name });
    document.getElementById("save-name-input").value = "";
    renderSavedGamesList(data.saves);
  } catch (e) {
    alert(e.message);
  }
}

async function loadSavedGame(name) {
  try {
    const state = await apiPost("/api/load", { name });
    lastFeedback = null;
    closeSavedGamesModal();
    render(state);
  } catch (e) {
    alert(e.message);
  }
}

async function deleteSavedGame(name) {
  if (!confirm(`Delete saved game "${name}"? This can't be undone.`)) return;
  try {
    const data = await apiDelete(`/api/saves/${encodeURIComponent(name)}`);
    renderSavedGamesList(data.saves);
  } catch (e) {
    alert(e.message);
  }
}

// -- new game (with opponent/play-mode picker) ------------------------

const PLAY_MODE_HINTS = {
  heuristic: "Club-level heuristics — fast, no extra setup needed.",
  dds: "Uses real double-dummy analysis for leads when `endplay` is installed; otherwise falls back to the standard bot automatically.",
  rl: "Uses a self-play-trained network when a checkpoint exists (see docs/rl_training.md); otherwise falls back to the standard bot automatically.",
};

function openNewGameModal() {
  document.getElementById("new-game-modal").hidden = false;
  updatePlayModeHint();
}

function closeNewGameModal() {
  document.getElementById("new-game-modal").hidden = true;
}

function updatePlayModeHint() {
  const mode = document.getElementById("play-mode-select").value;
  document.getElementById("play-mode-hint").textContent = PLAY_MODE_HINTS[mode] || "";
}

async function startNewGame() {
  const play_mode = document.getElementById("play-mode-select").value;
  try {
    const state = await apiPost("/api/new_session", { starting_board: 1, play_mode });
    lastFeedback = null;
    closeNewGameModal();
    render(state);
  } catch (e) {
    alert(e.message);
  }
}

document.getElementById("new-game-btn").addEventListener("click", openNewGameModal);
document.getElementById("cancel-new-game-btn").addEventListener("click", closeNewGameModal);
document.getElementById("start-new-game-btn").addEventListener("click", startNewGame);
document.getElementById("play-mode-select").addEventListener("change", updatePlayModeHint);

document.getElementById("undo-call-btn").addEventListener("click", undoCall);
document.getElementById("undo-card-btn").addEventListener("click", undoCard);
document.getElementById("analyze-btn").addEventListener("click", analyzePosition);
document.getElementById("restart-board-btn").addEventListener("click", restartBoard);
document.getElementById("next-board-btn").addEventListener("click", nextBoard);
document.getElementById("saved-games-btn").addEventListener("click", openSavedGamesModal);
document.getElementById("close-saved-games-btn").addEventListener("click", closeSavedGamesModal);
document.getElementById("save-as-btn").addEventListener("click", saveCurrentGame);

refresh();
