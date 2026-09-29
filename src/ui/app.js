/* AI Cofounder: Chat UI */
(function () {
  "use strict";

  // ── State ──
  var state = {
    currentSessionId: null,
    sessions: [],
    isLoading: false,
    progressTimer: null,
    progressStep: 0,
    currentUser: localStorage.getItem("aicofounder_user") || "founder",
  };

  var PROGRESS_MESSAGES = [
    "Loading context\u2026",
    "Routing to specialists\u2026",
    "Synthesizing response\u2026",
    "Almost there\u2026",
  ];
  var PROGRESS_INTERVAL = 5000;

  // ── DOM refs ──
  var $sidebar = document.getElementById("sidebar");
  var $sidebarOverlay = document.getElementById("sidebar-overlay");
  var $hamburger = document.getElementById("hamburger");
  var $newChatBtn = document.getElementById("new-chat-btn");
  var $sessionList = document.getElementById("session-list");
  var $messages = document.getElementById("messages");
  var $input = document.getElementById("message-input");
  var $sendBtn = document.getElementById("send-btn");
  var $errorToast = document.getElementById("error-toast");
  var $errorToastMsg = document.getElementById("error-toast-msg");
  var $errorToastClose = document.getElementById("error-toast-close");
  var $userSelect = document.getElementById("user-select");
  var $userAvatar = document.getElementById("sidebar-user-avatar");
  var $userName = document.getElementById("sidebar-user-name");

  // ── Markdown ──
  marked.setOptions({ breaks: true });

  function renderMarkdown(text) {
    return DOMPurify.sanitize(marked.parse(text || ""));
  }

  // ── Auth ──
  function getAuthHeader() {
    return "Bearer " + state.currentUser;
  }

  function setUser(userId) {
    state.currentUser = userId;
    localStorage.setItem("aicofounder_user", userId);
    if ($userSelect) $userSelect.value = userId;
    var initial = userId === "founder" ? "F" : "C";
    var name = userId === "founder" ? "Founder" : "Co-founder";
    if ($userAvatar) $userAvatar.textContent = initial;
    if ($userName) $userName.textContent = name;
  }

  // ── Error handling ──
  var lastFailedMessage = null;
  var errorDismissTimer = null;

  function showError(msg, retryable) {
    $errorToastMsg.textContent = msg;
    $errorToast.hidden = false;
    $errorToast.classList.add("error-enter");
    setTimeout(function () { $errorToast.classList.remove("error-enter"); }, 300);
    var retryBtn = document.getElementById("error-toast-retry");
    if (retryBtn) retryBtn.style.display = retryable ? "inline-block" : "none";
    clearTimeout(errorDismissTimer);
    errorDismissTimer = setTimeout(function () {
      $errorToast.hidden = true;
    }, retryable ? 15000 : 6000);
  }

  function hideError() {
    $errorToast.hidden = true;
    clearTimeout(errorDismissTimer);
  }

  $errorToastClose.addEventListener("click", hideError);

  var $retryBtn = document.getElementById("error-toast-retry");
  if ($retryBtn) {
    $retryBtn.addEventListener("click", function () {
      hideError();
      if (lastFailedMessage) {
        $input.value = lastFailedMessage;
        autoResize();
        updateSendButton();
        sendMessage();
      }
    });
  }

  // ── API helpers ──
  var API_TIMEOUT_MS = 120000;

  function fetchWithTimeout(url, options) {
    var controller = new AbortController();
    options.signal = controller.signal;
    var timer = setTimeout(function () { controller.abort(); }, API_TIMEOUT_MS);
    return fetch(url, options).finally(function () { clearTimeout(timer); });
  }

  function apiPost(path, body) {
    return fetchWithTimeout(window.location.origin + path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: getAuthHeader(),
      },
      body: JSON.stringify(body),
    }).then(function (res) {
      if (!res.ok) {
        return res.text().then(function (detail) {
          throw new Error("API error " + res.status + ": " + detail);
        });
      }
      return res.json();
    }).catch(function (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timed out after 120 seconds");
      }
      throw err;
    });
  }

  function apiGet(path) {
    return fetchWithTimeout(window.location.origin + path, {
      headers: { Authorization: getAuthHeader() },
    }).then(function (res) {
      if (!res.ok) {
        return res.text().then(function (detail) {
          throw new Error("API error " + res.status + ": " + detail);
        });
      }
      return res.json();
    }).catch(function (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timed out after 120 seconds");
      }
      throw err;
    });
  }

  // ── Sessions ──
  function saveSessionsToStorage() {
    try {
      localStorage.setItem("aicofounder_sessions", JSON.stringify(state.sessions));
    } catch (_) { /* quota exceeded */ }
  }

  function loadSessionsFromStorage() {
    try {
      var raw = localStorage.getItem("aicofounder_sessions");
      if (raw) state.sessions = JSON.parse(raw);
    } catch (_) {
      state.sessions = [];
    }
  }

  function addSession(id, preview) {
    if (state.sessions.some(function (s) { return s.id === id; })) return;
    state.sessions.unshift({
      id: id,
      preview: preview.slice(0, 60),
      timestamp: Date.now(),
    });
    saveSessionsToStorage();
    renderSessions();
  }

  function renderSessions() {
    $sessionList.innerHTML = "";
    state.sessions.forEach(function (s) {
      var li = document.createElement("li");
      li.className = "session-item" + (s.id === state.currentSessionId ? " active" : "");

      var previewSpan = document.createElement("span");
      previewSpan.textContent = s.preview || "New conversation";
      li.appendChild(previewSpan);

      var timeSpan = document.createElement("span");
      timeSpan.className = "session-item-time";
      timeSpan.textContent = formatTime(s.timestamp);
      li.appendChild(timeSpan);

      li.addEventListener("click", function () {
        loadSession(s.id);
      });
      $sessionList.appendChild(li);
    });
  }

  function formatTime(ts) {
    if (!ts) return "";
    var d = new Date(ts);
    // If timestamp is a string without timezone info, treat as UTC
    if (typeof ts === "string" && !ts.endsWith("Z") && !ts.includes("+")) {
      d = new Date(ts + "Z");
    }
    // No timeZone option: the browser's local timezone is used
    var opts = {};
    var now = new Date();
    if (d.toLocaleDateString("en-US", opts) === now.toLocaleDateString("en-US", opts)) {
      return d.toLocaleTimeString("en-US", Object.assign({ hour: "2-digit", minute: "2-digit" }, opts));
    }
    return d.toLocaleDateString("en-US", Object.assign({ month: "short", day: "numeric" }, opts));
  }

  function loadSession(sessionId) {
    if (state.isLoading) return;
    state.currentSessionId = sessionId;
    renderSessions();
    clearMessages();

    apiGet("/api/chat/sessions/" + sessionId)
      .then(function (data) {
        data.messages.forEach(function (msg) {
          appendMessage(msg.role, msg.content, null, null, null);
        });
        scrollToBottom();
      })
      .catch(function (err) {
        showError("Failed to load session: " + err.message);
      });
    closeSidebar();
  }

  // ── Messages ──
  function clearMessages() {
    $messages.innerHTML = "";
  }

  function hideWelcome() {
    var w = document.getElementById("welcome");
    if (w && w.parentNode) w.remove();
  }

  function appendMessage(role, content, agents, sources, agentOutputs) {
    hideWelcome();

    var row = document.createElement("div");
    row.className = "message-row " + role + " msg-enter";

    // Avatar
    var avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? (state.currentUser === "founder" ? "F" : "C") : "AI";
    row.appendChild(avatar);

    // Content wrapper
    var contentWrapper = document.createElement("div");
    contentWrapper.className = "message-content";

    // Bubble
    var bubble = document.createElement("div");
    bubble.className = "message-bubble";
    if (role === "assistant") {
      bubble.innerHTML = renderMarkdown(content);
      addCopyButtons(bubble);
    } else {
      bubble.textContent = content;
    }
    contentWrapper.appendChild(bubble);

    // Agent badges (expandable)
    if (agents && agents.length > 0) {
      var badges = document.createElement("div");
      badges.className = "agent-badges";
      agents.forEach(function (agentId, idx) {
        var badge = document.createElement("span");
        badge.className = "agent-badge badge-enter";
        badge.setAttribute("data-agent", agentId);
        badge.textContent = formatAgentName(agentId);
        // Stagger badge fade-in
        badge.style.animationDelay = (idx * 60) + "ms";

        // Tooltip on hover
        badge.setAttribute("data-tooltip", formatAgentName(agentId) + " Specialist");

        var agentOutput = agentOutputs ? findAgentOutput(agentOutputs, agentId) : null;
        if (agentOutput) {
          badge.classList.add("expandable");
          badge.setAttribute("data-tooltip", formatAgentName(agentId) + " — click to expand");
          badge.addEventListener("click", function () {
            toggleAgentDetail(badge, agentId, agentOutput);
          });
        }

        badges.appendChild(badge);
      });
      contentWrapper.appendChild(badges);

      var detailContainer = document.createElement("div");
      detailContainer.className = "agent-detail-container";
      contentWrapper.appendChild(detailContainer);
    }

    // Sources
    if (sources && sources.length > 0) {
      var toggle = document.createElement("button");
      toggle.className = "sources-toggle";
      toggle.innerHTML = '<span class="arrow">&#9654;</span> ' + sources.length + " source" + (sources.length > 1 ? "s" : "");

      var list = document.createElement("div");
      list.className = "sources-list";

      sources.forEach(function (src) {
        var card = document.createElement("div");
        card.className = "source-card";

        var header = document.createElement("div");
        header.className = "source-card-header";

        var nameEl = document.createElement("span");
        var displayName = src.title || src.source_name || "Unknown";
        nameEl.innerHTML =
          '<span class="source-name">' + escapeHtml(displayName) + "</span>" +
          (src.source_type ? '<span class="source-type">' + escapeHtml(src.source_type) + "</span>" : "");
        header.appendChild(nameEl);

        if (src.score != null) {
          var scoreEl = document.createElement("span");
          scoreEl.className = "source-score";
          scoreEl.textContent = (src.score * 100).toFixed(0) + "%";
          header.appendChild(scoreEl);
        }

        card.appendChild(header);

        if (src.score != null) {
          var bar = document.createElement("div");
          bar.className = "source-score-bar";
          var fill = document.createElement("div");
          fill.className = "source-score-fill";
          fill.style.width = (src.score * 100).toFixed(0) + "%";
          bar.appendChild(fill);
          card.appendChild(bar);
        }

        list.appendChild(card);
      });

      toggle.addEventListener("click", function () {
        toggle.classList.toggle("open");
        list.classList.toggle("show");
      });

      contentWrapper.appendChild(toggle);
      contentWrapper.appendChild(list);
    }

    row.appendChild(contentWrapper);
    $messages.appendChild(row);
    return row;
  }

  // ── Agent detail expand/collapse ──
  function findAgentOutput(outputs, agentId) {
    for (var i = 0; i < outputs.length; i++) {
      if (outputs[i].agent_id === agentId) return outputs[i];
    }
    return null;
  }

  function toggleAgentDetail(badge, agentId, output) {
    var container = badge.closest(".message-content").querySelector(".agent-detail-container");
    var existing = container.querySelector('[data-detail-agent="' + agentId + '"]');
    if (existing) {
      existing.classList.add("detail-collapse");
      setTimeout(function () { existing.remove(); }, 200);
      badge.classList.remove("active");
      return;
    }

    badge.classList.add("active");
    var detail = document.createElement("div");
    detail.className = "agent-detail detail-expand";
    detail.setAttribute("data-detail-agent", agentId);

    var header = document.createElement("div");
    header.className = "agent-detail-header";
    header.innerHTML =
      '<span class="agent-badge" data-agent="' + agentId + '">' + escapeHtml(formatAgentName(agentId)) + "</span>" +
      (output.confidence ? '<span class="agent-confidence">Confidence: ' + escapeHtml(output.confidence) + "</span>" : "");
    detail.appendChild(header);

    var body = document.createElement("div");
    body.className = "agent-detail-body";
    body.innerHTML = renderMarkdown(output.response);
    addCopyButtons(body);
    detail.appendChild(body);

    container.appendChild(detail);
    scrollToBottom();
  }

  // ── Loading indicator with progress messages ──
  function startLoadingIndicator() {
    hideWelcome();
    state.progressStep = 0;

    var row = document.createElement("div");
    row.className = "message-row assistant msg-enter";
    row.id = "typing-indicator";

    var avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "Z";
    row.appendChild(avatar);

    var content = document.createElement("div");
    content.className = "message-content";

    var bubble = document.createElement("div");
    bubble.className = "message-bubble loading-bubble";

    var dots = document.createElement("div");
    dots.className = "typing-indicator";
    dots.innerHTML = "<span></span><span></span><span></span>";
    bubble.appendChild(dots);

    var statusText = document.createElement("span");
    statusText.className = "loading-status";
    statusText.id = "loading-status";
    statusText.textContent = PROGRESS_MESSAGES[0];
    bubble.appendChild(statusText);

    content.appendChild(bubble);
    row.appendChild(content);
    $messages.appendChild(row);
    scrollToBottom();

    state.progressTimer = setInterval(function () {
      state.progressStep++;
      var el = document.getElementById("loading-status");
      if (el) {
        el.textContent = PROGRESS_MESSAGES[state.progressStep % PROGRESS_MESSAGES.length];
      }
    }, PROGRESS_INTERVAL);
  }

  function stopLoadingIndicator() {
    if (state.progressTimer) {
      clearInterval(state.progressTimer);
      state.progressTimer = null;
    }
    var el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  function updateLoadingStatus(statusText) {
    // Update the loading indicator with real status from the streaming pipeline
    var el = document.getElementById("loading-status");
    if (el) {
      el.textContent = statusText;
      // Reset the fake progress timer since we have real updates
      if (state.progressTimer) {
        clearInterval(state.progressTimer);
        state.progressTimer = null;
      }
    }
  }

  function scrollToBottom() {
    requestAnimationFrame(function () {
      $messages.scrollTop = $messages.scrollHeight;
    });
  }

  // ── Send message ──
  function sendMessage() {
    var text = $input.value.trim();
    if (!text || state.isLoading) return;

    state.isLoading = true;
    $sendBtn.disabled = true;
    $input.value = "";
    lastFailedMessage = null;
    autoResize();

    // Subtle send feedback
    $sendBtn.classList.add("transmit-flash");
    setTimeout(function () { $sendBtn.classList.remove("transmit-flash"); }, 200);

    appendMessage("user", text, null, null, null);
    scrollToBottom();
    startLoadingIndicator();

    var body = { message: text };
    if (state.currentSessionId) body.session_id = state.currentSessionId;

    // Use SSE streaming endpoint for real-time status updates
    fetch(window.location.origin + "/api/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: getAuthHeader(),
      },
      body: JSON.stringify(body),
    })
      .then(function (res) {
        if (!res.ok) throw new Error("Stream failed: " + res.status);
        var reader = res.body.getReader();
        var decoder = new TextDecoder();
        var buffer = "";
        var streamingBubble = null; // The message bubble receiving streamed tokens
        var streamedText = "";      // Accumulated raw text for final markdown render

        function processStream() {
          return reader.read().then(function (result) {
            if (result.done) {
              if (state.isLoading) {
                stopLoadingIndicator();
                lastFailedMessage = text;
                showError("Stream ended unexpectedly", true);
                state.isLoading = false;
                updateSendButton();
                $input.focus();
              }
              return;
            }

            buffer += decoder.decode(result.value, { stream: true });

            var lines = buffer.split("\n");
            buffer = lines.pop();

            var eventType = null;
            var eventData = null;

            for (var i = 0; i < lines.length; i++) {
              var line = lines[i];
              if (line.startsWith("event: ")) {
                eventType = line.substring(7).trim();
              } else if (line.startsWith("data: ")) {
                eventData = line.substring(6);

                if (eventType === "status") {
                  try {
                    var statusObj = JSON.parse(eventData);
                    updateLoadingStatus(statusObj.status);
                  } catch (e) { /* ignore parse errors */ }

                } else if (eventType === "token") {
                  try {
                    var tokenObj = JSON.parse(eventData);
                    var tokenText = tokenObj.token || "";
                    streamedText += tokenText;

                    // Create the assistant message row on first token
                    if (!streamingBubble) {
                      stopLoadingIndicator();
                      hideWelcome();
                      var row = document.createElement("div");
                      row.className = "message-row assistant msg-enter";

                      var avatar = document.createElement("div");
                      avatar.className = "message-avatar";
                      avatar.textContent = "Z";
                      row.appendChild(avatar);

                      var contentWrapper = document.createElement("div");
                      contentWrapper.className = "message-content";

                      streamingBubble = document.createElement("div");
                      streamingBubble.className = "message-bubble";
                      contentWrapper.appendChild(streamingBubble);
                      row.appendChild(contentWrapper);
                      $messages.appendChild(row);
                    }

                    // Append token text and render markdown progressively
                    streamingBubble.innerHTML = renderMarkdown(streamedText);
                    scrollToBottom();
                  } catch (e) { /* ignore parse errors */ }

                } else if (eventType === "done") {
                  try {
                    var data = JSON.parse(eventData);

                    state.currentSessionId = data.session_id;
                    addSession(data.session_id, text);
                    renderSessions();

                    if (streamingBubble) {
                      // Final markdown render + copy buttons
                      streamingBubble.innerHTML = renderMarkdown(streamedText);
                      addCopyButtons(streamingBubble);

                      // Add agent badges and sources to the existing message row
                      var contentWrapper = streamingBubble.parentNode;
                      if (data.agents_consulted && data.agents_consulted.length > 0) {
                        var badges = document.createElement("div");
                        badges.className = "agent-badges";
                        data.agents_consulted.forEach(function (agentId, idx) {
                          var badge = document.createElement("span");
                          badge.className = "agent-badge badge-enter";
                          badge.setAttribute("data-agent", agentId);
                          badge.textContent = formatAgentName(agentId);
                          badge.style.animationDelay = (idx * 60) + "ms";

                          if (data.agent_outputs) {
                            var agentOutput = data.agent_outputs.find(function (o) { return o.agent_id === agentId; });
                            if (agentOutput) {
                              badge.setAttribute("data-tooltip", agentOutput.agent_name);
                              badge.addEventListener("click", (function (ao) {
                                return function () {
                                  toggleAgentDetail(badge, ao);
                                };
                              })(agentOutput));
                            }
                          }

                          badges.appendChild(badge);
                        });
                        contentWrapper.appendChild(badges);
                      }
                      scrollToBottom();
                    } else {
                      // No tokens received — render from done data directly
                      stopLoadingIndicator();
                      appendMessage(
                        "assistant",
                        data.response || streamedText,
                        data.agents_consulted,
                        data.sources,
                        data.agent_outputs || null
                      );
                      scrollToBottom();
                    }
                  } catch (e) {
                    stopLoadingIndicator();
                    showError("Failed to parse response", false);
                  }
                  state.isLoading = false;
                  updateSendButton();
                  $input.focus();
                  return; // done

                } else if (eventType === "error") {
                  try {
                    var errObj = JSON.parse(eventData);
                    stopLoadingIndicator();
                    lastFailedMessage = text;
                    showError("Error: " + (errObj.detail || "Unknown"), true);
                  } catch (e) {
                    stopLoadingIndicator();
                    showError("Unknown error", true);
                  }
                  state.isLoading = false;
                  updateSendButton();
                  $input.focus();
                  return; // done
                }

                eventType = null;
                eventData = null;
              }
            }

            return processStream();
          });
        }

        return processStream();
      })
      .catch(function (err) {
        stopLoadingIndicator();
        lastFailedMessage = text;
        showError("Failed to send: " + err.message, true);
        state.isLoading = false;
        updateSendButton();
        $input.focus();
      });
  }

  // ── Input ──
  function autoResize() {
    $input.style.height = "auto";
    $input.style.height = Math.min($input.scrollHeight, 160) + "px";
  }

  function updateSendButton() {
    $sendBtn.disabled = !$input.value.trim() || state.isLoading;
  }

  $input.addEventListener("input", function () {
    autoResize();
    updateSendButton();
  });

  $input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      sendMessage();
    }
    if (e.key === "Escape") {
      $input.value = "";
      autoResize();
      updateSendButton();
    }
  });

  $sendBtn.addEventListener("click", function () {
    sendMessage();
  });

  // ── New chat ──
  $newChatBtn.addEventListener("click", function () {
    state.currentSessionId = null;
    $messages.innerHTML = "";
    var welcome = document.createElement("div");
    welcome.id = "welcome";
    welcome.className = "welcome";
    welcome.innerHTML =
      '<div class="welcome-brand" aria-hidden="true">' +
        '<div class="welcome-logo-ring">' +
          '<div class="welcome-logo-inner">' +
            '<svg width="32" height="32" viewBox="0 0 24 24" fill="none">' +
              '<path d="M12 2L3 7v10l9 5 9-5V7l-9-5z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>' +
              '<path d="M12 12L3 7m9 5l9-5m-9 5v10" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>' +
            '</svg>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<h2 class="welcome-headline">What are we building today?</h2>' +
      '<p class="welcome-sub">Your AI cofounder team is ready. 11 agents standing by for strategy, compliance, fundraising, GTM, and beyond.</p>' +
      '<div class="suggested-prompts">' +
        '<button class="suggested-prompt" data-prompt="What should our GTM strategy look like for our first customer segment?">' +
          '<span class="prompt-label">GTM Strategy</span>' +
          '<span class="prompt-detail">First customer segment</span>' +
        '</button>' +
        '<button class="suggested-prompt" data-prompt="Help me think through our pricing model">' +
          '<span class="prompt-label">Pricing Model</span>' +
          '<span class="prompt-detail">Pricing unit and first tests</span>' +
        '</button>' +
        '<button class="suggested-prompt" data-prompt="Help me prepare customer interview questions for my next call">' +
          '<span class="prompt-label">Customer Interviews</span>' +
          '<span class="prompt-detail">Discovery call preparation</span>' +
        '</button>' +
        '<button class="suggested-prompt" data-prompt="Analyze our current runway and suggest fundraising timing">' +
          '<span class="prompt-label">Runway Analysis</span>' +
          '<span class="prompt-detail">Fundraising timing &amp; strategy</span>' +
        '</button>' +
      '</div>';
    $messages.appendChild(welcome);
    bindSuggestedPrompts();
    renderSessions();
    $input.focus();
    closeSidebar();
  });

  // ── User selector ──
  if ($userSelect) {
    $userSelect.value = state.currentUser;
    $userSelect.addEventListener("change", function () {
      setUser($userSelect.value);
    });
  }

  // ── Suggested prompts ──
  function bindSuggestedPrompts() {
    document.querySelectorAll(".suggested-prompt").forEach(function (btn) {
      btn.addEventListener("click", function () {
        $input.value = btn.getAttribute("data-prompt");
        autoResize();
        updateSendButton();
        sendMessage();
      });
    });
  }

  // ── Sidebar toggle (mobile) ──
  function openSidebar() {
    $sidebar.classList.add("open");
    $sidebarOverlay.classList.add("show");
  }

  function closeSidebar() {
    $sidebar.classList.remove("open");
    $sidebarOverlay.classList.remove("show");
  }

  $hamburger.addEventListener("click", function () {
    if ($sidebar.classList.contains("open")) {
      closeSidebar();
    } else {
      openSidebar();
    }
  });

  $sidebarOverlay.addEventListener("click", closeSidebar);

  // ── Utilities ──
  function formatAgentName(id) {
    var names = {
      gtm: "GTM",
      finance: "Finance",
      marketing: "Marketing",
      fintech: "Fintech",
      healthcare: "Healthcare",
      product: "Product",
      legal: "Legal",
      data_analytics: "Data & Analytics",
      bizdev: "BizDev",
      cofounder: "Cofounder",
    };
    return names[id] || id;
  }

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function addCopyButtons(container) {
    container.querySelectorAll("pre").forEach(function (pre) {
      var btn = document.createElement("button");
      btn.className = "copy-code-btn";
      btn.textContent = "Copy";
      btn.addEventListener("click", function () {
        var code = pre.querySelector("code");
        var text = code ? code.textContent : pre.textContent;
        navigator.clipboard.writeText(text).then(function () {
          btn.textContent = "Copied";
          btn.classList.add("copied");
          btn.classList.add("copy-flash");
          setTimeout(function () { btn.classList.remove("copy-flash"); }, 150);
          setTimeout(function () {
            btn.textContent = "Copy";
            btn.classList.remove("copied");
          }, 2000);
        });
      });
      pre.appendChild(btn);
    });
  }

  // ── Micro-interactions ──
  document.addEventListener("mousedown", function (e) {
    var btn = e.target.closest("button");
    if (btn) {
      btn.classList.add("click-press");
      var remove = function () {
        btn.classList.remove("click-press");
        btn.removeEventListener("mouseup", remove);
        btn.removeEventListener("mouseleave", remove);
      };
      btn.addEventListener("mouseup", remove);
      btn.addEventListener("mouseleave", remove);
    }
  });

  // ── Init ──
  // Load sessions from API (persisted in DB), fall back to localStorage
  function loadSessionsFromAPI() {
    apiGet("/api/chat/sessions/")
      .then(function (sessions) {
        if (sessions && sessions.length > 0) {
          state.sessions = sessions.map(function (s) {
            return {
              id: s.session_id,
              preview: (s.first_message || "New session").slice(0, 60),
              timestamp: s.created_at ? new Date(s.created_at + (s.created_at.endsWith("Z") ? "" : "Z")).getTime() : Date.now(),
            };
          });
          saveSessionsToStorage();
          renderSessions();
        }
      })
      .catch(function () {
        // API unavailable — use localStorage fallback
      });
  }

  loadSessionsFromStorage();
  renderSessions();
  loadSessionsFromAPI();
  setUser(state.currentUser);
  bindSuggestedPrompts();
  $input.focus();
})();
