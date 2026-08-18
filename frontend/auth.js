/* ═══════════════════════════════════════════════════════════════════════════
   DrugBot — Auth Page Logic
   Handles login, signup, tab switching, password strength, and JWT storage.
   ═══════════════════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ── Theme initialization for Auth page ──
(function initAuthTheme() {
  const pref = localStorage.getItem("drugbot_theme") || "system";
  let theme = pref;
  if (pref === "system") {
    theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  document.documentElement.setAttribute("data-theme", theme);
})();

// ── Check if already authenticated ──
(function checkAuth() {
  const token = localStorage.getItem("drugbot_token");
  if (token) {
    // Verify token is still valid
    fetch(`${API_BASE}/api/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => {
        if (r.ok) window.location.href = "/frontend/index.html";
      })
      .catch(() => {});
  }
})();

// ══════════════════════════════════════════════════════════════════════════
//  DOM REFS
// ══════════════════════════════════════════════════════════════════════════
const $tabLogin = document.getElementById("tab-login");
const $tabSignup = document.getElementById("tab-signup");
const $tabIndicator = document.getElementById("tab-indicator");
const $panelLogin = document.getElementById("panel-login");
const $panelSignup = document.getElementById("panel-signup");

const $loginForm = document.getElementById("login-form");
const $loginEmail = document.getElementById("login-email");
const $loginPassword = document.getElementById("login-password");
const $loginAlert = document.getElementById("login-alert");
const $btnLogin = document.getElementById("btn-login");

const $signupForm = document.getElementById("signup-form");
const $signupEmail = document.getElementById("signup-email");
const $signupPassword = document.getElementById("signup-password");
const $signupConfirm = document.getElementById("signup-confirm");
const $signupAlert = document.getElementById("signup-alert");
const $btnSignup = document.getElementById("btn-signup");

// ══════════════════════════════════════════════════════════════════════════
//  TAB SWITCHING
// ══════════════════════════════════════════════════════════════════════════
$tabLogin.addEventListener("click", () => switchTab("login"));
$tabSignup.addEventListener("click", () => switchTab("signup"));

function switchTab(tab) {
  const isLogin = tab === "login";

  $tabLogin.classList.toggle("active", isLogin);
  $tabSignup.classList.toggle("active", !isLogin);
  $tabLogin.setAttribute("aria-selected", isLogin);
  $tabSignup.setAttribute("aria-selected", !isLogin);

  $panelLogin.classList.toggle("active", isLogin);
  $panelSignup.classList.toggle("active", !isLogin);

  $tabIndicator.classList.toggle("right", !isLogin);

  // Clear alerts
  hideAlert($loginAlert);
  hideAlert($signupAlert);
}

// ══════════════════════════════════════════════════════════════════════════
//  PASSWORD VISIBILITY TOGGLE
// ══════════════════════════════════════════════════════════════════════════
document.getElementById("toggle-login-pw").addEventListener("click", () => {
  togglePasswordVisibility($loginPassword);
});

document.getElementById("toggle-signup-pw").addEventListener("click", () => {
  togglePasswordVisibility($signupPassword);
});

function togglePasswordVisibility(input) {
  input.type = input.type === "password" ? "text" : "password";
}

// ══════════════════════════════════════════════════════════════════════════
//  PASSWORD STRENGTH METER
// ══════════════════════════════════════════════════════════════════════════
const strengthBars = [
  document.getElementById("str-bar-1"),
  document.getElementById("str-bar-2"),
  document.getElementById("str-bar-3"),
  document.getElementById("str-bar-4"),
];
const $strengthLabel = document.getElementById("strength-label");

const rules = {
  length: document.getElementById("rule-length"),
  upper: document.getElementById("rule-upper"),
  lower: document.getElementById("rule-lower"),
  digit: document.getElementById("rule-digit"),
  special: document.getElementById("rule-special"),
};

$signupPassword.addEventListener("input", () => {
  const pw = $signupPassword.value;
  updateStrength(pw);
  updateRules(pw);
});

function updateStrength(pw) {
  let score = 0;
  if (pw.length >= 8) score++;
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score++;
  if (/[0-9]/.test(pw)) score++;
  if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(pw)) score++;

  const levels = ["", "weak", "fair", "good", "strong"];
  const labels = ["", "Weak", "Fair", "Good", "Strong"];
  const level = levels[score] || "";

  strengthBars.forEach((bar, i) => {
    bar.className = "strength-bar";
    if (i < score) bar.classList.add(level);
  });

  $strengthLabel.textContent = labels[score] || "";
  $strengthLabel.className = "strength-label " + level;
}

function updateRules(pw) {
  setRule(rules.length, pw.length >= 8);
  setRule(rules.upper, /[A-Z]/.test(pw));
  setRule(rules.lower, /[a-z]/.test(pw));
  setRule(rules.digit, /[0-9]/.test(pw));
  setRule(rules.special, /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(pw));
}

function setRule(el, pass) {
  el.classList.toggle("pass", pass);
  el.querySelector(".rule-icon").textContent = pass ? "✓" : "○";
}

// ══════════════════════════════════════════════════════════════════════════
//  ALERTS
// ══════════════════════════════════════════════════════════════════════════
function showAlert(el, message, type = "error") {
  el.textContent = message;
  el.className = `auth-alert ${type}`;
  el.classList.remove("hidden");
}

function hideAlert(el) {
  el.classList.add("hidden");
}

// ══════════════════════════════════════════════════════════════════════════
//  BUTTON LOADING STATE
// ══════════════════════════════════════════════════════════════════════════
function setLoading(btn, loading) {
  const text = btn.querySelector(".btn-text");
  const spinner = btn.querySelector(".btn-spinner");
  btn.disabled = loading;
  text.style.opacity = loading ? "0" : "1";
  spinner.classList.toggle("hidden", !loading);
}

// ══════════════════════════════════════════════════════════════════════════
//  LOGIN
// ══════════════════════════════════════════════════════════════════════════
$loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideAlert($loginAlert);

  const email = $loginEmail.value.trim();
  const password = $loginPassword.value;

  if (!email || !password) {
    showAlert($loginAlert, "Please fill in all fields.");
    return;
  }

  setLoading($btnLogin, true);

  try {
    const res = await fetch(`${API_BASE}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await res.json();

    if (!res.ok) {
      const type = res.status === 429 ? "warning" : "error";
      showAlert($loginAlert, data.detail || "Login failed.", type);
      return;
    }

    // Store token and redirect
    localStorage.setItem("drugbot_token", data.token);
    localStorage.setItem("drugbot_user_email", data.email);
    showAlert($loginAlert, "Login successful! Redirecting…", "success");

    setTimeout(() => {
      window.location.href = "/frontend/index.html";
    }, 600);
  } catch (err) {
    showAlert($loginAlert, "Network error. Is the backend running?");
  } finally {
    setLoading($btnLogin, false);
  }
});

// ══════════════════════════════════════════════════════════════════════════
//  SIGNUP
// ══════════════════════════════════════════════════════════════════════════
$signupForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideAlert($signupAlert);

  const email = $signupEmail.value.trim();
  const password = $signupPassword.value;
  const confirm = $signupConfirm.value;

  if (!email || !password || !confirm) {
    showAlert($signupAlert, "Please fill in all fields.");
    return;
  }

  if (password !== confirm) {
    showAlert($signupAlert, "Passwords do not match.");
    return;
  }

  // Client-side validation (server validates too)
  if (password.length < 8) {
    showAlert($signupAlert, "Password must be at least 8 characters.");
    return;
  }

  setLoading($btnSignup, true);

  try {
    const res = await fetch(`${API_BASE}/api/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await res.json();

    if (!res.ok) {
      showAlert($signupAlert, data.detail || "Signup failed.");
      return;
    }

    // Store token and redirect
    localStorage.setItem("drugbot_token", data.token);
    localStorage.setItem("drugbot_user_email", data.email);
    showAlert($signupAlert, "Account created! Redirecting…", "success");

    setTimeout(() => {
      window.location.href = "/frontend/index.html";
    }, 600);
  } catch (err) {
    showAlert($signupAlert, "Network error. Is the backend running?");
  } finally {
    setLoading($btnSignup, false);
  }
});
