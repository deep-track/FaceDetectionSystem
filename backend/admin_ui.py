ADMIN_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepTrack — Admin</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #080b0f;
    --surface:  #0d1117;
    --surface2: #131920;
    --border:   #1c2530;
    --border2:  #243040;
    --text:     #c8d8e8;
    --muted:    #4a6070;
    --accent:   #e8ff47;
    --accent2:  #b8cc2a;
    --red:      #ff3355;
    --green:    #00e87a;
    --mono:     'DM Mono', monospace;
    --display:  'Syne', sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--mono); font-size: 13px; min-height: 100vh; }

  #auth-screen {
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background-image: linear-gradient(rgba(232,255,71,.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(232,255,71,.025) 1px, transparent 1px);
    background-size: 48px 48px;
  }
  .auth-box { border: 1px solid var(--border2); background: var(--surface); padding: 48px 52px; width: 100%; max-width: 420px; }
  .auth-wordmark { font-family: var(--display); font-size: 1.6rem; font-weight: 800; letter-spacing: -.02em; margin-bottom: 4px; }
  .auth-wordmark span { color: var(--accent); }
  .auth-sub { font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-bottom: 36px; }
  .field-label { font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; display: block; }

  input[type="password"], input[type="text"], input[type="number"], select {
    width: 100%; background: var(--bg); border: 1px solid var(--border2); color: var(--text);
    font-family: var(--mono); font-size: 13px; padding: 10px 14px; outline: none;
    transition: border-color .15s; appearance: none; -webkit-appearance: none;
  }
  input:focus, select:focus { border-color: var(--accent); }
  select {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='none'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%234a6070' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 12px center; padding-right: 32px; cursor: pointer;
  }

  .btn { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: .1em; text-transform: uppercase; border: none; padding: 10px 20px; cursor: pointer; transition: background .15s, opacity .15s; }
  .btn-accent        { background: var(--accent);  color: #000; }
  .btn-accent:hover  { background: var(--accent2); }
  .btn-ghost         { background: var(--border2); color: var(--text); }
  .btn-ghost:hover   { background: #2a3a4a; }
  .btn-danger        { background: rgba(255,51,85,.15); color: var(--red);   border: 1px solid rgba(255,51,85,.3); }
  .btn-danger:hover  { background: rgba(255,51,85,.25); }
  .btn-success       { background: rgba(0,232,122,.12); color: var(--green); border: 1px solid rgba(0,232,122,.25); }
  .btn-success:hover { background: rgba(0,232,122,.2); }
  .btn-full          { width: 100%; margin-top: 20px; padding: 12px; font-size: 12px; }
  .btn:disabled      { opacity: .4; cursor: not-allowed; }
  .auth-error { font-size: 11px; color: var(--red); margin-top: 12px; min-height: 16px; }

  #main-screen { display: none; }
  header { border-bottom: 1px solid var(--border); padding: 16px 40px; display: flex; align-items: center; gap: 20px; background: var(--surface); position: sticky; top: 0; z-index: 10; }
  .wordmark { font-family: var(--display); font-size: 1.1rem; font-weight: 800; letter-spacing: -.02em; }
  .wordmark span { color: var(--accent); }
  .header-badge { font-size: .62rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); border: 1px solid var(--border2); padding: 3px 8px; }
  .header-right { margin-left: auto; display: flex; align-items: center; gap: 16px; }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 6px var(--green); display: inline-block; margin-right: 6px; }

  main { max-width: 1200px; margin: 0 auto; padding: 40px 32px; }
  .section-title { font-family: var(--display); font-size: .65rem; letter-spacing: .2em; text-transform: uppercase; color: var(--muted); margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }

  .create-panel { background: var(--surface); border: 1px solid var(--border); padding: 28px 32px; margin-bottom: 40px; }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr 120px 120px; gap: 16px; align-items: end; margin-top: 20px; }
  .form-field { display: flex; flex-direction: column; gap: 8px; }

  .table-wrap { background: var(--surface); border: 1px solid var(--border); overflow: hidden; overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; min-width: 960px; }
  thead tr { border-bottom: 1px solid var(--border2); }
  th { font-size: .62rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); padding: 12px 16px; text-align: left; font-weight: 500; white-space: nowrap; }
  tbody tr { border-bottom: 1px solid var(--border); transition: background .1s; }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: var(--surface2); }
  td { padding: 14px 16px; color: var(--text); font-size: 12px; }

  .key-value { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: .04em; }
  .pill { display: inline-block; font-size: .6rem; font-weight: 500; letter-spacing: .1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; }
  .pill-active   { background: rgba(0,232,122,.12); color: var(--green); }
  .pill-inactive { background: rgba(255,51,85,.12);  color: var(--red); }
  .td-actions { display: flex; gap: 8px; align-items: center; }

  .limit-input { width: 90px; background: var(--bg); border: 1px solid var(--border2); color: var(--text); font-family: var(--mono); font-size: 12px; padding: 5px 8px; outline: none; }
  .limit-input:focus { border-color: var(--accent); }

  .usage-wrap { display: flex; align-items: center; gap: 8px; min-width: 160px; }
  .usage-track { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .usage-fill  { height: 100%; border-radius: 2px; background: var(--accent); transition: width .4s ease; }
  .usage-fill.warn   { background: #ffaa00; }
  .usage-fill.danger { background: var(--red); }
  .usage-label { font-size: 10px; color: var(--muted); white-space: nowrap; min-width: 70px; text-align: right; }

  #toast { position: fixed; bottom: 32px; right: 32px; background: var(--surface2); border: 1px solid var(--border2); padding: 12px 20px; font-size: 12px; letter-spacing: .04em; opacity: 0; transform: translateY(8px); transition: opacity .2s, transform .2s; pointer-events: none; z-index: 100; max-width: 360px; }
  #toast.show { opacity: 1; transform: translateY(0); }
  #toast.ok  { border-color: rgba(0,232,122,.4); color: var(--green); }
  #toast.err { border-color: rgba(255,51,85,.4);  color: var(--red); }

  .key-reveal { display: none; margin-top: 20px; background: var(--bg); border: 1px solid rgba(232,255,71,.3); padding: 16px 20px; }
  .key-reveal-label { font-size: .65rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin-bottom: 8px; }
  .key-reveal-value { font-family: var(--mono); font-size: 12px; color: var(--text); word-break: break-all; cursor: pointer; }
  .key-reveal-hint  { font-size: .65rem; color: var(--muted); margin-top: 8px; }

  .empty-state { padding: 48px; text-align: center; color: var(--muted); font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }
  .spinner-inline { display: inline-block; width: 12px; height: 12px; border: 1.5px solid var(--border2); border-top-color: var(--accent); border-radius: 50%; animation: spin .6s linear infinite; margin-right: 6px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div id="auth-screen">
  <div class="auth-box">
    <div class="auth-wordmark">Deep<span>Track</span></div>
    <div class="auth-sub">Admin Dashboard</div>
    <label class="field-label">Admin Secret</label>
    <input type="password" id="secret-input" placeholder="Enter ADMIN_SECRET"
           onkeydown="if(event.key==='Enter') doLogin()">
    <button class="btn btn-accent btn-full" onclick="doLogin()">Authenticate</button>
    <div class="auth-error" id="auth-error"></div>
  </div>
</div>

<div id="main-screen">
  <header>
    <div class="wordmark">Deep<span>Track</span></div>
    <div class="header-badge">Admin</div>
    <div class="header-right">
      <span class="status-dot"></span>
      <span style="font-size:.68rem;letter-spacing:.08em;color:var(--muted);text-transform:uppercase">Connected</span>
      <button class="btn btn-ghost" style="font-size:10px;padding:6px 14px" onclick="logout()">Sign out</button>
    </div>
  </header>

  <main>
    <div class="section-title">// Generate API Key</div>
    <div class="create-panel">
      <div class="form-grid">
        <div class="form-field">
          <label class="field-label">Owner / Company</label>
          <input type="text" id="new-owner" placeholder="e.g. Gotham Media">
        </div>
        <div class="form-field">
          <label class="field-label">User</label>
          <select id="new-user"><option value="">Loading users...</option></select>
        </div>
        <div class="form-field">
          <label class="field-label">Monthly Limit</label>
          <input type="number" id="new-limit" value="100" min="1">
        </div>
        <button class="btn btn-accent" style="height:41px" onclick="createKey()">Generate</button>
      </div>
      <div class="form-field" style="margin-top:16px">
        <label class="field-label">Notes (optional)</label>
        <input type="text" id="new-notes" placeholder="e.g. Gotham enterprise — March 2026">
      </div>
      <div class="key-reveal" id="key-reveal">
        <div class="key-reveal-label">⚡ New API Key — copy now, shown once</div>
        <div class="key-reveal-value" id="key-reveal-value" onclick="copyKey()" title="Click to copy"></div>
        <div class="key-reveal-hint">Click to copy · Share securely with the client</div>
      </div>
    </div>

    <div class="section-title">// Active Keys</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Owner</th>
            <th>API Key</th>
            <th>Monthly Limit</th>
            <th>Monthly Usage</th>
            <th>Status</th>
            <th>Created</th>
            <th>Notes</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="keys-tbody">
          <tr><td colspan="8" class="empty-state">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </main>
</div>

<div id="toast"></div>

<script>
  let SECRET = '';

  async function doLogin() {
    const s = document.getElementById('secret-input').value.trim();
    if (!s) return;
    try {
      const r = await fetch('/admin/keys', { headers: { 'X-Admin-Secret': s } });
      if (r.status === 403) { document.getElementById('auth-error').textContent = 'Incorrect secret.'; return; }
      SECRET = s;
      document.getElementById('auth-screen').style.display = 'none';
      document.getElementById('main-screen').style.display = 'block';
      loadUsers();
      loadKeys();
    } catch (e) {
      document.getElementById('auth-error').textContent = 'Could not reach API.';
    }
  }

  function logout() {
    SECRET = '';
    document.getElementById('auth-screen').style.display = 'flex';
    document.getElementById('main-screen').style.display = 'none';
    document.getElementById('secret-input').value = '';
    document.getElementById('auth-error').textContent = '';
  }

  let _toastTimer;
  function toast(msg, type = 'ok') {
    const el = document.getElementById('toast');
    el.textContent = msg; el.className = 'show ' + type;
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => el.className = '', 3000);
  }

  async function loadUsers() {
    const sel = document.getElementById('new-user');
    try {
      const r    = await fetch('/admin/users', { headers: { 'X-Admin-Secret': SECRET } });
      const data = await r.json();
      sel.innerHTML = '';
      if (!data.users || !data.users.length) { sel.innerHTML = '<option value="">No users found</option>'; return; }
      data.users.forEach(u => {
        const opt = document.createElement('option');
        opt.value = u.id; opt.textContent = u.name + ' (' + u.email + ')';
        sel.appendChild(opt);
      });
    } catch (e) { sel.innerHTML = '<option value="">Failed to load users</option>'; }
  }

  async function loadKeys() {
    const tbody = document.getElementById('keys-tbody');
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state"><span class="spinner-inline"></span>Loading...</td></tr>';
    try {
      const [keysRes, usageRes] = await Promise.all([
        fetch('/admin/keys',        { headers: { 'X-Admin-Secret': SECRET } }),
        fetch('/admin/usage/month', { headers: { 'X-Admin-Secret': SECRET } }),
      ]);
      const keysData  = await keysRes.json();
      const usageData = usageRes.ok ? await usageRes.json() : { usage: {} };
      const keys      = keysData.keys || [];

      if (!keys.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No keys yet — generate one above</td></tr>';
        return;
      }

      tbody.innerHTML = keys.map(k => {
        const limit     = k.monthly_limit;
        const used      = usageData.usage[k.id] || 0;
        const pct       = Math.min((used / limit) * 100, 100).toFixed(1);
        const fillCls   = pct >= 90 ? 'danger' : pct >= 70 ? 'warn' : '';
        const remaining = Math.max(limit - used, 0).toLocaleString();

        return `
        <tr id="row-${k.id}">
          <td><strong>${esc(k.owner || '—')}</strong></td>
          <td><span class="key-value">${esc((k.api_key || '').slice(0, 24))}...</span></td>
          <td>
            <input class="limit-input" type="number" value="${limit}" min="1"
                   id="limit-${k.id}" onkeydown="if(event.key==='Enter') updateLimit('${k.id}')">
          </td>
          <td>
            <div class="usage-wrap">
              <div class="usage-track">
                <div class="usage-fill ${fillCls}" style="width:${pct}%"></div>
              </div>
              <span class="usage-label">${remaining} left</span>
            </div>
          </td>
          <td><span class="pill ${k.is_active ? 'pill-active' : 'pill-inactive'}">${k.is_active ? 'Active' : 'Inactive'}</span></td>
          <td>${formatDate(k.created_at)}</td>
          <td style="color:var(--muted);font-size:11px">${esc(k.notes || '—')}</td>
          <td>
            <div class="td-actions">
              <button class="btn btn-ghost" style="font-size:10px;padding:5px 10px"
                      onclick="updateLimit('${k.id}')">Save</button>
              ${k.is_active
                ? `<button class="btn btn-danger" style="font-size:10px;padding:5px 10px"
                           onclick="toggleKey('${k.id}', false)">Deactivate</button>`
                : `<button class="btn btn-success" style="font-size:10px;padding:5px 10px"
                           onclick="toggleKey('${k.id}', true)">Activate</button>`
              }
            </div>
          </td>
        </tr>`;
      }).join('');
    } catch (e) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty-state" style="color:var(--red)">Failed to load keys</td></tr>';
    }
  }

  async function createKey() {
    const owner   = document.getElementById('new-owner').value.trim();
    const user_id = document.getElementById('new-user').value;
    const limit   = parseInt(document.getElementById('new-limit').value) || 100;
    const notes   = document.getElementById('new-notes').value.trim();
    if (!owner)   { toast('Owner name is required', 'err'); return; }
    if (!user_id) { toast('Please select a user', 'err'); return; }
    try {
      const r = await fetch('/admin/keys', {
        method: 'POST',
        headers: { 'X-Admin-Secret': SECRET, 'Content-Type': 'application/json' },
        body: JSON.stringify({ owner, user_id, monthly_limit: limit, notes }),
      });
      const data = await r.json();
      if (!r.ok) { toast(data.detail || 'Failed to create key', 'err'); return; }
      document.getElementById('key-reveal-value').textContent = data.key;
      document.getElementById('key-reveal').style.display = 'block';
      document.getElementById('new-owner').value = '';
      document.getElementById('new-notes').value = '';
      document.getElementById('new-limit').value = '100';
      toast('Key generated for ' + owner, 'ok');
      loadKeys();
    } catch (e) { toast('Network error', 'err'); }
  }

  function copyKey() {
    const val = document.getElementById('key-reveal-value').textContent;
    navigator.clipboard.writeText(val).then(() => toast('Key copied to clipboard', 'ok'));
  }

  async function updateLimit(keyId) {
    const newLimit = parseInt(document.getElementById('limit-' + keyId).value);
    if (!newLimit || newLimit < 1) { toast('Invalid limit', 'err'); return; }
    try {
      const r = await fetch('/admin/keys/' + keyId + '/limit', {
        method: 'PATCH',
        headers: { 'X-Admin-Secret': SECRET, 'Content-Type': 'application/json' },
        body: JSON.stringify({ monthly_limit: newLimit }),
      });
      if (!r.ok) { toast('Failed to update', 'err'); return; }
      toast('Limit updated to ' + newLimit + '/month', 'ok');
      loadKeys();
    } catch (e) { toast('Network error', 'err'); }
  }

  async function toggleKey(keyId, activate) {
    const endpoint = activate ? 'activate' : 'deactivate';
    try {
      const r = await fetch('/admin/keys/' + keyId + '/' + endpoint, {
        method: 'PATCH', headers: { 'X-Admin-Secret': SECRET },
      });
      if (!r.ok) { toast('Failed', 'err'); return; }
      toast(activate ? 'Key activated' : 'Key deactivated', 'ok');
      loadKeys();
    } catch (e) { toast('Network error', 'err'); }
  }

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function formatDate(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
  }
</script>
</body>
</html>"""