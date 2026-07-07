(function () {
  'use strict';

  // ==================== 常量 ====================
  const HOME_CENTER = [24.6080, 118.0450];
  const HOME_ZOOM = 17;
  const MAX_TRAIL_POINTS = 12000;
  const RECONNECT_MAX_DELAY = 30000;
  const CACHE_KEY = 'gps_map_tracking_trails';

  // ==================== 瓦片 ====================
  const TILE_SAT = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 19, attribution: '&copy; Esri' });
  const TILE_VEC = L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
    { subdomains: ['a','b','c','d'], maxZoom: 19, attribution: '&copy; OSM, CartoDB' });

  // ==================== DOM ====================
  const $ = id => document.getElementById(id);
  const el = {
    host: $('host-input'), port: $('port-input'),
    rosHost: $('ros-host-input'), rosPort: $('ros-port-input'), rosNs: $('ros-ns-input'),
    connectWs: $('connect-ws-btn'), disconnectWs: $('disconnect-ws-btn'),
    mapStatus: $('lio-status'), poseStatus: $('fused-status'),
    curPos: $('cur-pos'), motionStatus: $('motion-status'),
    connStatus: $('conn-status'), connDot: $('conn-dot'),
    gpsSource: $('gps-source-status'), rtkSource: $('rtk-source-status'),
    fitBtn: $('fit-btn'), clearBtn: $('clear-btn'),
    modeNote: $('mode-note'),
    leafletDiv: $('map'),
    gridCanvas: $('grid-canvas'),
    gridZoom: $('grid-zoom'),
  };

  // ==================== 状态 ====================
  const S = {
    displayMode: 'gps',           // 'gps' | 'grid'

    // --- GPS 轨迹 (WS) ---
    ws: null, wsConnected: false,
    lioCoords: [], fusedCoords: [],
    lioLine: null, fusedLine: null, robotMarker: null,
    autoFollow: true, firstPointReceived: false,
    pushMode: 'incremental', gpsSource: '/fix', rtkSource: 'auto',
    trailFilter: 'both', reconnectAttempts: 0, reconnectTimer: null,
    activeTileLayer: TILE_SAT,

    // --- 占用网格 (ROSBridge + Canvas) ---
    ros: null, rosConnected: false, rosConnecting: false, rosRetryCount: 0, _rosRetryTimer: null,
    _mapCache: null, _mapCacheValid: false,
    currentMap: null,
    robotPose: { x: 0, y: 0, theta: 0 },
    planPath: [],
    gScale: 1.0,
    gOffsetX: 0,
    gOffsetY: 0,
    gDragging: false,
    gLastMouse: { x: 0, y: 0 },
    gDrawPending: false,
    _mapWidth: 0,
    _mapHeight: 0,
    _mapOriginX: 0,
    _mapOriginY: 0,
    rosTopics: [],

    // --- 通用 ---
    legend: null,
    map: null,
  };

  // ==================== 初始化 ====================
  function init() {
    bindElements();
    initLeafletMap();
    initGridCanvas();
    loadSaved();
    applyDisplayMode('gps');
  }

  function bindElements() {
    document.querySelectorAll('[data-display]').forEach(b =>
      b.addEventListener('click', () => applyDisplayMode(b.dataset.display)));
    document.querySelectorAll('[data-gps]').forEach(b =>
      b.addEventListener('click', () => setGpsSource(b.dataset.gps)));
    document.querySelectorAll('[data-rtk]').forEach(b =>
      b.addEventListener('click', () => setRtkSource(b.dataset.rtk)));
    document.querySelectorAll('[data-mode]').forEach(b =>
      b.addEventListener('click', () => setMode(b.dataset.mode)));
    document.querySelectorAll('[data-layer]').forEach(b =>
      b.addEventListener('click', () => switchLayer(b.dataset.layer)));
    document.querySelectorAll('[data-trail]').forEach(b =>
      b.addEventListener('click', () => setTrailFilter(b.dataset.trail)));

    el.connectWs.addEventListener('click', wsConnect);
    el.disconnectWs.addEventListener('click', wsDisconnect);
    el.fitBtn.addEventListener('click', fitMap);
    el.clearBtn.addEventListener('click', clearTrails);
    const rosUpdateBtn = $('ros-update-btn');
    if (rosUpdateBtn) {
      rosUpdateBtn.addEventListener('click', () => {
        rosDisconnect();
        rosConnect();
      });
    }

    document.addEventListener('keydown', e => {
      if (e.key === 'f' && !e.ctrlKey && !e.metaKey && !e.target.closest('input,select')) toggleFollow();
    });
    window.addEventListener('resize', resizeGridCanvas);
  }

  function loadSaved() {
    const s = localStorage;
    const host = window.location.hostname;
    if (host && host !== 'localhost' && host !== '127.0.0.1') {
      el.host.value = host;
      el.rosHost.value = host;
    }
    const load = (key, elm) => { const v = s.getItem(key); if (v) elm.value = v; };
    load('mt_host', el.host); load('mt_port', el.port);
    load('mt_ros_host', el.rosHost); load('mt_ros_port', el.rosPort); load('mt_ros_ns', el.rosNs);
    if (s.getItem('mt_gps')) setGpsSource(s.getItem('mt_gps'));
    if (s.getItem('mt_rtk')) setRtkSource(s.getItem('mt_rtk'));
    if (s.getItem('mt_mode')) setMode(s.getItem('mt_mode'));
    if (s.getItem('mt_layer')) switchLayer(s.getItem('mt_layer'));
    if (s.getItem('mt_trail')) setTrailFilter(s.getItem('mt_trail'));

    // restore cached WS trails
    try {
      const c = JSON.parse(s.getItem(CACHE_KEY) || 'null');
      if (c) {
        if (c.lio && c.lio.length) { S.lioCoords = c.lio; S.lioLine.setLatLngs(S.lioCoords).addTo(S.map); }
        if (c.fused && c.fused.length) { S.fusedCoords = c.fused; S.fusedLine.setLatLngs(S.fusedCoords).addTo(S.map); }
        updateTrailVis();
      }
    } catch (e) {}
  }

  // ==================== 显示模式切换 ====================
  function applyDisplayMode(mode) {
    if (mode === S.displayMode) return;
    S.displayMode = mode;
    document.querySelectorAll('[data-display]').forEach(b =>
      b.className = 'secondary' + (b.dataset.display === mode ? ' active' : ''));
    document.getElementById('section-gps').className = 'section-gps' + (mode === 'gps' ? ' visible' : '');
    document.getElementById('section-grid').className = 'section-grid' + (mode === 'grid' ? ' visible' : '');

    if (mode === 'gps') {
      rosDisconnect();
      el.leafletDiv.style.display = '';
      el.gridCanvas.style.display = 'none';
      el.gridZoom.style.display = 'none';
      if (S.map) {
        showTileLayer(S.activeTileLayer);
        updateTrailVis();
        if (S.lioLine && S.lioLine._map) S.map.removeLayer(S.lioLine);
        if (S.fusedLine && S.fusedLine._map) S.map.removeLayer(S.fusedLine);
        if (S.robotMarker && S.map.hasLayer(S.robotMarker)) S.map.removeLayer(S.robotMarker);
        updateTrailVis();
        S.map.invalidateSize();
      }
      el.modeNote.textContent = 'GPS 轨迹模式：WebSocket 8765 · Esri/CartoDB 瓦片底图 · 按 F 切换跟随';
    } else {
      wsDisconnect();
      el.leafletDiv.style.display = 'none';
      el.gridCanvas.style.display = 'block';
      el.gridZoom.style.display = 'flex';
      hideAllTiles();
      if (S.lioLine && S.lioLine._map && S.map) S.map.removeLayer(S.lioLine);
      if (S.fusedLine && S.fusedLine._map && S.map) S.map.removeLayer(S.fusedLine);
      if (S.robotMarker && S.map && S.map.hasLayer(S.robotMarker)) S.map.removeLayer(S.robotMarker);
      resizeGridCanvas();
      gridDraw();
      el.modeNote.textContent = '占用网格模式：ROSBridge 9090 · Canvas 渲染 /map · 滚轮缩放+拖拽';
      // 自动连接 ROSBridge（断连后自动重试）
      rosConnect();
    }
  }

  // ==================== Leaflet (GPS 模式) ====================
  function initLeafletMap() {
    const map = L.map('map', { zoomControl: true, preferCanvas: true, attributionControl: false })
      .setView(HOME_CENTER, HOME_ZOOM);
    TILE_SAT.addTo(map);
    S.activeTileLayer = TILE_SAT;

    S.lioLine = L.polyline([], { color: '#3b82f6', weight: 3, opacity: 0.85, smoothFactor: 1 });
    S.fusedLine = L.polyline([], { color: '#22c55e', weight: 3, opacity: 0.85, smoothFactor: 1 });
    S.robotMarker = L.marker([0, 0], { icon: makeArrow('#22c55e', 0), interactive: false });

    // 图例
    S.legend = L.control({ position: 'bottomleft' });
    S.legend.onAdd = function () {
      const d = L.DomUtil.create('div');
      d.id = 'legend-box';
      d.style.cssText = 'background:rgba(17,24,39,0.92);border:1px solid #1e293b;border-radius:6px;padding:6px 10px;font-size:12px;color:#c8d5e6;line-height:1.7;';
      renderLegendLeaflet(d);
      return d;
    };
    S.legend.addTo(map);
    map.on('dragstart', () => { S.autoFollow = false; });
    S.map = map;
  }

  function showTileLayer(layer) {
    if (!S.map) return;
    try {
      hideAllTiles();
      if (S.map) layer.addTo(S.map);
    } catch (e) { console.warn('[map] showTileLayer err:', e); }
  }
  function hideAllTiles() {
    if (!S.map) return;
    try {
      if (TILE_SAT._map) S.map.removeLayer(TILE_SAT);
      if (TILE_VEC._map) S.map.removeLayer(TILE_VEC);
    } catch (e) { /* ignore */ }
  }

  function switchLayer(type) {
    S.activeTileLayer = type === 'sat' ? TILE_SAT : TILE_VEC;
    document.querySelectorAll('[data-layer]').forEach(b =>
      b.className = 'secondary' + (b.dataset.layer === type ? ' active' : ''));
    localStorage.setItem('mt_layer', type);
    if (S.displayMode === 'gps') showTileLayer(S.activeTileLayer);
  }

  function makeArrow(color, deg) {
    return L.divIcon({
      className: '',
      html: `<div style="transform:rotate(${deg||0}deg);width:24px;height:24px;"><svg width="24" height="24" viewBox="-12 -12 24 24"><polygon points="0,-10 8,5 -8,5" fill="${color}" stroke="#fff" stroke-width="1.2" stroke-linejoin="round"/></svg></div>`,
      iconSize: [24, 24], iconAnchor: [12, 12],
    });
  }

  function renderLegendLeaflet(el) {
    el.innerHTML = `<div><span style="display:inline-block;width:20px;height:3px;border-radius:1px;background:#3b82f6;vertical-align:middle;margin-right:6px"></span>LIO</div>
<div><span style="display:inline-block;width:20px;height:3px;border-radius:1px;background:#22c55e;vertical-align:middle;margin-right:6px"></span>融合</div>
<div style="font-size:10px;color:#64748b;margin-top:2px">${S.activeTileLayer===TILE_SAT?'Esri卫星':'CartoDB矢量'}</div>`;
  }

  // ==================== Canvas (占用网格模式) ====================
  function initGridCanvas() {
    const cvs = el.gridCanvas;
    cvs.addEventListener('wheel', onGridWheel, { passive: false });
    cvs.addEventListener('mousedown', onGridMouseDown);
    cvs.addEventListener('mousemove', onGridMouseMove);
    cvs.addEventListener('mouseup', onGridMouseUp);
    window.gridZoomIn = gridZoomIn;
    window.gridZoomOut = gridZoomOut;
    window.gridResetView = gridResetView;
  }

  function resizeGridCanvas() {
    const cvs = el.gridCanvas;
    const parent = cvs.parentElement;
    if (!parent) return;
    cvs.width = parent.clientWidth;
    cvs.height = parent.clientHeight;
    gridDraw();
  }

  function gridZoomIn() { S.gScale *= 1.3; gridDraw(); }
  function gridZoomOut() { S.gScale /= 1.3; gridDraw(); }
  function gridResetView() { S.gScale = 1; S.gOffsetX = 0; S.gOffsetY = 0; gridDraw(); }

  function onGridWheel(e) {
    e.preventDefault();
    const rect = el.gridCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    // zoom toward mouse position
    S.gOffsetX = mx - (mx - S.gOffsetX) * factor;
    S.gOffsetY = my - (my - S.gOffsetY) * factor;
    S.gScale *= factor;
    gridDraw();
  }

  function onGridMouseDown(e) {
    if (e.button !== 0) return;
    S.gDragging = true;
    S.gLastMouse.x = e.clientX;
    S.gLastMouse.y = e.clientY;
  }

  function onGridMouseMove(e) {
    if (!S.gDragging) return;
    const dx = e.clientX - S.gLastMouse.x;
    const dy = e.clientY - S.gLastMouse.y;
    S.gLastMouse.x = e.clientX;
    S.gLastMouse.y = e.clientY;
    S.gOffsetX += dx;
    S.gOffsetY += dy;
    gridDraw();
  }

  function onGridMouseUp() { S.gDragging = false; }

  // ==================== Canvas 渲染 ====================
  function gridDraw() {
    if (S.displayMode !== 'grid') return;
    const cvs = el.gridCanvas;
    const ctx = cvs.getContext('2d');
    const W = cvs.width, H = cvs.height;
    if (!W || !H) return;

    ctx.clearRect(0, 0, W, H);

    if (!S.currentMap) {
      ctx.fillStyle = '#151b24';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#64748b';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('等待 /map 数据...', W / 2, H / 2);
      return;
    }

    const map = S.currentMap;
    const info = map.info;
    const res = info.resolution;
    const ox = info.origin.position.x;
    const oy = info.origin.position.y;
    const gw = info.width;
    const gh = info.height;
    const scale = S.gScale;
    const offX = S.gOffsetX || (W / 2);
    const offY = S.gOffsetY || (H / 2);

    // 地图缓存：只在 /map 数据变化时重建
    if (!S._mapCacheValid) {
      rebuildMapCache(map, gw, gh);
    }

    // 从缓存画地图（带缩放/平移变换）
    if (S._mapCache) {
      const cw = S._mapCache.width;
      const ch = S._mapCache.height;
      // map 坐标 → 屏幕坐标 的映射
      const sx = offX + (ox - ox) * scale; // = offX (地图原点x)
      const sy = offY - (oy - oy) * scale; // = offY (地图原点y)
      // 缓存中每个像素 = 1 grid cell，格子大小 = res 米
      // 屏幕像素/格 = scale * res
      const pixelPerCell = scale * res;
      const dstW = cw * pixelPerCell;
      const dstH = ch * pixelPerCell;
      const dstX = offX + (ox - ox) * scale;
      const dstY = offY - (oy - oy) * scale;

      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(S._mapCache, dstX, dstY - dstH, dstW, dstH);
    } else {
      // 降级：逐像素渲染（首次或缓存失败）
      legacyGridDraw(ctx, W, H, scale, offX, offY, ox, oy, res, gw, gh, map.data);
    }

    // 画机器人（只叠加，不重绘地图）
    gridDrawRobot(ctx, W, H, scale, offX, offY, ox, oy);
    gridDrawPlan(ctx, W, H, scale, offX, offY, ox, oy);
  }

  function rebuildMapCache(map, gw, gh) {
    const info = map.info;
    const res = info.resolution;
    const ox = info.origin.position.x;
    const oy = info.origin.position.y;
    const data = map.data;

    // 创建离屏 Canvas，一个像素对应一个 grid cell
    const cache = document.createElement('canvas');
    cache.width = gw;
    cache.height = gh;
    const cctx = cache.getContext('2d');
    const imgData = cctx.createImageData(gw, gh);
    const px = imgData.data;

    for (let gy = 0; gy < gh; gy++) {
      for (let gx = 0; gx < gw; gx++) {
        const di = gy * gw + gx;
        const val = di < data.length ? data[di] : -1;
        const pi = ((gh - 1 - gy) * gw + gx) * 4; // Y 翻转

        if (val === -1) {
          px[pi] = 100; px[pi+1] = 110; px[pi+2] = 130;
        } else if (val === 0) {
          px[pi] = 240; px[pi+1] = 245; px[pi+2] = 255;
        } else {
          const t = val / 100;
          px[pi] = 10 + t * 30; px[pi+1] = 15 + t * 20; px[pi+2] = 30 + t * 40;
        }
        px[pi+3] = 255;
      }
    }
    cctx.putImageData(imgData, 0, 0);

    S._mapCache = cache;
    S._mapCacheValid = true;
  }

  // 降级：逐像素渲染（仅在缓存不可用时）
  function legacyGridDraw(ctx, W, H, scale, offX, offY, ox, oy, res, gw, gh, data) {
    const imageData = ctx.createImageData(W, H);
    const pixels = imageData.data;
    for (let sy = 0; sy < H; sy++) {
      for (let sx = 0; sx < W; sx++) {
        const mx = (sx - offX) / scale + ox;
        const my = (sy - offY) / (-scale) + oy;
        const gx = Math.floor((mx - ox) / res);
        const gy = Math.floor((my - oy) / res);
        const idx = (sy * W + sx) * 4;
        if (gx < 0 || gx >= gw || gy < 0 || gy >= gh) {
          pixels[idx] = 21; pixels[idx+1] = 24; pixels[idx+2] = 36; pixels[idx+3] = 255;
          continue;
        }
        const di = gy * gw + gx;
        const val = di < data.length ? data[di] : -1;
        if (val === -1) { pixels[idx]=100; pixels[idx+1]=110; pixels[idx+2]=130; }
        else if (val === 0) { pixels[idx]=240; pixels[idx+1]=245; pixels[idx+2]=255; }
        else { const t=val/100; pixels[idx]=10+t*30; pixels[idx+1]=15+t*20; pixels[idx+2]=30+t*40; }
        pixels[idx+3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }

  function gridDrawRobot(ctx, W, H, scale, offX, offY, ox, oy) {
    const p = S.robotPose;
    const localized = !(p.x === 0 && p.y === 0);
    const sx = offX + (p.x - ox) * scale;
    const sy = offY - (p.y - oy) * scale;
    const size = Math.max(10, scale * 0.6);

    ctx.save();
    ctx.translate(sx, sy);

    if (!localized) {
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.6, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(168,85,247,0.2)';
      ctx.fill();
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#a855f7';
      ctx.font = 'bold ' + Math.round(size * 0.8) + 'px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('?', 0, 1);
      ctx.restore();
      ctx.save();
      ctx.fillStyle = '#fbbf24';
      ctx.font = '13px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('未定位 — 点击地图设置初始点', sx, sy + size + 16);
      ctx.restore();
      return;
    }

    ctx.rotate(-p.theta);
    ctx.beginPath();
    ctx.moveTo(0, -size);
    ctx.lineTo(size * 0.6, size * 0.5);
    ctx.lineTo(-size * 0.6, size * 0.5);
    ctx.closePath();
    ctx.fillStyle = '#a855f7';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }

  function gridDrawPlan(ctx, W, H, scale, offX, offY, ox, oy) {
    if (!S.planPath || S.planPath.length < 2) return;
    ctx.save();
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = Math.max(2, scale * 0.3);
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    let moved = false, lastSx, lastSy;
    const MAX_GAP_M = 5; // 超过 5 米的跳变断开分段
    for (const p of S.planPath) {
      const sx = offX + (p.x - ox) * scale;
      const sy = offY - (p.y - oy) * scale;
      if (sx < -10000 || sx > W + 10000 || sy < -10000 || sy > H + 10000) {
        moved = false;
        continue;
      }
      // 距离跳变检查（屏幕像素距离折算到地图米）
      if (moved) {
        const gapM = Math.hypot((sx - lastSx) / scale, (sy - lastSy) / scale);
        if (gapM > MAX_GAP_M) moved = false;
      }
      if (!moved) { ctx.moveTo(sx, sy); moved = true; }
      else { ctx.lineTo(sx, sy); }
      lastSx = sx; lastSy = sy;
    }
    ctx.stroke();
    ctx.restore();
  }

  // ==================== WS (GPS 轨迹) ====================
  function wsConnect() {
    clearWsReconnect();
    wsCloseInternal();
    const host = el.host.value.trim();
    const port = parseInt(el.port.value) || 8765;
    if (!host) { alert('请输入 IP'); return; }
    localStorage.setItem('mt_host', host);
    localStorage.setItem('mt_port', port);
    const url = 'ws://' + host + ':' + port;
    wsSetUI(true);
    setConn('连接中...');

    const ws = new WebSocket(url);
    S.ws = ws;
    ws.onopen = () => {
      S.wsConnected = true; S.reconnectAttempts = 0;
      S.firstPointReceived = false; S.autoFollow = true;
      wsSetUI(false);
      setConn('已连接 (WS)');
      el.connDot.className = 'status-dot online';
      wsSend({ type: 'mode', mode: S.pushMode });
      wsSend({ type: 'gps_source', source: S.gpsSource });
      wsSend({ type: 'rtk_source', source: S.rtkSource });
      wsSend({ type: 'get_full' });
    };
    ws.onmessage = e => { try { wsHandle(JSON.parse(e.data)); } catch (e) {} };
    ws.onerror = () => { if (ws === S.ws) { wsSetUI(false); setConn('❌ 连接失败'); el.connDot.className = 'status-dot offline'; } };
    ws.onclose = () => {
      S.wsConnected = false;
      if (ws !== S.ws) return;
      wsSetUI(false); setConn('已断开'); el.connDot.className = 'status-dot offline';
      wsReconnect(host, port);
    };
  }

  function wsDisconnect() { clearWsReconnect(); wsCloseInternal(); S.reconnectAttempts = 0; setConn('已断开'); el.connDot.className = 'status-dot offline'; el.disconnectWs.disabled = true; el.connectWs.disabled = false; }
  function wsCloseInternal() { if (S.ws) { S.ws.onclose = null; S.ws.close(); S.ws = null; } }
  function wsSend(obj) { if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(obj)); }
  function wsSetUI(connecting) { el.connectWs.disabled = connecting; el.disconnectWs.disabled = !connecting; el.connectWs.textContent = connecting ? '连接中...' : '连接'; }
  function wsReconnect(host, port) { S.reconnectAttempts++; const d = Math.min(3000 * Math.pow(2, Math.min(S.reconnectAttempts - 1, 4)), RECONNECT_MAX_DELAY); setConn('⏳ 重连 ' + (d / 1000).toFixed(0) + 's...'); S.reconnectTimer = setTimeout(() => { if (!S.ws || S.ws.readyState !== WebSocket.OPEN) wsConnect(); }, d); }
  function clearWsReconnect() { if (S.reconnectTimer) { clearTimeout(S.reconnectTimer); S.reconnectTimer = null; } }

  function wsHandle(msg) {
    switch (msg.type) {
      case 'full':
        if (msg.lio) { S.lioCoords = msg.lio.map(p => [p[1], p[0]]); S.lioLine.setLatLngs(S.lioCoords).addTo(S.map); }
        if (msg.fused) { S.fusedCoords = msg.fused.map(p => [p[1], p[0]]); S.fusedLine.setLatLngs(S.fusedCoords).addTo(S.map); wsUpdateRobot(S.fusedCoords); }
        else if (S.lioCoords.length) wsCenter(S.lioCoords[S.lioCoords.length - 1]);
        wsStats(msg.stats); updateTrailVis(); break;
      case 'incremental':
        let ch = false;
        if (msg.lio && msg.lio.length) { msg.lio.forEach(p => S.lioCoords.push([p[1], p[0]])); if (S.lioCoords.length > MAX_TRAIL_POINTS) S.lioCoords = S.lioCoords.slice(-MAX_TRAIL_POINTS); S.lioLine.setLatLngs(S.lioCoords); if (!S.lioLine._map) S.lioLine.addTo(S.map); ch = true; }
        if (msg.fused && msg.fused.length) { msg.fused.forEach(p => S.fusedCoords.push([p[1], p[0]])); if (S.fusedCoords.length > MAX_TRAIL_POINTS) S.fusedCoords = S.fusedCoords.slice(-MAX_TRAIL_POINTS); S.fusedLine.setLatLngs(S.fusedCoords); if (!S.fusedLine._map) S.fusedLine.addTo(S.map); wsUpdateRobot(S.fusedCoords); ch = true; } else if (msg.lio && msg.lio.length) { const l = msg.lio[msg.lio.length - 1]; wsCenter([l[1], l[0]]); }
        if (ch) { wsStats(msg.stats); updateTrailVis(); } break;
      case 'pong': break;
      case 'info': if (msg.gps_source) syncGps(msg.gps_source); if (msg.rtk_source) syncRtk(msg.rtk_source); break;
    }
  }

  function wsUpdateRobot(c) {
    if (!c.length) return;
    const last = c[c.length - 1];
    const h = c.length >= 2 ? calcHeading(c[c.length - 2], last) : 0;
    S.robotMarker.setLatLng(last);
    S.robotMarker.setIcon(makeArrow('#22c55e', h));
    if (S.displayMode === 'gps' && !S.map.hasLayer(S.robotMarker)) S.robotMarker.addTo(S.map);
    wsCenter(last);
    el.motionStatus.textContent = dirName(h) + ' ' + h.toFixed(1) + '°';
  }

  function wsCenter(l) { if (!l || (l[0] === 0 && l[1] === 0)) return; if (!S.firstPointReceived) { S.firstPointReceived = true; S.autoFollow = true; S.map.setView(l, Math.max(S.map.getZoom(), HOME_ZOOM), { animate: true }); return; } if (S.autoFollow) S.map.panTo(l, { animate: true, maxZoom: 19 }); }

  function wsStats(st) {
    if (!st) return;
    el.mapStatus.textContent = (st.lio_count || S.lioCoords.length) + ' lio / ' + (st.fused_count || S.fusedCoords.length) + ' fused';
    const dL = dist(S.lioCoords), dF = dist(S.fusedCoords);
    if (st.fused_last) el.curPos.textContent = st.fused_last[1].toFixed(7) + ', ' + st.fused_last[0].toFixed(7);
    else if (st.lio_last) el.curPos.textContent = st.lio_last[1].toFixed(7) + ', ' + st.lio_last[0].toFixed(7);
    if (st.gps_source) syncGps(st.gps_source);
    if (st.rtk_source) syncRtk(st.rtk_source);
    try { const max = 5000; localStorage.setItem(CACHE_KEY, JSON.stringify({ lio: S.lioCoords.slice(-max), fused: S.fusedCoords.slice(-max) })); } catch (e) {}
  }

  // ==================== WS 控制 ====================
  function setGpsSource(src) { S.gpsSource = src; document.querySelectorAll('[data-gps]').forEach(b => b.className = 'secondary' + (b.dataset.gps === src ? ' active' : '')); localStorage.setItem('mt_gps', src); wsSend({ type: 'gps_source', source: src }); }
  function setRtkSource(src) { S.rtkSource = src; document.querySelectorAll('[data-rtk]').forEach(b => b.className = 'secondary' + (b.dataset.rtk === src ? ' active' : '')); localStorage.setItem('mt_rtk', src); wsSend({ type: 'rtk_source', source: src }); }
  function setMode(mode) { S.pushMode = mode; document.querySelectorAll('[data-mode]').forEach(b => b.className = 'secondary' + (b.dataset.mode === mode ? ' active' : '')); localStorage.setItem('mt_mode', mode); wsSend({ type: 'mode', mode: mode }); }
  function syncGps(src) { if (src !== S.gpsSource) { S.gpsSource = src; document.querySelectorAll('[data-gps]').forEach(b => b.className = 'secondary' + (b.dataset.gps === src ? ' active' : '')); } el.gpsSource.textContent = src; }
  function syncRtk(src) { if (src !== S.rtkSource) { S.rtkSource = src; document.querySelectorAll('[data-rtk]').forEach(b => b.className = 'secondary' + (b.dataset.rtk === src ? ' active' : '')); } el.rtkSource.textContent = src; }

  function setTrailFilter(f) { S.trailFilter = f; document.querySelectorAll('[data-trail]').forEach(b => b.className = 'secondary' + (b.dataset.trail === f ? ' active' : '')); localStorage.setItem('mt_trail', f); updateTrailVis(); }
  function updateTrailVis() {
    if (S.displayMode !== 'gps') return;
    const f = S.trailFilter, a = f === 'both' || f === 'lio', b = f === 'both' || f === 'fused';
    if (S.lioLine) { S.lioLine.setStyle({ opacity: a ? 0.85 : 0 }); if (a && !S.lioLine._map) S.lioLine.addTo(S.map); }
    if (S.fusedLine) { S.fusedLine.setStyle({ opacity: b ? 0.85 : 0 }); if (b && !S.fusedLine._map) S.fusedLine.addTo(S.map); }
    const show = a && S.fusedCoords.length > 0;
    if (show && !S.map.hasLayer(S.robotMarker)) S.robotMarker.addTo(S.map);
    if (!show && S.map.hasLayer(S.robotMarker)) S.map.removeLayer(S.robotMarker);
  }

  // ==================== ROSBridge + Canvas (占用网格) ====================
  function rosConnect() {
    if (S.rosConnected || S.rosConnecting) return;
    rosDisconnectInternal();
    const host = el.rosHost.value.trim() || window.location.hostname || '127.0.0.1';
    const port = parseInt(el.rosPort.value) || 9090;
    localStorage.setItem('mt_ros_host', host);
    localStorage.setItem('mt_ros_port', port);
    localStorage.setItem('mt_ros_ns', el.rosNs.value.trim());

    S.rosConnecting = true;
    const url = 'ws://' + host + ':' + port;
    setConn('连接中...');

    const ros = new ROSLIB.Ros({ url });
    S.ros = ros;

    ros.on('connection', () => {
      S.rosConnected = true;
      S.rosConnecting = false;
      S.rosRetryCount = 0;
      setConn('已连接 (ROS)');
      el.connDot.className = 'status-dot online';
      rosSubscribe();
    });

    ros.on('error', err => {
      S.rosConnecting = false;
      if (S.ros === ros) { setConn('❌ ROS 连接错误'); el.connDot.className = 'status-dot offline'; }
    });

    ros.on('close', () => {
      S.rosConnected = false;
      S.rosConnecting = false;
      if (S.ros !== ros) return;
      setConn('ROS 已断开');
      el.connDot.className = 'status-dot offline';
      // 自动重连（指数退避）
      S.rosRetryCount = (S.rosRetryCount || 0) + 1;
      const delay = Math.min(3000 * Math.pow(2, Math.min(S.rosRetryCount - 1, 4)), 30000);
      S._rosRetryTimer = setTimeout(() => { if (S.displayMode === 'grid') rosConnect(); }, delay);
    });
  }

  function rosDisconnect() { S.rosRetryCount = 0; clearTimeout(S._rosRetryTimer); rosDisconnectInternal(); setConn('已断开'); el.connDot.className = 'status-dot offline'; }

  function rosDisconnectInternal() {
    clearTimeout(S._rosRetryTimer);
    clearTimeout(S._rosRetryTimer);
    S.rosTopics.forEach(t => t.unsubscribe());
    S.rosTopics = [];
    if (S.ros) { S.ros.close(); S.ros = null; }
    S.rosConnected = false;
    S.rosConnecting = false;
    S.currentMap = null;
    S.robotPose = { x: 0, y: 0, theta: 0 };
    S.planPath = [];
    gridDraw();
  }

  function setConn(text) { el.connStatus.textContent = text; }

  function rosSubscribe() {
    const ns = el.rosNs.value.trim().replace(/^\//, '');  // 去掉前导 /
    const mk = name => ns ? '/' + ns + name : name;

    const mapT = new ROSLIB.Topic({
      ros: S.ros, name: mk('/map'), messageType: 'nav_msgs/OccupancyGrid',
      qos: { durability: 'transient_local', reliability: 'reliable' }
    });
    mapT.subscribe(msg => {
      S.currentMap = msg;
      S._mapCacheValid = false;  // 新地图需重建缓存
      // 首次 /map 自动复位视角
      if (!S._mapWidth) {
        S._mapWidth = msg.info.width;
        S._mapHeight = msg.info.height;
        S._mapOriginX = msg.info.origin.position.x;
        S._mapOriginY = msg.info.origin.position.y;
        const cvs = el.gridCanvas;
        const mapW = msg.info.width * msg.info.resolution;
        const mapH = msg.info.height * msg.info.resolution;
        const fit = Math.min(cvs.width / mapW, cvs.height / mapH) * 0.9;
        S.gScale = fit;
        // 地图居中：地图中心点 → Canvas 中心
        const ox = msg.info.origin.position.x;
        const oy = msg.info.origin.position.y;
        const cx = ox + mapW / 2;
        const cy = oy + mapH / 2;
        S.gOffsetX = cvs.width / 2 - (cx - ox) * fit;
        S.gOffsetY = cvs.height / 2 + (cy - oy) * fit;
      }
      el.mapStatus.textContent = `${msg.info.width}x${msg.info.height} @ ${msg.info.resolution}m`;
      gridDraw();
    });
    S.rosTopics.push(mapT);

    const amclT = new ROSLIB.Topic({ ros: S.ros, name: '/amcl_pose', messageType: 'geometry_msgs/PoseWithCovarianceStamped' });
    amclT.subscribe(msg => {
      S.robotPose.x = msg.pose.pose.position.x;
      S.robotPose.y = msg.pose.pose.position.y;
      S.robotPose.theta = (() => { const o = msg.pose.pose.orientation; return Math.atan2(2 * (o.w * o.z), 1 - 2 * (o.z * o.z)); })();
      el.poseStatus.textContent = `(${S.robotPose.x.toFixed(2)}, ${S.robotPose.y.toFixed(2)})`;
      el.curPos.textContent = `x:${S.robotPose.x.toFixed(2)} y:${S.robotPose.y.toFixed(2)}`;
      el.motionStatus.textContent = (S.robotPose.theta * 180 / Math.PI).toFixed(1) + '°';
      gridDraw();
    });
    S.rosTopics.push(amclT);

    // 也订阅 /rkbot/amcl_pose 作为备选
    const amclT2 = new ROSLIB.Topic({ ros: S.ros, name: mk('/amcl_pose'), messageType: 'geometry_msgs/PoseWithCovarianceStamped' });
    amclT2.subscribe(msg => {
      S.robotPose.x = msg.pose.pose.position.x;
      S.robotPose.y = msg.pose.pose.position.y;
      S.robotPose.theta = (() => { const o = msg.pose.pose.orientation; return Math.atan2(2 * (o.w * o.z), 1 - 2 * (o.z * o.z)); })();
    });
    S.rosTopics.push(amclT2);

    // 路径规划
    const planT = new ROSLIB.Topic({ ros: S.ros, name: mk('/plan'), messageType: 'nav_msgs/Path' });
    planT.subscribe(msg => {
      S.planPath = msg.poses.map(p => ({ x: p.pose.position.x, y: p.pose.position.y }));
      gridDraw();
    });
    S.rosTopics.push(planT);
    const pathT = new ROSLIB.Topic({ ros: S.ros, name: mk('/path'), messageType: 'nav_msgs/Path' });
    pathT.subscribe(msg => {
      S.planPath = msg.poses.map(p => ({ x: p.pose.position.x, y: p.pose.position.y }));
      gridDraw();
    });
    S.rosTopics.push(pathT);

    setConn('已连接 (ROS)');
  }

  // ==================== 通用工具 ====================
  function calcHeading(p1, p2) { if (!p1 || !p2) return 0; return Math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / Math.PI; }
  function dirName(h) { return ['北','东北','东','东南','南','西南','西','西北'][Math.round(((h+360)%360)/45)%8]; }
  function dist(c) { if (c.length < 2) return 0; let d = 0; for (let i=1;i<c.length;i++) { const ml = (c[i][0]+c[i-1][0])/2*Math.PI/180; d += Math.hypot((c[i][1]-c[i-1][1])*111320*Math.cos(ml), (c[i][0]-c[i-1][0])*111320); } return d; }

  function toggleFollow() {
    S.autoFollow = !S.autoFollow;
    if (S.autoFollow) {
      const last = S.fusedCoords.length > 0 ? S.fusedCoords[S.fusedCoords.length-1] : (S.lioCoords.length > 0 ? S.lioCoords[S.lioCoords.length-1] : null);
      if (last) S.map.panTo(last, { maxZoom: 19 });
    }
  }

  function fitMap() {
    if (S.displayMode === 'gps') {
      const a = []; if (S.trailFilter === 'both' || S.trailFilter === 'lio') a.push(...S.lioCoords); if (S.trailFilter === 'both' || S.trailFilter === 'fused') a.push(...S.fusedCoords);
      if (a.length) S.map.fitBounds(a, { padding: [30, 30], maxZoom: 19 }); else S.map.setView(HOME_CENTER, HOME_ZOOM);
    } else { gridResetView(); }
    S.autoFollow = false;
  }

  function clearTrails() {
    S.lioCoords = []; S.fusedCoords = []; S.planPath = [];
    S.lioLine.setLatLngs([]); S.fusedLine.setLatLngs([]);
    if (S.map.hasLayer(S.robotMarker)) S.map.removeLayer(S.robotMarker);
    S.robotMarker.setLatLng([0, 0]);
    S.firstPointReceived = false;
    el.mapStatus.textContent = '--'; el.poseStatus.textContent = '--'; el.curPos.textContent = '--'; el.motionStatus.textContent = '--';
    localStorage.removeItem(CACHE_KEY);
  }

  // ==================== 启动 ====================
  window.addEventListener('load', init);
})();
