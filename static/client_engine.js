/**
 * Eraseon - Pure Browser Client-Side Engine
 * Runs 100% serverless on GitHub Pages (images & multi-page PDFs) with:
 * - Multi-scale Top-Hat / Black-Hat + PCA oriented polygon + repeating NCC template detection
 * - Dekel-style Multi-Instance Aligned Median Matte Recovery in YCbCr (zero blur on faces/hair/clouds)
 * - Multi-scale Push-Pull Harmonic Pyramid + Relaxation + Sensor-Grain inpainting for opaque logos/sparkles
 */
(function (global) {
  const PDFJS_CDN = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js';
  const PDFJS_WORKER_CDN = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js';
  const PDFLIB_CDN = 'https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.min.js';

  const loadedScripts = {};
  function loadScriptOnce(src) {
    if (loadedScripts[src]) return loadedScripts[src];
    loadedScripts[src] = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });
    return loadedScripts[src];
  }

  async function ensurePdfJs() {
    await loadScriptOnce(PDFJS_CDN);
    if (global.pdfjsLib && !global.pdfjsLib.GlobalWorkerOptions.workerSrc) {
      global.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_CDN;
    }
    return global.pdfjsLib;
  }

  async function ensurePdfLib() {
    await loadScriptOnce(PDFLIB_CDN);
    return global.PDFLib;
  }

  const clientSessions = {};

  function clamp255(v) {
    return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
  }

  function rgbToY(r, g, b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
  }

  function extractGray(data, w, h) {
    const gray = new Float32Array(w * h);
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    }
    return gray;
  }

  function fastBoxBlurFloat(src, w, h, radius) {
    const tmp = new Float32Array(w * h);
    const dst = new Float32Array(w * h);
    const win = 2 * radius + 1;

    for (let y = 0; y < h; y++) {
      let sum = 0;
      const off = y * w;
      for (let kx = -radius; kx <= radius; kx++) {
        sum += src[off + Math.max(0, Math.min(w - 1, kx))];
      }
      for (let x = 0; x < w; x++) {
        tmp[off + x] = sum / win;
        sum += src[off + Math.min(w - 1, x + radius + 1)] - src[off + Math.max(0, x - radius)];
      }
    }

    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let ky = -radius; ky <= radius; ky++) {
        sum += tmp[Math.max(0, Math.min(h - 1, ky)) * w + x];
      }
      for (let y = 0; y < h; y++) {
        dst[y * w + x] = sum / win;
        sum += tmp[Math.min(h - 1, y + radius + 1) * w + x] - tmp[Math.max(0, y - radius) * w + x];
      }
    }
    return dst;
  }

  function morphExtremaGray(src, w, h, radius, isMax) {
    const tmp = new Float32Array(w * h);
    const dst = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const off = y * w;
      for (let x = 0; x < w; x++) {
        let best = isMax ? -1e9 : 1e9;
        const x0 = Math.max(0, x - radius), x1 = Math.min(w - 1, x + radius);
        for (let kx = x0; kx <= x1; kx++) {
          const v = src[off + kx];
          if (isMax ? v > best : v < best) best = v;
        }
        tmp[off + x] = best;
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let best = isMax ? -1e9 : 1e9;
        const y0 = Math.max(0, y - radius), y1 = Math.min(h - 1, y + radius);
        for (let ky = y0; ky <= y1; ky++) {
          const v = tmp[ky * w + x];
          if (isMax ? v > best : v < best) best = v;
        }
        dst[y * w + x] = best;
      }
    }
    return dst;
  }

  function morphOpenGray(src, w, h, radius) {
    return morphExtremaGray(morphExtremaGray(src, w, h, radius, false), w, h, radius, true);
  }

  function morphCloseGray(src, w, h, radius) {
    return morphExtremaGray(morphExtremaGray(src, w, h, radius, true), w, h, radius, false);
  }

  function dilateBinary(src, w, h, rx, ry = rx) {
    if (rx <= 0 && ry <= 0) return new Uint8Array(src);
    const tmp = new Uint8Array(w * h);
    const dst = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const off = y * w;
      for (let x = 0; x < w; x++) {
        let on = 0;
        const x0 = Math.max(0, x - rx), x1 = Math.min(w - 1, x + rx);
        for (let kx = x0; kx <= x1; kx++) {
          if (src[off + kx]) { on = 1; break; }
        }
        tmp[off + x] = on;
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let on = 0;
        const y0 = Math.max(0, y - ry), y1 = Math.min(h - 1, y + ry);
        for (let ky = y0; ky <= y1; ky++) {
          if (tmp[ky * w + x]) { on = 1; break; }
        }
        dst[y * w + x] = on;
      }
    }
    return dst;
  }

  function connectedComponents(binMask, w, h) {
    const labels = new Int32Array(w * h);
    const comps = [];
    const stack = new Int32Array(w * h);
    let labelId = 0;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (!binMask[idx] || labels[idx]) continue;

        labelId++;
        let top = 0;
        stack[top++] = idx;
        labels[idx] = labelId;
        let minX = x, maxX = x, minY = y, maxY = y, area = 0;

        while (top > 0) {
          const cur = stack[--top];
          const cx = cur % w;
          const cy = (cur - cx) / w;
          area++;
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;

          for (let dy = -1; dy <= 1; dy++) {
            const ny = cy + dy;
            if (ny < 0 || ny >= h) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const nx = cx + dx;
              if (nx < 0 || nx >= w) continue;
              const nIdx = ny * w + nx;
              if (binMask[nIdx] && !labels[nIdx]) {
                labels[nIdx] = labelId;
                stack[top++] = nIdx;
              }
            }
          }
        }
        comps.push({
          id: labelId,
          x: minX,
          y: minY,
          width: maxX - minX + 1,
          height: maxY - minY + 1,
          area
        });
      }
    }
    return { labels, comps };
  }

  function findEmbeddedPhotoRegions(gray, w, h) {
    const borderVals = [];
    for (let x = 0; x < w; x += 4) {
      borderVals.push(gray[x], gray[(h - 1) * w + x]);
    }
    borderVals.sort((a, b) => a - b);
    const borderMed = borderVals[Math.floor(borderVals.length / 2)] || 0;
    if (borderMed < 235) return [];

    const nonWhite = new Uint8Array(w * h);
    for (let i = 0; i < nonWhite.length; i++) {
      if (gray[i] < 236) nonWhite[i] = 1;
    }
    const dilated = dilateBinary(nonWhite, w, h, 3);
    const { comps } = connectedComponents(dilated, w, h);
    const regions = [];
    const minArea = w * h * 0.04;
    for (const c of comps) {
      if (c.width * c.height >= minArea && c.width >= w * 0.35 && c.height >= h * 0.14) {
        let darkCount = 0;
        const total = c.width * c.height;
        for (let yy = c.y; yy < c.y + c.height; yy++) {
          for (let xx = c.x; xx < c.x + c.width; xx++) {
            if (gray[yy * w + xx] < 235) darkCount++;
          }
        }
        if (darkCount / total > 0.70) {
          regions.push({ x: c.x, y: c.y, width: c.width, height: c.height });
        }
      }
    }
    return regions;
  }

  function detectCornerSparklesInRegion(imgData, gray, imgW, rx, ry, rw, rh, sensitivity, dilatePx) {
    const boxes = [];
    const y1 = ry + Math.floor(rh * 0.60);
    const y2 = ry + rh;
    const x1 = rx + Math.floor(rw * 0.68);
    const x2 = rx + rw;
    const cw = x2 - x1;
    const ch = y2 - y1;
    if (cw < 30 || ch < 30) return boxes;

    const subGray = new Float32Array(cw * ch);
    const sampleVals = [];
    for (let yy = 0; yy < ch; yy++) {
      for (let xx = 0; xx < cw; xx++) {
        const v = gray[(y1 + yy) * imgW + (x1 + xx)];
        subGray[yy * cw + xx] = v;
        if ((yy + xx) % 4 === 0) sampleVals.push(v);
      }
    }
    sampleVals.sort((a, b) => a - b);
    if ((sampleVals[Math.floor(sampleVals.length / 2)] || 0) > 235) return boxes;

    const blur = fastBoxBlurFloat(subGray, cw, ch, 14);
    const thThresh = Math.max(16, 28 - sensitivity * 12);
    const bin = new Uint8Array(cw * ch);
    for (let i = 0; i < bin.length; i++) {
      if (subGray[i] - blur[i] >= thThresh) bin[i] = 1;
    }

    const { comps } = connectedComponents(bin, cw, ch);
    const pad = Math.max(4, dilatePx);
    for (const c of comps) {
      const aspect = c.width / Math.max(1, c.height);
      if (c.width >= 10 && c.width <= 60 && c.height >= 10 && c.height <= 60 && aspect >= 0.68 && aspect <= 1.45 && c.area >= 30 && c.area <= 2000) {
        let colorDiff = 0;
        let cnt = 0;
        for (let yy = c.y; yy < c.y + c.height; yy++) {
          for (let xx = c.x; xx < c.x + c.width; xx++) {
            const p = ((y1 + yy) * imgW + (x1 + xx)) * 4;
            colorDiff += Math.abs(imgData[p] - imgData[p + 1]) + Math.abs(imgData[p + 1] - imgData[p + 2]);
            cnt++;
          }
        }
        if (cnt > 0 && colorDiff / cnt < 55) {
          boxes.push({
            x: Math.max(0, x1 + c.x - pad),
            y: Math.max(0, y1 + c.y - pad),
            width: c.width + pad * 2,
            height: c.height + pad * 2,
            type: 'ai_sparkles',
            label: 'AI Corner Sparkle'
          });
        }
      }
    }
    return boxes;
  }

  function detectLetterboxInRegion(gray, imgW, rx, ry, rw, rh) {
    const boxes = [];
    let topH = 0;
    const maxTop = Math.floor(rh * 0.15);
    for (let r = 0; r < maxTop; r++) {
      let sum = 0;
      const rowOff = (ry + r) * imgW + rx;
      for (let c = 0; c < rw; c++) sum += gray[rowOff + c];
      if (sum / rw < 35) topH = r + 1;
      else if (topH > 0) break;
    }
    if (topH > 5) {
      boxes.push({
        x: rx,
        y: ry,
        width: rw,
        height: topH + 3,
        type: 'letterbox_bar',
        label: 'Top Letterbox Bar'
      });
    }

    let botH = 0;
    const minBot = Math.floor(rh * 0.85);
    for (let r = rh - 1; r >= minBot; r--) {
      let sum = 0;
      const rowOff = (ry + r) * imgW + rx;
      for (let c = 0; c < rw; c++) sum += gray[rowOff + c];
      if (sum / rw < 35) botH = rh - r;
      else if (botH > 0) break;
    }
    if (botH > 5) {
      boxes.push({
        x: rx,
        y: ry + rh - botH - 3,
        width: rw,
        height: botH + 3,
        type: 'letterbox_bar',
        label: 'Bottom Letterbox Bar'
      });
    }
    return boxes;
  }

  function computeOrientedPolygonFromPoints(pts, pad, imgW, imgH) {
    const n = pts.length;
    if (n < 12) return null;
    let sx = 0, sy = 0;
    for (let i = 0; i < n; i++) { sx += pts[i][0]; sy += pts[i][1]; }
    const mx = sx / n, my = sy / n;

    let sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
      const dx = pts[i][0] - mx, dy = pts[i][1] - my;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    const angleRad = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const angleDeg = Math.abs(angleRad * 180 / Math.PI);
    const ux = Math.cos(angleRad), uy = Math.sin(angleRad);
    const vx = -uy, vy = ux;
    let minU = 1e9, maxU = -1e9, minV = 1e9, maxV = -1e9;
    for (let i = 0; i < n; i++) {
      const dx = pts[i][0] - mx, dy = pts[i][1] - my;
      const pu = dx * ux + dy * uy;
      const pv = dx * vx + dy * vy;
      if (pu < minU) minU = pu;
      if (pu > maxU) maxU = pu;
      if (pv < minV) minV = pv;
      if (pv > maxV) maxV = pv;
    }
    minU -= pad; maxU += pad;
    minV -= pad; maxV += pad;
    const clampPt = (u, v) => [
      Math.max(0, Math.min(imgW - 1, Math.round(mx + u * ux + v * vx))),
      Math.max(0, Math.min(imgH - 1, Math.round(my + u * uy + v * vy)))
    ];
    return {
      angleRad,
      angleDeg,
      lengthU: maxU - minU,
      heightV: maxV - minV,
      polygon: [
        clampPt(minU, minV),
        clampPt(maxU, minV),
        clampPt(maxU, maxV),
        clampPt(minU, maxV)
      ]
    };
  }

  /**
   * Computes normalized cross-correlation (NCC) template matches across the full image
   * using a high-pass luminance template from the cleanest seed candidate box,
   * finding ALL repeating copies (even over face, hair, eyes, or clothing!).
   */
  function findRepeatingTemplatePeaks(hpGray, w, h, seedBox, minNcc = 0.40) {
    const sx0 = Math.max(0, Math.floor(seedBox.x));
    const sy0 = Math.max(0, Math.floor(seedBox.y));
    const sw = Math.min(w - sx0, Math.floor(seedBox.width));
    const sh = Math.min(h - sy0, Math.floor(seedBox.height));
    if (sw < 16 || sh < 16) return [];

    // Sample active high-pass template points inside seed polygon/bbox
    const activePts = [];
    let meanT = 0;
    const threshT = 4.0;
    for (let y = 0; y < sh; y++) {
      const off = (sy0 + y) * w + sx0;
      for (let x = 0; x < sw; x++) {
        const val = hpGray[off + x];
        if (Math.abs(val) >= threshT) {
          activePts.push([x, y, val]);
          meanT += val;
        }
      }
    }
    if (activePts.length < 30) return [];
    meanT /= activePts.length;
    let normT = 0;
    for (let i = 0; i < activePts.length; i++) {
      activePts[i][2] -= meanT;
      normT += activePts[i][2] * activePts[i][2];
    }
    normT = Math.sqrt(normT);
    if (normT < 1e-4) return [];

    // Subsample template points for fast coarse 2px scan (~200-400 points)
    const stepPt = Math.max(1, Math.floor(activePts.length / 260));
    const sparsePts = [];
    let sparseNorm = 0;
    for (let i = 0; i < activePts.length; i += stepPt) {
      sparsePts.push(activePts[i]);
      sparseNorm += activePts[i][2] * activePts[i][2];
    }
    sparseNorm = Math.sqrt(sparseNorm);

    // Allow partial border-cropped placements up to 30% outside edges
    const padX = Math.floor(sw * 0.30);
    const padY = Math.floor(sh * 0.30);
    const stride = 2;
    const candidates = [];

    for (let py = -padY; py <= h - sh + padY; py += stride) {
      for (let px = -padX; px <= w - sw + padX; px += stride) {
        let dot = 0, normI = 0, validCnt = 0;
        for (let k = 0; k < sparsePts.length; k++) {
          const [tx, ty, tv] = sparsePts[k];
          const gx = px + tx, gy = py + ty;
          if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue;
          const iv = hpGray[gy * w + gx];
          dot += iv * tv;
          normI += iv * iv;
          validCnt++;
        }
        if (validCnt < sparsePts.length * 0.45 || normI < 1e-3) continue;
        const ncc = dot / (Math.sqrt(normI) * sparseNorm);
        if (ncc >= minNcc * 0.85) {
          candidates.push({ px, py, ncc });
        }
      }
    }

    // Non-maximum suppression + exact 1px full-point refinement
    candidates.sort((a, b) => b.ncc - a.ncc);
    const peaks = [];
    const nmsDistSq = Math.pow(Math.max(18, Math.min(sw, sh) * 0.55), 2);
    for (const cand of candidates) {
      let tooClose = false;
      for (const pk of peaks) {
        const dx = cand.px - pk.px, dy = cand.py - pk.py;
        if (dx * dx + dy * dy < nmsDistSq) { tooClose = true; break; }
      }
      if (tooClose) continue;

      // Exact 1px refinement around coarse peak
      let bestX = cand.px, bestY = cand.py, bestNcc = -1;
      for (let ry = cand.py - 2; ry <= cand.py + 2; ry++) {
        for (let rx = cand.px - 2; rx <= cand.px + 2; rx++) {
          let dot = 0, normI = 0, validCnt = 0;
          for (let k = 0; k < activePts.length; k++) {
            const [tx, ty, tv] = activePts[k];
            const gx = rx + tx, gy = ry + ty;
            if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue;
            const iv = hpGray[gy * w + gx];
            dot += iv * tv;
            normI += iv * iv;
            validCnt++;
          }
          if (validCnt < activePts.length * 0.45 || normI < 1e-3) continue;
          const ncc = dot / (Math.sqrt(normI) * normT);
          if (ncc > bestNcc) {
            bestNcc = ncc; bestX = rx; bestY = ry;
          }
        }
      }
      if (bestNcc >= minNcc) {
        peaks.push({ px: bestX, py: bestY, ncc: bestNcc, sw, sh });
      }
      if (peaks.length >= 48) break;
    }

    return peaks;
  }

  /**
   * Detects both light AND dark translucent/diagonal watermarks without eroding
   * thin 1-2px glyph strokes, then expands any repeating pattern via NCC template
   * matching so diagonal strings over faces/hair/clothing are found too.
   */
  function detectTranslucentInRegion(gray, imgW, imgH, rx, ry, rw, rh, sensitivity, dilatePx) {
    const sub = new Float32Array(rw * rh);
    for (let y = 0; y < rh; y++) {
      const srcOff = (ry + y) * imgW + rx;
      const dstOff = y * rw;
      for (let x = 0; x < rw; x++) {
        sub[dstOff + x] = gray[srcOff + x];
      }
    }

    // Dual-scale Top-Hat (bright text) & Black-Hat (dark text) — preserves 1-2px strokes!
    const open2 = morphOpenGray(sub, rw, rh, 2);
    const open4 = morphOpenGray(sub, rw, rh, 4);
    const close2 = morphCloseGray(sub, rw, rh, 2);
    const close4 = morphCloseGray(sub, rw, rh, 4);

    const thresh = Math.max(8, 17 - sensitivity * 12);
    const bin = new Uint8Array(rw * rh);
    for (let i = 0; i < bin.length; i++) {
      const th = Math.max(sub[i] - open2[i], sub[i] - open4[i]);
      const bh = Math.max(close2[i] - sub[i], close4[i] - sub[i]);
      if (Math.max(th, bh) >= thresh) bin[i] = 1;
    }

    // Bridge nearby glyphs into words/phrases WITHOUT destroying thin strokes first
    const connected = dilateBinary(bin, rw, rh, 5, 4);
    const { labels, comps } = connectedComponents(connected, rw, rh);
    const pad = Math.max(3, Math.min(8, dilatePx));

    const boxes = [];
    for (const c of comps) {
      if (c.area < 45 || c.width < 14 || c.height < 10) continue;
      if (c.width > rw * 0.78 && c.height > rh * 0.78) continue;

      // Collect ONLY raw stroke pixels (`bin[idx] === 1`) inside this component for PCA & white-paper check
      const rawPts = [];
      let bgHigh = 0, darkLow = 0, samples = 0;
      for (let yy = c.y; yy < c.y + c.height; yy++) {
        const off = yy * rw;
        for (let xx = c.x; xx < c.x + c.width; xx++) {
          if (labels[off + xx] === c.id && bin[off + xx]) {
            rawPts.push([xx, yy]);
          }
          if ((yy & 1) === 0 && (xx & 1) === 0) {
            const v = sub[off + xx];
            if (v > 235) bgHigh++;
            if (v < 95) darkLow++;
            samples++;
          }
        }
      }
      if (rawPts.length < 22) continue;
      if (samples > 0 && (bgHigh / samples > 0.45 && darkLow / samples > 0.03)) {
        continue;
      }

      const orient = computeOrientedPolygonFromPoints(rawPts, pad, rw, rh);
      if (!orient) continue;

      // Reject giant amorphous texture blocks; keep sleek text lines & compact emblems
      const aspect = Math.max(orient.lengthU, 1) / Math.max(orient.heightV, 1);
      const fillRatio = rawPts.length / Math.max(1, orient.lengthU * orient.heightV);
      if (aspect < 1.2 && orient.lengthU > rw * 0.25) continue;
      if (fillRatio > 0.65 && orient.lengthU > rw * 0.25) continue;

      const bx = Math.max(0, rx + c.x - pad);
      const by = Math.max(0, ry + c.y - pad);
      const bw = Math.min(imgW - bx, c.width + pad * 2);
      const bh = Math.min(imgH - by, c.height + pad * 2);

      // Score candidate quality as a clean repeating seed (elongated + high crisp contrast + uniform bg)
      let bgVar = 0, meanSub = 0;
      for (let k = 0; k < rawPts.length; k++) {
        const [px, py] = rawPts[k];
        meanSub += open4[py * rw + px];
      }
      meanSub /= rawPts.length;
      for (let k = 0; k < rawPts.length; k++) {
        const [px, py] = rawPts[k];
        const d = open4[py * rw + px] - meanSub;
        bgVar += d * d;
      }
      bgVar = Math.sqrt(bgVar / rawPts.length);

      const box = {
        x: bx,
        y: by,
        width: bw,
        height: bh,
        type: 'translucent_grid',
        label: orient.angleDeg > 8 ? 'Diagonal Watermark' : 'Watermark Overlay',
        _seedScore: (orient.angleDeg > 8 ? 2.0 : 1.0) * Math.min(3.5, aspect) * Math.sqrt(rawPts.length) / (1.0 + bgVar * 0.08)
      };
      if (orient.angleDeg > 8) {
        box.polygon = orient.polygon.map(([px, py]) => [rx + px, ry + py]);
        box.angle = orient.angleDeg;
      }
      boxes.push(box);
    }

    // If we found clean candidate(s), run fast NCC template matching across the region to capture ALL repeating copies!
    if (boxes.length >= 1) {
      boxes.sort((a, b) => (b._seedScore || 0) - (a._seedScore || 0));
      const bestSeed = boxes[0];
      const hpSub = new Float32Array(rw * rh);
      const blur9 = fastBoxBlurFloat(sub, rw, rh, 6);
      for (let i = 0; i < hpSub.length; i++) hpSub[i] = sub[i] - blur9[i];

      const localSeed = {
        x: bestSeed.x - rx,
        y: bestSeed.y - ry,
        width: bestSeed.width,
        height: bestSeed.height
      };
      const peaks = findRepeatingTemplatePeaks(hpSub, rw, rh, localSeed, 0.36);
      if (peaks.length >= 3) {
        const expandedBoxes = [];
        for (const pk of peaks) {
          const bx = Math.max(0, rx + pk.px);
          const by = Math.max(0, ry + pk.py);
          const bw = Math.min(imgW - bx, pk.sw - Math.max(0, -pk.px));
          const bh = Math.min(imgH - by, pk.sh - Math.max(0, -pk.py));
          if (bw < 10 || bh < 10) continue;

          const b = {
            x: bx,
            y: by,
            width: bw,
            height: bh,
            type: 'translucent_grid',
            label: bestSeed.label
          };
          if (bestSeed.polygon) {
            const dx = (rx + pk.px) - bestSeed.x;
            const dy = (ry + pk.py) - bestSeed.y;
            b.polygon = bestSeed.polygon.map(([px, py]) => [
              Math.max(0, Math.min(imgW - 1, Math.round(px + dx))),
              Math.max(0, Math.min(imgH - 1, Math.round(py + dy)))
            ]);
            b.angle = bestSeed.angle;
          }
          expandedBoxes.push(b);
        }
        return expandedBoxes;
      }
    }

    for (const b of boxes) delete b._seedScore;
    return boxes;
  }

  function autoDetectClient(imageData, sensitivity = 0.5, targetType = 'all', dilatePx = 4) {
    const { data, width: w, height: h } = imageData;
    const gray = extractGray(data, w, h);
    const embedded = findEmbeddedPhotoRegions(gray, w, h);
    const scanRegions = embedded.length > 0 ? embedded : [{ x: 0, y: 0, width: w, height: h }];

    const boxes = [];
    if (targetType === 'all' || targetType === 'ai_sparkles') {
      for (const r of scanRegions) {
        boxes.push(...detectCornerSparklesInRegion(data, gray, w, r.x, r.y, r.width, r.height, sensitivity, dilatePx));
      }
    }
    if (targetType === 'all' || targetType === 'letterbox') {
      for (const r of scanRegions) {
        boxes.push(...detectLetterboxInRegion(gray, w, r.x, r.y, r.width, r.height));
      }
    }
    let gridCount = 0;
    if (targetType === 'all' || targetType === 'text' || targetType === 'translucent_grid') {
      for (const r of scanRegions) {
        const gridBoxes = detectTranslucentInRegion(gray, w, h, r.x, r.y, r.width, r.height, sensitivity, dilatePx);
        gridCount += gridBoxes.length;
        boxes.push(...gridBoxes);
      }
    }

    const diagCount = boxes.filter((b) => b.angle && b.angle > 8).length;
    const recommendedMode = (gridCount >= 3 || diagCount >= 2) ? 'deblend' : 'auto';
    return {
      detected_count: boxes.length,
      boxes,
      recommended_mode: recommendedMode
    };
  }

  /**
   * Computes background RGB values at masked pixels using normalized multi-scale
   * push-pull pyramid convolution strictly from unmasked pixels.
   */
  function harmonicMaskedFill(srcData, binMask, w, h) {
    const len = w * h;
    const rArr = new Float32Array(len);
    const gArr = new Float32Array(len);
    const bArr = new Float32Array(len);
    const valid = new Float32Array(len);

    for (let i = 0; i < len; i++) {
      if (!binMask[i]) {
        const p = i * 4;
        rArr[i] = srcData[p];
        gArr[i] = srcData[p + 1];
        bArr[i] = srcData[p + 2];
        valid[i] = 1.0;
      }
    }

    const levels = [{ w, h, r: rArr, g: gArr, b: bArr, v: valid }];
    let cur = levels[0];
    while (cur.w > 2 && cur.h > 2 && levels.length < 7) {
      const nw = Math.max(1, Math.ceil(cur.w / 2));
      const nh = Math.max(1, Math.ceil(cur.h / 2));
      const nr = new Float32Array(nw * nh);
      const ng = new Float32Array(nw * nh);
      const nb = new Float32Array(nw * nh);
      const nv = new Float32Array(nw * nh);

      for (let y = 0; y < nh; y++) {
        for (let x = 0; x < nw; x++) {
          let sr = 0, sg = 0, sb = 0, sw = 0;
          for (let dy = 0; dy < 2; dy++) {
            const sy = Math.min(cur.h - 1, y * 2 + dy);
            for (let dx = 0; dx < 2; dx++) {
              const sx = Math.min(cur.w - 1, x * 2 + dx);
              const si = sy * cur.w + sx;
              const weight = cur.v[si];
              if (weight > 0) {
                sr += cur.r[si] * weight;
                sg += cur.g[si] * weight;
                sb += cur.b[si] * weight;
                sw += weight;
              }
            }
          }
          const di = y * nw + x;
          if (sw > 0) {
            nr[di] = sr / sw;
            ng[di] = sg / sw;
            nb[di] = sb / sw;
            nv[di] = Math.min(1.0, sw * 0.25);
          }
        }
      }
      cur = { w: nw, h: nh, r: nr, g: ng, b: nb, v: nv };
      levels.push(cur);
    }

    for (let lvl = levels.length - 2; lvl >= 0; lvl--) {
      const fine = levels[lvl];
      const coarse = levels[lvl + 1];
      for (let y = 0; y < fine.h; y++) {
        const cy = (y + 0.5) * (coarse.h / fine.h) - 0.5;
        const y0 = Math.max(0, Math.min(coarse.h - 1, Math.floor(cy)));
        const y1 = Math.max(0, Math.min(coarse.h - 1, y0 + 1));
        const fy = Math.max(0, Math.min(1, cy - y0));

        for (let x = 0; x < fine.w; x++) {
          const fi = y * fine.w + x;
          if (fine.v[fi] >= 0.999) continue;

          const cx = (x + 0.5) * (coarse.w / fine.w) - 0.5;
          const x0 = Math.max(0, Math.min(coarse.w - 1, Math.floor(cx)));
          const x1 = Math.max(0, Math.min(coarse.w - 1, x0 + 1));
          const fx = Math.max(0, Math.min(1, cx - x0));

          const i00 = y0 * coarse.w + x0, i10 = y0 * coarse.w + x1;
          const i01 = y1 * coarse.w + x0, i11 = y1 * coarse.w + x1;
          const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
          const norm = w00 * (coarse.v[i00] > 0) + w10 * (coarse.v[i10] > 0) + w01 * (coarse.v[i01] > 0) + w11 * (coarse.v[i11] > 0);
          if (norm > 1e-5) {
            const cr = (w00 * (coarse.v[i00] > 0 ? coarse.r[i00] : 0) + w10 * (coarse.v[i10] > 0 ? coarse.r[i10] : 0) + w01 * (coarse.v[i01] > 0 ? coarse.r[i01] : 0) + w11 * (coarse.v[i11] > 0 ? coarse.r[i11] : 0)) / norm;
            const cg = (w00 * (coarse.v[i00] > 0 ? coarse.g[i00] : 0) + w10 * (coarse.v[i10] > 0 ? coarse.g[i10] : 0) + w01 * (coarse.v[i01] > 0 ? coarse.g[i01] : 0) + w11 * (coarse.v[i11] > 0 ? coarse.g[i11] : 0)) / norm;
            const cb = (w00 * (coarse.v[i00] > 0 ? coarse.b[i00] : 0) + w10 * (coarse.v[i10] > 0 ? coarse.b[i10] : 0) + w01 * (coarse.v[i01] > 0 ? coarse.b[i01] : 0) + w11 * (coarse.v[i11] > 0 ? coarse.b[i11] : 0)) / norm;
            const mix = fine.v[fi];
            fine.r[fi] = mix * fine.r[fi] + (1 - mix) * cr;
            fine.g[fi] = mix * fine.g[fi] + (1 - mix) * cg;
            fine.b[fi] = mix * fine.b[fi] + (1 - mix) * cb;
            fine.v[fi] = 1.0;
          }
        }
      }
    }

    return { rArr, gArr, bArr };
  }

  function clientContentAwareInpaint(srcImgData, binMask, w, h, pdePasses = 12) {
    const src = srcImgData.data;
    const out = new Uint8ClampedArray(src);
    const { rArr, gArr, bArr } = harmonicMaskedFill(src, binMask, w, h);

    const maskedIdx = [];
    for (let i = 0; i < binMask.length; i++) {
      if (binMask[i]) maskedIdx.push(i);
    }
    if (maskedIdx.length === 0) return srcImgData;

    for (let pass = 0; pass < pdePasses; pass++) {
      for (let m = 0; m < maskedIdx.length; m++) {
        const idx = maskedIdx[m];
        const x = idx % w;
        const y = (idx - x) / w;
        let sr = 0, sg = 0, sb = 0, cnt = 0;
        if (x > 0) { sr += rArr[idx - 1]; sg += gArr[idx - 1]; sb += bArr[idx - 1]; cnt++; }
        if (x + 1 < w) { sr += rArr[idx + 1]; sg += gArr[idx + 1]; sb += bArr[idx + 1]; cnt++; }
        if (y > 0) { sr += rArr[idx - w]; sg += gArr[idx - w]; sb += bArr[idx - w]; cnt++; }
        if (y + 1 < h) { sr += rArr[idx + w]; sg += gArr[idx + w]; sb += bArr[idx + w]; cnt++; }
        if (cnt > 0) {
          rArr[idx] = sr / cnt;
          gArr[idx] = sg / cnt;
          bArr[idx] = sb / cnt;
        }
      }
    }

    const dilatedOuter = dilateBinary(binMask, w, h, 8);
    const blurGray = fastBoxBlurFloat(extractGray(src, w, h), w, h, 2);
    let sqSum = 0, ringCnt = 0;
    for (let i = 0; i < binMask.length; i++) {
      if (dilatedOuter[i] && !binMask[i]) {
        const p = i * 4;
        const gVal = rgbToY(src[p], src[p + 1], src[p + 2]);
        const d = gVal - blurGray[i];
        sqSum += d * d;
        ringCnt++;
      }
    }
    const noiseStd = ringCnt > 20 ? Math.min(6.0, Math.sqrt(sqSum / ringCnt) * 0.45) : 1.2;

    let seed = 1337;
    const nextNoise = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const u1 = ((seed & 0xffff) / 65535.0) - 0.5;
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const u2 = ((seed & 0xffff) / 65535.0) - 0.5;
      return (u1 + u2) * noiseStd;
    };

    for (let m = 0; m < maskedIdx.length; m++) {
      const idx = maskedIdx[m];
      const n = nextNoise();
      const p = idx * 4;
      out[p] = clamp255(rArr[idx] + n);
      out[p + 1] = clamp255(gArr[idx] + n);
      out[p + 2] = clamp255(bArr[idx] + n);
      out[p + 3] = 255;
    }

    return new ImageData(out, w, h);
  }

  /**
   * Dekel-style Multi-Instance Median Matte Recovery in Browser JS:
   * Aligns >=3 repeating copies inside the mask (or auto-discovered via NCC),
   * extracts the exact per-pixel median matte (Y, Cb, Cr) so background textures
   * cancel out completely, and removes all copies with inverse alpha recovery.
   */
  function tryDekelMultiInstanceClient(srcImgData, binMask, w, h, colorHint) {
    const src = srcImgData.data;
    const { comps } = connectedComponents(binMask, w, h);
    if (comps.length < 3) return null;

    const ySrc = extractGray(src, w, h);
    const blur6 = fastBoxBlurFloat(ySrc, w, h, 6);
    const hpY = new Float32Array(w * h);
    for (let i = 0; i < hpY.length; i++) hpY[i] = ySrc[i] - blur6[i];

    // Score each masked component to pick the cleanest template seed
    let bestComp = null;
    let bestScore = -1;
    for (const c of comps) {
      if (c.width < 16 || c.height < 16 || c.area < 80) continue;
      let hpEnergy = 0, bgVar = 0, meanBg = 0, cnt = 0;
      for (let yy = c.y; yy < c.y + c.height; yy++) {
        const off = yy * w;
        for (let xx = c.x; xx < c.x + c.width; xx++) {
          const idx = off + xx;
          if (!binMask[idx]) continue;
          hpEnergy += Math.abs(hpY[idx]);
          meanBg += blur6[idx];
          cnt++;
        }
      }
      if (cnt < 40) continue;
      meanBg /= cnt;
      for (let yy = c.y; yy < c.y + c.height; yy++) {
        const off = yy * w;
        for (let xx = c.x; xx < c.x + c.width; xx++) {
          const idx = off + xx;
          if (!binMask[idx]) continue;
          const d = blur6[idx] - meanBg;
          bgVar += d * d;
        }
      }
      const stdBg = Math.sqrt(bgVar / cnt);
      const score = (hpEnergy / cnt) * Math.sqrt(cnt) / (1.0 + stdBg * 0.1);
      if (score > bestScore) {
        bestScore = score;
        bestComp = c;
      }
    }
    if (!bestComp) return null;

    const peaks = findRepeatingTemplatePeaks(hpY, w, h, bestComp, 0.36);
    if (peaks.length < 3) return null;

    const pw = bestComp.width;
    const ph = bestComp.height;
    const patchLen = pw * ph;

    // Determine watermark target polarity & color
    let sumSigned = 0, sCnt = 0;
    for (const pk of peaks) {
      for (let y = 0; y < ph; y++) {
        const gy = pk.py + y;
        if (gy < 0 || gy >= h) continue;
        for (let x = 0; x < pw; x++) {
          const gx = pk.px + x;
          if (gx < 0 || gx >= w) continue;
          const val = hpY[gy * w + gx];
          if (Math.abs(val) > 5.0) { sumSigned += val; sCnt++; }
        }
      }
    }

    let isLight = sCnt === 0 || sumSigned >= 0;
    let wmTargetR = isLight ? 255.0 : 0.0;
    let wmTargetG = wmTargetR, wmTargetB = wmTargetR;
    if (colorHint && colorHint !== 'auto' && colorHint.startsWith('#') && colorHint.length === 7) {
      wmTargetR = parseInt(colorHint.slice(1, 3), 16);
      wmTargetG = parseInt(colorHint.slice(3, 5), 16);
      wmTargetB = parseInt(colorHint.slice(5, 7), 16);
      isLight = rgbToY(wmTargetR, wmTargetG, wmTargetB) > 128.0;
    }
    const wmTargetY = rgbToY(wmTargetR, wmTargetG, wmTargetB);

    // Step 1: Identify true watermark stroke footprint via median high-pass across aligned copies
    const medHp = new Float32Array(patchLen);
    const hpSamples = new Float32Array(peaks.length);
    for (let i = 0; i < patchLen; i++) {
      const lx = i % pw;
      const ly = (i - lx) / pw;
      let k = 0;
      for (let pIdx = 0; pIdx < peaks.length; pIdx++) {
        const gx = peaks[pIdx].px + lx;
        const gy = peaks[pIdx].py + ly;
        if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue;
        const val = hpY[gy * w + gx];
        hpSamples[k++] = isLight ? val : -val;
      }
      if (k >= 2) {
        const sub = hpSamples.subarray(0, k).slice().sort();
        medHp[i] = sub[Math.floor(k * 0.5)];
      }
    }

    // Build tight binary stroke template (dilated 1px ONLY so thin letters never fuse into blur bands)
    const rawStroke = new Uint8Array(patchLen);
    let rawStrokeCount = 0;
    for (let i = 0; i < patchLen; i++) {
      if (medHp[i] >= 4.0) {
        rawStroke[i] = 1;
        rawStrokeCount++;
      }
    }
    if (rawStrokeCount < 20) return null;
    const strokePatchMask = dilateBinary(rawStroke, pw, ph, 1);

    // Check how many user-masked pixels overlap the watermark peaks
    let overlapPeaks = 0;
    for (const pk of peaks) {
      let hit = 0;
      for (let y = 0; y < ph; y++) {
        const gy = pk.py + y;
        if (gy < 0 || gy >= h) continue;
        for (let x = 0; x < pw; x++) {
          const gx = pk.px + x;
          if (gx < 0 || gx >= w) continue;
          if (binMask[gy * w + gx] && strokePatchMask[y * pw + x]) { hit++; break; }
        }
        if (hit) break;
      }
      if (hit) overlapPeaks++;
    }

    // Step 2: Project exact stroke mask globally and compute UNCONTAMINATED background via harmonicMaskedFill
    const globalStrokeMask = new Uint8Array(w * h);
    for (const pk of peaks) {
      for (let y = 0; y < ph; y++) {
        const gy = pk.py + y;
        if (gy < 0 || gy >= h) continue;
        for (let x = 0; x < pw; x++) {
          const gx = pk.px + x;
          if (gx < 0 || gx >= w) continue;
          const gIdx = gy * w + gx;
          if (overlapPeaks <= 1 && !binMask[gIdx]) continue;
          if (strokePatchMask[y * pw + x]) globalStrokeMask[gIdx] = 1;
        }
      }
    }

    const cleanBg = harmonicMaskedFill(src, globalStrokeMask, w, h);

    // Step 3: Compute uncontaminated per-pixel median alpha across aligned copies
    const medianAlpha = new Float32Array(patchLen);
    const samplesArr = new Float32Array(peaks.length);
    for (let i = 0; i < patchLen; i++) {
      if (!strokePatchMask[i]) continue;
      const lx = i % pw;
      const ly = (i - lx) / pw;
      let k = 0;
      for (let pIdx = 0; pIdx < peaks.length; pIdx++) {
        const gx = peaks[pIdx].px + lx;
        const gy = peaks[pIdx].py + ly;
        if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue;
        const idx = gy * w + gx;
        const bgY = rgbToY(cleanBg.rArr[idx], cleanBg.gArr[idx], cleanBg.bArr[idx]);
        const dev = isLight ? (ySrc[idx] - bgY) : (bgY - ySrc[idx]);
        const denom = isLight ? Math.max(20.0, 255.0 - bgY) : Math.max(20.0, bgY);
        samplesArr[k++] = Math.max(0.0, Math.min(0.75, dev / denom));
      }
      if (k >= 2) {
        const sub = samplesArr.subarray(0, k).slice().sort();
        medianAlpha[i] = sub[Math.floor(k * 0.5)];
      }
    }

    // Assemble global alpha map via max across peaks (prevents double-subtraction)
    const globalAlpha = new Float32Array(w * h);
    for (const pk of peaks) {
      for (let y = 0; y < ph; y++) {
        const gy = pk.py + y;
        if (gy < 0 || gy >= h) continue;
        for (let x = 0; x < pw; x++) {
          const gx = pk.px + x;
          if (gx < 0 || gx >= w) continue;
          const gIdx = gy * w + gx;
          if (!globalStrokeMask[gIdx]) continue;
          const a = medianAlpha[y * pw + x];
          if (a > globalAlpha[gIdx]) globalAlpha[gIdx] = a;
        }
      }
    }

    // Step 4: Estimate background texture std on CLEAN background (zero watermark leakage!)
    const cleanY = new Float32Array(w * h);
    const cleanYSq = new Float32Array(w * h);
    for (let i = 0; i < cleanY.length; i++) {
      const v = rgbToY(cleanBg.rArr[i], cleanBg.gArr[i], cleanBg.bArr[i]);
      cleanY[i] = v;
      cleanYSq[i] = v * v;
    }
    const b1 = fastBoxBlurFloat(cleanY, w, h, 3);
    const b2 = fastBoxBlurFloat(cleanYSq, w, h, 3);

    const out = new Uint8ClampedArray(src);
    const smoothHole = new Uint8Array(w * h);
    let hasSmooth = false;

    for (let i = 0; i < globalStrokeMask.length; i++) {
      if (!globalStrokeMask[i]) continue;
      const rawA = globalAlpha[i];
      const p = i * 4;
      const bgStd = Math.sqrt(Math.max(0, b2[i] - b1[i] * b1[i]));

      if (bgStd < 6.5) {
        // Enclose entire stroke + JPEG ringing rim on smooth sky/clouds cleanly in harmonic fill
        smoothHole[i] = 1;
        hasSmooth = true;
      } else if (rawA >= 0.04) {
        // Textured hair, eyes, teeth & skin folds: YCbCr inverse alpha preserves natural skin/hair chroma
        const a = Math.min(0.45, rawA * 0.85);
        const inv = Math.max(0.55, 1.0 - a);
        const r = src[p], g = src[p + 1], b = src[p + 2];
        const yVal = 0.299 * r + 0.587 * g + 0.114 * b;
        const cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
        const cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

        let yRec = (yVal - a * wmTargetY) / inv;
        // Clamp luminance deviation relative to clean local bg to prevent dark JPEG crush dots
        if (isLight) yRec = Math.max(cleanY[i] - 28.0, yRec);
        else yRec = Math.min(cleanY[i] + 28.0, yRec);

        const cbRec = 128.0 + (cb - 128.0) / Math.max(0.65, 1.0 - a * 0.7);
        const crRec = 128.0 + (cr - 128.0) / Math.max(0.65, 1.0 - a * 0.7);

        out[p] = clamp255(yRec + 1.402 * (crRec - 128.0));
        out[p + 1] = clamp255(yRec - 0.344136 * (cbRec - 128.0) - 0.714136 * (crRec - 128.0));
        out[p + 2] = clamp255(yRec + 1.772 * (cbRec - 128.0));
      }
    }

    // 3x3 median spike suppression on textured high-alpha strokes
    const medCleaned = new Uint8ClampedArray(out);
    const tmp9 = new Float32Array(9);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        if (smoothHole[i] || globalAlpha[i] <= 0.12) continue;
        const p = i * 4;
        for (let ch = 0; ch < 3; ch++) {
          let k = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              tmp9[k++] = out[((y + dy) * w + (x + dx)) * 4 + ch];
            }
          }
          tmp9.sort();
          const med = tmp9[4];
          if (Math.abs(out[p + ch] - med) > 7.5) medCleaned[p + ch] = med;
        }
      }
    }

    if (hasSmooth) {
      return clientContentAwareInpaint(new ImageData(medCleaned, w, h), smoothHole, w, h, 8);
    }
    return new ImageData(medCleaned, w, h);
  }

  /**
   * Single-region Stroke-Aware Inverse Alpha Recovery when watermark doesn't repeat:
   * isolates stroke ink via Top-Hat/Black-Hat strictly inside mask so cloud/object edges
   * are never smeared.
   */
  function clientStrokeDeblend(srcImgData, binMask, w, h, colorHint, alphaHint = 0.5) {
    const dekelResult = tryDekelMultiInstanceClient(srcImgData, binMask, w, h, colorHint);
    if (dekelResult) return dekelResult;

    const src = srcImgData.data;
    const out = new Uint8ClampedArray(src);
    const len = w * h;
    const ySrc = extractGray(src, w, h);
    const open3 = morphOpenGray(ySrc, w, h, 3);
    const close3 = morphCloseGray(ySrc, w, h, 3);

    let isLight = true;
    let wmTargetR = 255.0, wmTargetG = 255.0, wmTargetB = 255.0, wmTargetY = 255.0;
    if (colorHint && colorHint !== 'auto' && colorHint.startsWith('#') && colorHint.length === 7) {
      wmTargetR = parseInt(colorHint.slice(1, 3), 16);
      wmTargetG = parseInt(colorHint.slice(3, 5), 16);
      wmTargetB = parseInt(colorHint.slice(5, 7), 16);
      wmTargetY = rgbToY(wmTargetR, wmTargetG, wmTargetB);
      isLight = wmTargetY > 128.0;
    } else {
      let sumTH = 0, sumBH = 0;
      for (let i = 0; i < len; i++) {
        if (!binMask[i]) continue;
        sumTH += Math.max(0, ySrc[i] - open3[i]);
        sumBH += Math.max(0, close3[i] - ySrc[i]);
      }
      isLight = sumTH >= sumBH;
      wmTargetY = isLight ? 255.0 : 0.0;
      wmTargetR = wmTargetG = wmTargetB = wmTargetY;
    }

    const strokeMask = new Uint8Array(len);
    let strokeCnt = 0;
    for (let i = 0; i < len; i++) {
      if (!binMask[i]) continue;
      const dev = isLight ? (ySrc[i] - open3[i]) : (close3[i] - ySrc[i]);
      if (dev >= 5.5) {
        strokeMask[i] = 1;
        strokeCnt++;
      }
    }
    if (strokeCnt < 8) {
      return clientContentAwareInpaint(srcImgData, binMask, w, h, 14);
    }

    const softStroke = dilateBinary(strokeMask, w, h, 1);
    for (let i = 0; i < len; i++) {
      if (!binMask[i]) softStroke[i] = 0;
    }

    const cleanBg = harmonicMaskedFill(src, softStroke, w, h);
    const bgY = new Float32Array(len);
    const bgYSq = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      const val = rgbToY(cleanBg.rArr[i], cleanBg.gArr[i], cleanBg.bArr[i]);
      bgY[i] = val;
      bgYSq[i] = val * val;
    }
    const m1 = fastBoxBlurFloat(bgY, w, h, 4);
    const m2 = fastBoxBlurFloat(bgYSq, w, h, 4);

    const smoothHole = new Uint8Array(len);
    let hasSmooth = false;
    for (let i = 0; i < len; i++) {
      if (!softStroke[i]) continue;
      const denom = isLight ? Math.max(24.0, wmTargetY - bgY[i]) : Math.max(24.0, bgY[i] - wmTargetY);
      const diff = isLight ? (ySrc[i] - bgY[i]) : (bgY[i] - ySrc[i]);
      const a = Math.max(0.0, Math.min(Math.max(0.35, alphaHint * 1.4), diff / denom));
      if (a < 0.04) continue;

      const bgStd = Math.sqrt(Math.max(0, m2[i] - m1[i] * m1[i]));
      if (bgStd < 4.5) {
        smoothHole[i] = 1;
        hasSmooth = true;
      } else {
        const inv = 1.0 - a;
        const p = i * 4;
        out[p] = clamp255((src[p] - a * wmTargetR) / inv);
        out[p + 1] = clamp255((src[p + 1] - a * wmTargetG) / inv);
        out[p + 2] = clamp255((src[p + 2] - a * wmTargetB) / inv);
      }
    }

    if (hasSmooth) {
      return clientContentAwareInpaint(new ImageData(out, w, h), smoothHole, w, h, 8);
    }
    return new ImageData(out, w, h);
  }

  function processImageDataClient(srcImgData, maskImgData, mode = 'auto', colorHint = 'auto', alphaHint = 0.5) {
    const w = srcImgData.width;
    const h = srcImgData.height;
    const binMask = new Uint8Array(w * h);
    const mData = maskImgData.data;
    let maskPixels = 0;
    for (let i = 0, p = 3; i < binMask.length; i++, p += 4) {
      if (mData[p] > 10) {
        binMask[i] = 1;
        maskPixels++;
      }
    }
    if (maskPixels === 0) return srcImgData;

    if (mode === 'deblend') {
      return clientStrokeDeblend(srcImgData, binMask, w, h, colorHint, alphaHint);
    }
    if (mode === 'inpaint_telea') {
      return clientContentAwareInpaint(srcImgData, binMask, w, h, 10);
    }
    if (mode === 'inpaint_ns') {
      return clientContentAwareInpaint(srcImgData, binMask, w, h, 24);
    }

    const { comps } = connectedComponents(binMask, w, h);
    if (comps.length >= 3) {
      return clientStrokeDeblend(srcImgData, binMask, w, h, colorHint, alphaHint);
    }
    return clientContentAwareInpaint(srcImgData, binMask, w, h, 14);
  }

  async function uploadClientFile(file) {
    const sessionId = 'local-' + Math.random().toString(36).slice(2, 10);
    const isPdf = file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf';

    if (isPdf) {
      const pdfjsLib = await ensurePdfJs();
      const buf = await file.arrayBuffer();
      const pdfDoc = await pdfjsLib.getDocument({ data: buf }).promise;
      clientSessions[sessionId] = {
        fileType: 'pdf',
        filename: file.name,
        pageCount: pdfDoc.numPages,
        pdfDoc,
        pageCache: {}
      };
      return {
        session_id: sessionId,
        file_type: 'pdf',
        filename: file.name,
        page_count: pdfDoc.numPages
      };
    } else {
      const objectUrl = URL.createObjectURL(file);
      clientSessions[sessionId] = {
        fileType: 'image',
        filename: file.name,
        pageCount: 1,
        objectUrl
      };
      return {
        session_id: sessionId,
        file_type: 'image',
        filename: file.name,
        page_count: 1
      };
    }
  }

  async function getClientPageUrl(sessionId, pageNum) {
    const sess = clientSessions[sessionId];
    if (!sess) throw new Error('Session not found');
    if (sess.fileType === 'image') return sess.objectUrl;
    if (sess.pageCache[pageNum]) return sess.pageCache[pageNum];

    const page = await sess.pdfDoc.getPage(pageNum + 1);
    const viewport = page.getViewport({ scale: 2.0 });
    const c = document.createElement('canvas');
    c.width = Math.round(viewport.width);
    c.height = Math.round(viewport.height);
    const ctx = c.getContext('2d');
    await page.render({ canvasContext: ctx, viewport }).promise;
    const url = c.toDataURL('image/png');
    sess.pageCache[pageNum] = url;
    return url;
  }

  async function processAllPdfClient(sessionId, templateMaskDataUrl, pageMasksMap, mode, colorHint, alphaHint = 0.5) {
    const sess = clientSessions[sessionId];
    if (!sess || sess.fileType !== 'pdf') throw new Error('PDF session not found');
    const PDFLib = await ensurePdfLib();
    const outPdf = await PDFLib.PDFDocument.create();

    const loadImg = (url) => new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = reject;
      im.src = url;
    });

    for (let p = 0; p < sess.pageCount; p++) {
      const pageUrl = await getClientPageUrl(sessionId, p);
      const pageImg = await loadImg(pageUrl);
      const pw = pageImg.naturalWidth;
      const ph = pageImg.naturalHeight;

      const c = document.createElement('canvas');
      c.width = pw;
      c.height = ph;
      const ctx = c.getContext('2d');
      ctx.drawImage(pageImg, 0, 0);

      const maskUrl = (pageMasksMap && pageMasksMap[p]) ? pageMasksMap[p] : templateMaskDataUrl;
      if (maskUrl) {
        const mImg = await loadImg(maskUrl);
        const mc = document.createElement('canvas');
        mc.width = pw;
        mc.height = ph;
        const mctx = mc.getContext('2d');
        mctx.drawImage(mImg, 0, 0, pw, ph);

        const srcData = ctx.getImageData(0, 0, pw, ph);
        const maskData = mctx.getImageData(0, 0, pw, ph);
        const cleaned = processImageDataClient(srcData, maskData, mode, colorHint, alphaHint);
        ctx.putImageData(cleaned, 0, 0);
      }

      const pngBytes = await new Promise((resolve) => c.toBlob((b) => b.arrayBuffer().then(resolve), 'image/png'));
      const embedded = await outPdf.embedPng(pngBytes);
      const ptW = pw * 0.5;
      const ptH = ph * 0.5;
      const pdfPage = outPdf.addPage([ptW, ptH]);
      pdfPage.drawImage(embedded, { x: 0, y: 0, width: ptW, height: ptH });
    }

    const pdfBytes = await outPdf.save();
    return new Blob([pdfBytes], { type: 'application/pdf' });
  }

  global.EraseonClientEngine = {
    uploadClientFile,
    getClientPageUrl,
    autoDetectClient,
    processImageDataClient,
    processAllPdfClient
  };
})(window);
