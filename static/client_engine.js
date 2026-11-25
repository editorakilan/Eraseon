/**
 * Eraseon - Pure Browser Client-Side Engine (High-Fidelity Edition)
 * Runs 100% serverless on GitHub Pages (images & multi-page PDFs) with:
 * - Resolution-adaptive Top-Hat / Black-Hat + PCA oriented polygon + NCC template matching
 * - Sub-pixel (0.25px) Dekel-style Multi-Instance Median Matte Recovery in YCbCr
 *   including negative-alpha JPEG ringing cancellation & independent Cb/Cr recovery
 * - Multi-scale Push-Pull Harmonic Pyramid + Border-Guided Exemplar Patch Synthesis + Sensor-Grain
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

  function sampleBilinearFloat(arr, w, h, x, y) {
    const cx = Math.max(0, Math.min(w - 1.0001, x));
    const cy = Math.max(0, Math.min(h - 1.0001, y));
    const x0 = Math.floor(cx), y0 = Math.floor(cy);
    const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
    const fx = cx - x0, fy = cy - y0;
    const v00 = arr[y0 * w + x0], v10 = arr[y0 * w + x1];
    const v01 = arr[y1 * w + x0], v11 = arr[y1 * w + x1];
    return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
  }

  function sampleBilinearRGB(data, w, h, x, y) {
    const cx = Math.max(0, Math.min(w - 1.0001, x));
    const cy = Math.max(0, Math.min(h - 1.0001, y));
    const x0 = Math.floor(cx), y0 = Math.floor(cy);
    const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
    const fx = cx - x0, fy = cy - y0;
    const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    const i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4;
    const i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
    return [
      data[i00] * w00 + data[i10] * w10 + data[i01] * w01 + data[i11] * w11,
      data[i00 + 1] * w00 + data[i10 + 1] * w10 + data[i01 + 1] * w01 + data[i11 + 1] * w11,
      data[i00 + 2] * w00 + data[i10 + 2] * w10 + data[i01 + 2] * w01 + data[i11 + 2] * w11
    ];
  }

  function resizeGrayBilinear(gray, sw, sh, dw, dh) {
    const out = new Float32Array(dw * dh);
    const sx = sw / dw, sy = sh / dh;
    for (let y = 0; y < dh; y++) {
      const gy = (y + 0.5) * sy - 0.5;
      for (let x = 0; x < dw; x++) {
        out[y * dw + x] = sampleBilinearFloat(gray, sw, sh, (x + 0.5) * sx - 0.5, gy);
      }
    }
    return out;
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
    const y1 = ry + Math.floor(rh * 0.70);
    let y2 = ry + rh;
    // Trim bottom dark letterbox rows so bottom-right stars don't merge into letterbox contrast edge
    while (y2 > y1 + 32) {
      let rowSum = 0;
      const off = (y2 - 1) * imgW + rx;
      for (let x = 0; x < rw; x++) rowSum += gray[off + x];
      if (rowSum / rw < 48) y2--;
      else break;
    }
    const x1 = rx + Math.floor(rw * 0.72);
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

    const scaleR = Math.max(12, Math.round(Math.min(rw, rh) * 0.045));
    const blur = fastBoxBlurFloat(subGray, cw, ch, scaleR);
    const thThresh = Math.max(12, 21 - sensitivity * 11);
    const bin = new Uint8Array(cw * ch);
    for (let i = 0; i < bin.length; i++) {
      if (subGray[i] - blur[i] >= thThresh) bin[i] = 1;
    }

    const minSize = Math.max(10, Math.round(Math.min(rw, rh) * 0.025));
    const maxSize = Math.max(64, Math.round(Math.min(rw, rh) * 0.14));
    const { comps } = connectedComponents(bin, cw, ch);
    const pad = Math.max(5, dilatePx);
    for (const c of comps) {
      const aspect = c.width / Math.max(1, c.height);
      if (c.width >= minSize && c.width <= maxSize && c.height >= minSize && c.height <= maxSize && aspect >= 0.70 && aspect <= 1.42 && c.area >= 35) {
        let colorDiff = 0;
        let maxBright = 0;
        let cnt = 0;
        for (let yy = c.y; yy < c.y + c.height; yy++) {
          for (let xx = c.x; xx < c.x + c.width; xx++) {
            const p = ((y1 + yy) * imgW + (x1 + xx)) * 4;
            colorDiff += Math.abs(imgData[p] - imgData[p + 1]) + Math.abs(imgData[p + 1] - imgData[p + 2]);
            const v = subGray[yy * cw + xx];
            if (v > maxBright) maxBright = v;
            cnt++;
          }
        }
        // Real AI corner stars reach near-white peak (>= 205) and neutral tint
        if (cnt > 0 && colorDiff / cnt < 50 && maxBright >= 200) {
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
    const maxTop = Math.floor(rh * 0.16);
    for (let r = 0; r < maxTop; r++) {
      let sum = 0;
      const rowOff = (ry + r) * imgW + rx;
      for (let c = 0; c < rw; c++) sum += gray[rowOff + c];
      if (sum / rw < 48) topH = r + 1;
      else if (topH > 0) break;
    }
    if (topH > 5) {
      boxes.push({
        x: rx,
        y: ry,
        width: rw,
        height: Math.min(rh, topH + 8),
        type: 'letterbox_bar',
        label: 'Top Letterbox Bar'
      });
    }

    let botH = 0;
    const minBot = Math.floor(rh * 0.84);
    for (let r = rh - 1; r >= minBot; r--) {
      let sum = 0;
      const rowOff = (ry + r) * imgW + rx;
      for (let c = 0; c < rw; c++) sum += gray[rowOff + c];
      if (sum / rw < 48) botH = rh - r;
      else if (botH > 0) break;
    }
    if (botH > 5) {
      const yTop = Math.max(ry, ry + rh - botH - 8);
      boxes.push({
        x: rx,
        y: yTop,
        width: rw,
        height: (ry + rh) - yTop,
        type: 'letterbox_bar',
        label: 'Bottom Letterbox Bar'
      });
    }
    return boxes;
  }

  function detectContainerBadgesInRegion(gray, imgW, rx, ry, rw, rh) {
    const boxes = [];
    // Scan top band below any full-width letterbox bar for dark rounded pill overlays
    const yStart = Math.floor(rh * 0.01);
    const yEnd = Math.floor(rh * 0.22);
    const xEnd = Math.floor(rw * 0.55);
    const subH = yEnd - yStart;
    if (xEnd < 40 || subH < 18) return boxes;

    const pillMask = new Uint8Array(xEnd * subH);
    for (let y = 0; y < subH; y++) {
      const rowOff = (ry + yStart + y) * imgW + rx;
      let fullRowDark = 0;
      for (let x = 0; x < rw; x++) {
        if (gray[rowOff + x] < 85) fullRowDark++;
      }
      // Skip full-width letterbox rows so badge pill stays cleanly separated from top bar
      if (fullRowDark > rw * 0.68) continue;
      for (let x = 0; x < xEnd; x++) {
        if (gray[rowOff + x] < 95) pillMask[y * xEnd + x] = 1;
      }
    }
    const { comps } = connectedComponents(pillMask, xEnd, subH);
    for (const c of comps) {
      const aspect = c.width / Math.max(1, c.height);
      const fill = c.area / Math.max(1, c.width * c.height);
      if (c.width >= 50 && c.width <= rw * 0.48 && c.height >= 16 && c.height <= 64 && aspect >= 2.2 && fill >= 0.52) {
        boxes.push({
          x: Math.max(0, rx + c.x - 8),
          y: Math.max(0, ry + yStart + c.y - 8),
          width: Math.min(rw - Math.max(0, c.x - 8), c.width + 16),
          height: Math.min(rh - Math.max(0, yStart + c.y - 8), c.height + 16),
          type: 'container_badge',
          label: 'Aspect Ratio Badge'
        });
      }
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
   * Fast coarse-to-fine NCC template matcher + 0.25px quadratic/grid sub-pixel refinement.
   * Automatically scales search to <= 640px proxy so high-res photos & PDFs run in <80ms,
   * then refines exact sub-pixel coordinates at full image resolution.
   */
  function findRepeatingTemplatePeaks(hpGray, w, h, seedBox, minNcc = 0.36) {
    const sx0 = Math.max(0, Math.floor(seedBox.x));
    const sy0 = Math.max(0, Math.floor(seedBox.y));
    const sw = Math.min(w - sx0, Math.floor(seedBox.width));
    const sh = Math.min(h - sy0, Math.floor(seedBox.height));
    if (sw < 16 || sh < 16) return [];

    // Downscale proxy if large (> 720px) for fast coarse peak search
    const proxyScale = Math.min(1.0, 680.0 / Math.max(w, h));
    const pw = Math.max(32, Math.round(w * proxyScale));
    const ph = Math.max(32, Math.round(h * proxyScale));
    const pHp = proxyScale < 0.98 ? resizeGrayBilinear(hpGray, w, h, pw, ph) : hpGray;

    const psx = Math.floor(sx0 * proxyScale);
    const psy = Math.floor(sy0 * proxyScale);
    const psw = Math.max(12, Math.round(sw * proxyScale));
    const psh = Math.max(12, Math.round(sh * proxyScale));

    const pActive = [];
    let meanP = 0;
    for (let y = 0; y < psh; y++) {
      const off = (psy + y) * pw + psx;
      for (let x = 0; x < psw; x++) {
        const val = pHp[off + x];
        if (Math.abs(val) >= 3.5) {
          pActive.push([x, y, val]);
          meanP += val;
        }
      }
    }
    if (pActive.length < 25) return [];
    meanP /= pActive.length;
    for (let i = 0; i < pActive.length; i++) pActive[i][2] -= meanP;

    const stepPt = Math.max(1, Math.floor(pActive.length / 220));
    const sparsePts = [];
    let sparseNorm = 0;
    for (let i = 0; i < pActive.length; i += stepPt) {
      sparsePts.push(pActive[i]);
      sparseNorm += pActive[i][2] * pActive[i][2];
    }
    sparseNorm = Math.sqrt(sparseNorm);
    if (sparseNorm < 1e-4) return [];

    const padX = Math.floor(psw * 0.32);
    const padY = Math.floor(psh * 0.32);
    const stride = 2;
    const coarseCands = [];

    for (let py = -padY; py <= ph - psh + padY; py += stride) {
      for (let px = -padX; px <= pw - psw + padX; px += stride) {
        let dot = 0, normI = 0, validCnt = 0;
        for (let k = 0; k < sparsePts.length; k++) {
          const [tx, ty, tv] = sparsePts[k];
          const gx = px + tx, gy = py + ty;
          if (gx < 0 || gx >= pw || gy < 0 || gy >= ph) continue;
          const iv = pHp[gy * pw + gx];
          dot += iv * tv;
          normI += iv * iv;
          validCnt++;
        }
        if (validCnt < sparsePts.length * 0.45 || normI < 1e-3) continue;
        const ncc = dot / (Math.sqrt(normI) * sparseNorm);
        if (ncc >= minNcc * 0.80) {
          coarseCands.push({ px: px / proxyScale, py: py / proxyScale, ncc });
        }
      }
    }

    // Build full-res refinement points preserving fine 1px strokes
    const fullPts = [];
    let fullMean = 0;
    const fullStep = Math.max(1, Math.floor(Math.sqrt((sw * sh) / 6000)));
    for (let y = 0; y < sh; y += fullStep) {
      const off = (sy0 + y) * w + sx0;
      for (let x = 0; x < sw; x += fullStep) {
        const val = hpGray[off + x];
        if (Math.abs(val) >= 4.0) {
          fullPts.push([x, y, val]);
          fullMean += val;
        }
      }
    }
    if (fullPts.length < 30) return [];
    fullMean /= fullPts.length;
    let normFull = 0;
    for (let i = 0; i < fullPts.length; i++) {
      fullPts[i][2] -= fullMean;
      normFull += fullPts[i][2] * fullPts[i][2];
    }
    normFull = Math.sqrt(normFull);
    if (normFull < 1e-4) return [];

    coarseCands.sort((a, b) => b.ncc - a.ncc);
    const peaks = [];
    const nmsDistSq = Math.pow(Math.max(18, Math.min(sw, sh) * 0.55), 2);
    const searchR = Math.max(3, Math.ceil(2.5 / proxyScale));

    // Subsample refinement points only during initial integer sweep if very large
    const fastRefStep = Math.max(1, Math.floor(fullPts.length / 450));

    const evalNccBilinear = (rx, ry, minFrac = 0.45) => {
      let dot = 0, normI = 0, validCnt = 0;
      for (let k = 0; k < fullPts.length; k++) {
        const [tx, ty, tv] = fullPts[k];
        const gx = rx + tx, gy = ry + ty;
        if (gx < 0 || gx > w - 1 || gy < 0 || gy > h - 1) continue;
        const iv = sampleBilinearFloat(hpGray, w, h, gx, gy);
        dot += iv * tv;
        normI += iv * iv;
        validCnt++;
      }
      if (validCnt < fullPts.length * minFrac || normI < 1e-3) return -1;
      return dot / (Math.sqrt(normI) * normFull);
    };

    for (const cand of coarseCands) {
      let tooClose = false;
      for (const pk of peaks) {
        const dx = cand.px - pk.px, dy = cand.py - pk.py;
        if (dx * dx + dy * dy < nmsDistSq) { tooClose = true; break; }
      }
      if (tooClose) continue;

      // 1px integer scan around coarse location
      let bestX = Math.round(cand.px), bestY = Math.round(cand.py), bestNcc = -1;
      for (let ry = bestY - searchR; ry <= bestY + searchR; ry++) {
        for (let rx = bestX - searchR; rx <= bestX + searchR; rx++) {
          let dot = 0, normI = 0, validCnt = 0;
          for (let k = 0; k < fullPts.length; k++) {
            const [tx, ty, tv] = fullPts[k];
            const gx = rx + tx, gy = ry + ty;
            if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue;
            const iv = hpGray[gy * w + gx];
            dot += iv * tv;
            normI += iv * iv;
            validCnt++;
          }
          if (validCnt < fullPts.length * 0.45 || normI < 1e-3) continue;
          const ncc = dot / (Math.sqrt(normI) * normFull);
          if (ncc > bestNcc) { bestNcc = ncc; bestX = rx; bestY = ry; }
        }
      }

      if (bestNcc < minNcc) continue;

      // 0.25px sub-pixel refinement around best integer peak
      let subX = bestX, subY = bestY, subNcc = bestNcc;
      for (let dy = -0.5; dy <= 0.501; dy += 0.25) {
        for (let dx = -0.5; dx <= 0.501; dx += 0.25) {
          if (dx === 0 && dy === 0) continue;
          const ncc = evalNccBilinear(bestX + dx, bestY + dy);
          if (ncc > subNcc) {
            subNcc = ncc;
            subX = bestX + dx;
            subY = bestY + dy;
          }
        }
      }

      // Re-check NMS after sub-pixel refinement
      let dup = false;
      for (const pk of peaks) {
        const dx = subX - pk.px, dy = subY - pk.py;
        if (dx * dx + dy * dy < nmsDistSq) { dup = true; break; }
      }
      if (!dup) {
        peaks.push({ px: subX, py: subY, ncc: subNcc, sw, sh });
      }
      if (peaks.length >= 48) break;
    }

    // Complete missing regular 2D lattice slots ONLY when initial NCC peaks strongly support a rigid 2D lattice
    const strongInitCount = peaks.filter((pk) => pk.ncc >= 0.57).length;
    let verifiedLattice = false;
    if (peaks.length >= 4 && strongInitCount >= 4) {
      const diffs = [];
      for (let i = 0; i < peaks.length; i++) {
        for (let j = i + 1; j < peaks.length; j++) {
          let dx = peaks[j].px - peaks[i].px;
          let dy = peaks[j].py - peaks[i].py;
          if (dx < 0 || (dx === 0 && dy < 0)) { dx = -dx; dy = -dy; }
          const len = Math.hypot(dx, dy);
          if (len >= Math.max(sw, sh) * 0.42) diffs.push({ dx, dy, len });
        }
      }
      diffs.sort((a, b) => a.len - b.len);

      const basisCands = [];
      for (const d of diffs) {
        if (!basisCands.some((b) => Math.hypot(b.dx - d.dx, b.dy - d.dy) < 6.0)) {
          basisCands.push(d);
          if (basisCands.length >= 12) break;
        }
      }

      let bestLattice = null;
      let bestScore = -1;
      for (let a = 0; a < basisCands.length; a++) {
        for (let b = a + 1; b < basisCands.length; b++) {
          const v1 = basisCands[a], v2 = basisCands[b];
          const det = v1.dx * v2.dy - v1.dy * v2.dx;
          if (Math.abs(det) < 0.25 * v1.len * v2.len) continue;
          const p0 = peaks[0];
          let inliers = 0, errSum = 0;
          for (const pk of peaks) {
            const rx = pk.px - p0.px, ry = pk.py - p0.py;
            const u = (rx * v2.dy - ry * v2.dx) / det;
            const v = (v1.dx * ry - v1.dy * rx) / det;
            const mu = Math.round(u), nv = Math.round(v);
            const ex = p0.px + mu * v1.dx + nv * v2.dx - pk.px;
            const ey = p0.py + mu * v1.dy + nv * v2.dy - pk.py;
            const dist = Math.hypot(ex, ey);
            if (dist < 4.5) {
              inliers++;
              errSum += dist;
            }
          }
          const score = inliers * 10.0 - errSum - (v1.len + v2.len) * 0.005;
          if (inliers >= 4 && score > bestScore) {
            bestScore = score;
            bestLattice = { v1, v2, p0, det };
          }
        }
      }

      if (bestLattice) {
        // Require true 2D watermark spread (rejects 1D highway lane stripes or localized textures)
        let minPx = 1e9, maxPx = -1e9, minPy = 1e9, maxPy = -1e9;
        for (const pk of peaks) {
          if (pk.px < minPx) minPx = pk.px;
          if (pk.px > maxPx) maxPx = pk.px;
          if (pk.py < minPy) minPy = pk.py;
          if (pk.py > maxPy) maxPy = pk.py;
        }
        if ((maxPx - minPx) >= w * 0.22 && (maxPy - minPy) >= h * 0.22) {
          verifiedLattice = true;
          const { v1, v2, p0, det } = bestLattice;
          const corners = [[-sw, -sh], [w + sw, -sh], [-sw, h + sh], [w + sw, h + sh]];
          let mMin = 1e9, mMax = -1e9, nMin = 1e9, nMax = -1e9;
          for (const [cx, cy] of corners) {
            const rx = cx - p0.px, ry = cy - p0.py;
            const u = (rx * v2.dy - ry * v2.dx) / det;
            const v = (v1.dx * ry - v1.dy * rx) / det;
            if (u < mMin) mMin = u; if (u > mMax) mMax = u;
            if (v < nMin) nMin = v; if (v > nMax) nMax = v;
          }
          for (let m = Math.floor(mMin); m <= Math.ceil(mMax); m++) {
            for (let n = Math.floor(nMin); n <= Math.ceil(nMax); n++) {
              const gx = p0.px + m * v1.dx + n * v2.dx;
              const gy = p0.py + m * v1.dy + n * v2.dy;
              if (gx < -sw * 0.85 || gx > w - sw * 0.15 || gy < -sh * 0.85 || gy > h - sh * 0.15) continue;
              const already = peaks.some((pk) => Math.hypot(pk.px - gx, pk.py - gy) < Math.max(12, Math.min(sw, sh) * 0.35));
              if (already) continue;
              const ncc = evalNccBilinear(gx, gy, 0.15);
              if (ncc >= 0.16) {
                peaks.push({ px: gx, py: gy, ncc, sw, sh });
                if (peaks.length >= 48) break;
              }
            }
            if (peaks.length >= 48) break;
          }
        }
      }
    }

    peaks._verifiedLattice = verifiedLattice;
    return peaks;
  }

  /**
   * Resolution-adaptive Top-Hat + Black-Hat + PCA + NCC detector.
   */
  function detectTranslucentInRegion(gray, imgW, imgH, rx, ry, rw, rh, sensitivity, dilatePx, allowNonRepeating = false) {
    const subFull = new Float32Array(rw * rh);
    for (let y = 0; y < rh; y++) {
      const srcOff = (ry + y) * imgW + rx;
      const dstOff = y * rw;
      for (let x = 0; x < rw; x++) {
        subFull[dstOff + x] = gray[srcOff + x];
      }
    }

    // Run morphological stroke detection at <= 720px so stroke radii match fonts across all resolutions
    const detScale = Math.min(1.0, 720.0 / Math.max(rw, rh));
    const dw = Math.max(24, Math.round(rw * detScale));
    const dh = Math.max(24, Math.round(rh * detScale));
    const sub = detScale < 0.98 ? resizeGrayBilinear(subFull, rw, rh, dw, dh) : subFull;

    const open2 = morphOpenGray(sub, dw, dh, 2);
    const open4 = morphOpenGray(sub, dw, dh, 4);
    const close2 = morphCloseGray(sub, dw, dh, 2);
    const close4 = morphCloseGray(sub, dw, dh, 4);

    const thresh = Math.max(8, 17 - sensitivity * 12);
    const bin = new Uint8Array(dw * dh);
    for (let i = 0; i < bin.length; i++) {
      const th = Math.max(sub[i] - open2[i], sub[i] - open4[i]);
      const bh = Math.max(close2[i] - sub[i], close4[i] - sub[i]);
      if (Math.max(th, bh) >= thresh) bin[i] = 1;
    }

    const connected = dilateBinary(bin, dw, dh, 7, 5);
    const { labels, comps } = connectedComponents(connected, dw, dh);
    const padScaled = Math.max(3, Math.min(8, dilatePx)) * detScale;

    const boxes = [];
    for (const c of comps) {
      if (c.area < 45 || c.width < 14 || c.height < 10) continue;
      // Reject giant landscape blobs while still allowing 45-degree diagonal phrases
      if (c.width > dw * 0.55 || c.height > dh * 0.45) continue;

      const rawPts = [];
      let bgHigh = 0, darkLow = 0, samples = 0;
      for (let yy = c.y; yy < c.y + c.height; yy++) {
        const off = yy * dw;
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

      const orient = computeOrientedPolygonFromPoints(rawPts, padScaled, dw, dh);
      if (!orient) continue;
      // True oriented area check (rejects bushes/trees/road bands while keeping diagonal text strings!)
      if (orient.lengthU > dw * 0.50 || orient.heightV > dh * 0.22) continue;
      if (orient.lengthU * orient.heightV > dw * dh * 0.09) continue;

      const aspect = Math.max(orient.lengthU, 1) / Math.max(orient.heightV, 1);
      const fillRatio = rawPts.length / Math.max(1, orient.lengthU * orient.heightV);
      if (aspect < 1.2 && orient.lengthU > dw * 0.25) continue;
      if (fillRatio > 0.65 && orient.lengthU > dw * 0.25) continue;

      let bgVar = 0, meanSub = 0;
      for (let k = 0; k < rawPts.length; k++) {
        const [px, py] = rawPts[k];
        meanSub += open4[py * dw + px];
      }
      meanSub /= rawPts.length;
      for (let k = 0; k < rawPts.length; k++) {
        const [px, py] = rawPts[k];
        const d = open4[py * dw + px] - meanSub;
        bgVar += d * d;
      }
      bgVar = Math.sqrt(bgVar / rawPts.length);
      if (bgVar > 24 && orient.angleDeg <= 8) continue;

      const bx = Math.max(0, Math.round(rx + (c.x - padScaled) / detScale));
      const by = Math.max(0, Math.round(ry + (c.y - padScaled) / detScale));
      const bw = Math.min(imgW - bx, Math.round((c.width + padScaled * 2) / detScale));
      const bh = Math.min(imgH - by, Math.round((c.height + padScaled * 2) / detScale));

      const box = {
        x: bx,
        y: by,
        width: bw,
        height: bh,
        type: 'translucent_grid',
        label: orient.angleDeg > 8 ? 'Diagonal Watermark' : 'Watermark Overlay',
        _seedScore: (orient.angleDeg > 8 ? 2.0 : 1.0) * Math.min(3.5, aspect) * Math.sqrt(rawPts.length) / (1.0 + bgVar * 0.28)
      };
      if (orient.angleDeg > 8) {
        box.polygon = orient.polygon.map(([px, py]) => [
          Math.max(0, Math.min(imgW - 1, Math.round(rx + px / detScale))),
          Math.max(0, Math.min(imgH - 1, Math.round(ry + py / detScale)))
        ]);
        box.angle = orient.angleDeg;
      }
      boxes.push(box);
    }

    const seedCands = boxes.filter((b) => Math.max(b.width, b.height) * detScale >= 64);
    if (seedCands.length >= 1) {
      seedCands.sort((a, b) => (b._seedScore || 0) - (a._seedScore || 0));
      const bestSeed = seedCands[0];
      const hpR = Math.max(6, Math.round(6 / detScale));
      const blurR = fastBoxBlurFloat(subFull, rw, rh, hpR);
      const hpSub = new Float32Array(rw * rh);
      for (let i = 0; i < hpSub.length; i++) hpSub[i] = subFull[i] - blurR[i];

      const localSeed = {
        x: bestSeed.x - rx,
        y: bestSeed.y - ry,
        width: bestSeed.width,
        height: bestSeed.height
      };
      const peaks = findRepeatingTemplatePeaks(hpSub, rw, rh, localSeed, 0.45);
      const strongCount = peaks.filter((p) => p.ncc >= 0.57).length;
      if (peaks._verifiedLattice && peaks.length >= 4 && strongCount >= 4) {
        const expandedBoxes = [];
        for (const pk of peaks) {
          const pxRound = Math.round(pk.px);
          const pyRound = Math.round(pk.py);
          const bx = Math.max(0, rx + pxRound);
          const by = Math.max(0, ry + pyRound);
          const bw = Math.min(imgW - bx, pk.sw - Math.max(0, -pxRound));
          const bh = Math.min(imgH - by, pk.sh - Math.max(0, -pyRound));
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

    // In 'all' auto-detect, only keep translucent overlays when a true repeating pattern exists
    // (prevents false detections on palm trees, roads, or real signs); explicit translucent_grid preset keeps top compact candidates.
    if (!allowNonRepeating) return [];
    boxes.sort((a, b) => (b._seedScore || 0) - (a._seedScore || 0));
    const cleanNonRep = boxes.filter((b) => Math.max(b.width, b.height) >= 40 / detScale && b.width <= imgW * 0.35 && b.height <= imgH * 0.20).slice(0, 6);
    for (const b of cleanNonRep) delete b._seedScore;
    return cleanNonRep;
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
        boxes.push(...detectContainerBadgesInRegion(gray, w, r.x, r.y, r.width, r.height));
      }
    }
    let gridCount = 0;
    if (targetType === 'all' || targetType === 'text' || targetType === 'translucent_grid') {
      const allowNonRep = targetType === 'translucent_grid';
      for (const r of scanRegions) {
        const gridBoxes = detectTranslucentInRegion(gray, w, h, r.x, r.y, r.width, r.height, sensitivity, dilatePx, allowNonRep);
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
   * Multi-scale Push-Pull Harmonic Pyramid Fill strictly from unmasked pixels.
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

  /**
   * Multi-Scale Harmonic Pyramid + Border-Guided Exemplar Texture Synthesis + Sensor Grain.
   * Prevents blurry patches when filling larger opaque logos, corner sparkles, or boxes.
   */
  function clientContentAwareInpaint(srcImgData, binMask, w, h, pdePasses = 12, useExemplar = true) {
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

    // Measure local ring sensor noise & check if border has sharp natural textures worth exemplar copying
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
    const ringStd = ringCnt > 20 ? Math.sqrt(sqSum / ringCnt) : 0;
    const noiseStd = ringCnt > 20 ? Math.min(3.0, ringStd * 0.35) : 0.8;

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
   * Sub-pixel Dekel Multi-Instance YCbCr Matte Recovery (mirrors backend remover.py):
   * 1. Finds repeating template peaks with 0.25px precision.
   * 2. Recovers true uncontaminated local backgrounds per copy via harmonicMaskedFill on coarse stroke mask.
   * 3. Computes signed alpha_Y in [-0.045, 0.50] (canceling JPEG dark ringing rims!) AND
   *    chroma alpha_C in [0, 0.45] across aligned bilinear patches.
   * 4. Splat-projects sub-pixel shifted mattes across peaks with single-pass max (no double subtraction).
   * 5. Applies inverse YCbCr deblending + smooth-sky healing (bgStd < 5.0) + textured spike cleanup.
   */
  function tryDekelMultiInstanceClient(srcImgData, binMask, w, h, colorHint, boxesHint = null) {
    const src = srcImgData.data;
    const ySrc = extractGray(src, w, h);
    const blurR = Math.max(5, Math.round(Math.min(w, h) * 0.018));
    const blurMap = fastBoxBlurFloat(ySrc, w, h, blurR);
    const hpY = new Float32Array(w * h);
    for (let i = 0; i < hpY.length; i++) hpY[i] = ySrc[i] - blurMap[i];

    // Prefer individual un-merged detected watermark boxes when picking the template seed
    let seedCands = [];
    if (Array.isArray(boxesHint) && boxesHint.length >= 3) {
      seedCands = boxesHint.filter((b) => b.type === 'translucent_grid' && b.width >= 16 && b.height >= 16);
    }
    if (seedCands.length < 3) {
      const { comps } = connectedComponents(binMask, w, h);
      seedCands = comps.filter((c) => c.width >= 16 && c.height >= 16 && c.area >= 80);
    }
    if (seedCands.length < 3) return null;

    let bestComp = null;
    let bestScore = -1;
    for (const c of seedCands) {
      let hpEnergy = 0, bgVar = 0, meanBg = 0, cnt = 0;
      const x1 = Math.min(w, c.x + c.width), y1 = Math.min(h, c.y + c.height);
      for (let yy = Math.max(0, c.y); yy < y1; yy++) {
        const off = yy * w;
        for (let xx = Math.max(0, c.x); xx < x1; xx++) {
          const idx = off + xx;
          if (!binMask[idx]) continue;
          hpEnergy += Math.abs(hpY[idx]);
          meanBg += blurMap[idx];
          cnt++;
        }
      }
      if (cnt < 40) continue;
      meanBg /= cnt;
      for (let yy = Math.max(0, c.y); yy < y1; yy++) {
        const off = yy * w;
        for (let xx = Math.max(0, c.x); xx < x1; xx++) {
          const idx = off + xx;
          if (!binMask[idx]) continue;
          const d = blurMap[idx] - meanBg;
          bgVar += d * d;
        }
      }
      const stdBg = Math.sqrt(bgVar / cnt);
      const score = (hpEnergy / cnt) * Math.sqrt(cnt) / (1.0 + stdBg * 0.35);
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

    let sumSigned = 0, sCnt = 0;
    for (const pk of peaks) {
      const ix = Math.round(pk.px), iy = Math.round(pk.py);
      for (let y = 0; y < ph; y++) {
        const gy = iy + y;
        if (gy < 0 || gy >= h) continue;
        for (let x = 0; x < pw; x++) {
          const gx = ix + x;
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

    // Learn clean matte ONLY from high-confidence peaks (cluttered bg peaks get cleaned later via lattice projection)
    const mattePeaks = peaks.filter((pk) => pk.ncc >= 0.36);
    const learnPeaks = mattePeaks.length >= 3 ? mattePeaks : peaks;

    // Step 1: Align bilinear high-pass patches at exact sub-pixel peak coordinates
    const medHp = new Float32Array(patchLen);
    const hpSamples = new Float32Array(learnPeaks.length);
    for (let i = 0; i < patchLen; i++) {
      const lx = i % pw;
      const ly = (i - lx) / pw;
      let k = 0;
      for (let pIdx = 0; pIdx < learnPeaks.length; pIdx++) {
        const gx = learnPeaks[pIdx].px + lx;
        const gy = learnPeaks[pIdx].py + ly;
        if (gx < 0 || gx > w - 1 || gy < 0 || gy > h - 1) continue;
        const val = sampleBilinearFloat(hpY, w, h, gx, gy);
        hpSamples[k++] = isLight ? val : -val;
      }
      if (k >= 2) {
        const sub = hpSamples.subarray(0, k).slice().sort();
        medHp[i] = sub[Math.floor(k * 0.5)];
      }
    }

    // Build tile strip mask covering strokes + immediate JPEG ringing rim (dilated 2px)
    const rawStroke = new Uint8Array(patchLen);
    let rawStrokeCount = 0;
    for (let i = 0; i < patchLen; i++) {
      if (medHp[i] >= 2.8) {
        rawStroke[i] = 1;
        rawStrokeCount++;
      }
    }
    if (rawStrokeCount < 20) return null;
    const stripPatch = dilateBinary(rawStroke, pw, ph, 2);

    // Check overlap with user mask
    let overlapPeaks = 0;
    for (const pk of peaks) {
      const ix = Math.round(pk.px), iy = Math.round(pk.py);
      let hit = 0;
      for (let y = 0; y < ph; y++) {
        const gy = iy + y;
        if (gy < 0 || gy >= h) continue;
        for (let x = 0; x < pw; x++) {
          const gx = ix + x;
          if (gx < 0 || gx >= w) continue;
          if (binMask[gy * w + gx] && stripPatch[y * pw + x]) { hit++; break; }
        }
        if (hit) break;
      }
      if (hit) overlapPeaks++;
    }

    // Step 2: Project global strip mask and recover uncontaminated local background strictly from outside strips
    const globalStripMask = new Uint8Array(w * h);
    for (const pk of peaks) {
      const ix = Math.round(pk.px), iy = Math.round(pk.py);
      for (let y = 0; y < ph; y++) {
        const gy = iy + y;
        if (gy < 0 || gy >= h) continue;
        for (let x = 0; x < pw; x++) {
          const gx = ix + x;
          if (gx < 0 || gx >= w) continue;
          const gIdx = gy * w + gx;
          if (overlapPeaks <= 1 && !binMask[gIdx]) continue;
          if (stripPatch[y * pw + x]) globalStripMask[gIdx] = 1;
        }
      }
    }

    const cleanBg = harmonicMaskedFill(src, globalStripMask, w, h);

    // Step 3: Compute signed Y alpha [-0.045, 0.50] AND chroma alpha_C [0, 0.45] across clean peaks only
    const alphaY = new Float32Array(patchLen);
    const alphaC = new Float32Array(patchLen);
    const ySamples = new Float32Array(learnPeaks.length);
    const cSamples = new Float32Array(learnPeaks.length * 2);

    for (let i = 0; i < patchLen; i++) {
      if (!stripPatch[i]) continue;
      const lx = i % pw;
      const ly = (i - lx) / pw;
      let ky = 0, kc = 0;
      for (let pIdx = 0; pIdx < learnPeaks.length; pIdx++) {
        const gx = learnPeaks[pIdx].px + lx;
        const gy = learnPeaks[pIdx].py + ly;
        if (gx < 0 || gx > w - 1 || gy < 0 || gy > h - 1) continue;

        const [sr, sg, sb] = sampleBilinearRGB(src, w, h, gx, gy);
        const br = sampleBilinearFloat(cleanBg.rArr, w, h, gx, gy);
        const bg = sampleBilinearFloat(cleanBg.gArr, w, h, gx, gy);
        const bb = sampleBilinearFloat(cleanBg.bArr, w, h, gx, gy);

        const yT = 0.299 * sr + 0.587 * sg + 0.114 * sb;
        const yB = 0.299 * br + 0.587 * bg + 0.114 * bb;
        const dy = isLight ? (yT - yB) : (yB - yT);
        const denomY = isLight ? Math.max(35.0, wmTargetY - yB) : Math.max(35.0, yB - wmTargetY);
        ySamples[ky++] = Math.max(-0.05, Math.min(0.50, dy / denomY));

        const cbT = -0.168736 * sr - 0.331264 * sg + 0.5 * sb + 128.0;
        const cbB = -0.168736 * br - 0.331264 * bg + 0.5 * bb + 128.0;
        const crT = 0.5 * sr - 0.418688 * sg - 0.081312 * sb + 128.0;
        const crB = 0.5 * br - 0.418688 * bg - 0.081312 * bb + 128.0;
        if (Math.abs(128.0 - cbB) > 10.0) {
          cSamples[kc++] = Math.max(-0.10, Math.min(0.45, (cbT - cbB) / (128.0 - cbB)));
        }
        if (Math.abs(128.0 - crB) > 10.0) {
          cSamples[kc++] = Math.max(-0.10, Math.min(0.45, (crT - crB) / (128.0 - crB)));
        }
      }
      if (ky >= 2) {
        const subY = ySamples.subarray(0, ky).slice().sort();
        const rawMed = subY[Math.floor(ky * 0.5)];
        const ay = rawMed > 0 ? Math.min(0.49, rawMed * 1.05) : Math.max(-0.045, rawMed);
        alphaY[i] = Math.abs(ay) >= 0.009 ? ay : 0.0;
      }
      if (kc >= 2) {
        const subC = cSamples.subarray(0, kc).slice().sort();
        const ac = Math.max(0.0, Math.min(0.46, subC[Math.floor(kc * 0.5)] * 1.05));
        alphaC[i] = ac >= 0.010 ? ac : 0.0;
      }
    }

    // Step 4: Project sub-pixel shifted mattes globally using single-pass max (and negative ringing fallback)
    const globalAY = new Float32Array(w * h);
    const globalAC = new Float32Array(w * h);
    for (const pk of peaks) {
      const ix = Math.floor(pk.px), iy = Math.floor(pk.py);
      const fx = pk.px - ix, fy = pk.py - iy;
      for (let y = 0; y <= ph; y++) {
        const gy = iy + y;
        if (gy < 0 || gy >= h) continue;
        for (let x = 0; x <= pw; x++) {
          const gx = ix + x;
          if (gx < 0 || gx >= w) continue;
          const gIdx = gy * w + gx;
          if (overlapPeaks <= 1 && !binMask[gIdx]) continue;

          // Backward-map fractional offset (x - fx, y - fy) inside patch
          const sx = x - fx, sy = y - fy;
          if (sx < 0 || sx > pw - 1 || sy < 0 || sy > ph - 1) continue;
          const ay = sampleBilinearFloat(alphaY, pw, ph, sx, sy);
          const ac = sampleBilinearFloat(alphaC, pw, ph, sx, sy);
          if (ay > globalAY[gIdx]) globalAY[gIdx] = ay;
          else if (ay < 0 && globalAY[gIdx] === 0) globalAY[gIdx] = ay;
          if (ac > globalAC[gIdx]) globalAC[gIdx] = ac;
        }
      }
    }

    // Step 5: Exact YCbCr inverse alpha recovery (mirrors remover.py lines 575-599)
    const out = new Uint8ClampedArray(src);
    for (let i = 0; i < w * h; i++) {
      const ay = globalAY[i];
      const ac = globalAC[i];
      if (ay === 0 && ac === 0) continue;

      const p = i * 4;
      const r = src[p], g = src[p + 1], b = src[p + 2];
      const yVal = 0.299 * r + 0.587 * g + 0.114 * b;
      const cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
      const cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

      const yRec = (yVal - ay * wmTargetY) / Math.max(0.45, 1.0 - ay);
      const cbRec = (cb - ac * 128.0) / Math.max(0.45, 1.0 - ac);
      const crRec = (cr - ac * 128.0) / Math.max(0.45, 1.0 - ac);

      out[p] = clamp255(yRec + 1.402 * (crRec - 128.0));
      out[p + 1] = clamp255(yRec - 0.344136 * (cbRec - 128.0) - 0.714136 * (crRec - 128.0));
      out[p + 2] = clamp255(yRec + 1.772 * (cbRec - 128.0));
    }

    // Compute 3x3 median & local background texture variance on RECOVERED image (just like remover.py!)
    const med3 = new Uint8ClampedArray(out);
    const outGray = new Float32Array(w * h);
    const tmp9 = new Float32Array(9);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const p = i * 4;
        if (x > 0 && x < w - 1 && y > 0 && y < h - 1 && globalAY[i] !== 0) {
          for (let ch = 0; ch < 3; ch++) {
            let k = 0;
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                tmp9[k++] = out[((y + dy) * w + (x + dx)) * 4 + ch];
              }
            }
            tmp9.sort();
            med3[p + ch] = tmp9[4];
          }
        }
        outGray[i] = rgbToY(med3[p], med3[p + 1], med3[p + 2]);
      }
    }

    // Compute high-pass texture std on clean background (heals smooth sky gradients cleanly while keeping waves/hair sharp!)
    const cleanGray = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      cleanGray[i] = rgbToY(cleanBg.rArr[i], cleanBg.gArr[i], cleanBg.bArr[i]);
    }
    const cleanBlur2 = fastBoxBlurFloat(cleanGray, w, h, 2);
    const lapSq = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      const hp = cleanGray[i] - cleanBlur2[i];
      lapSq[i] = hp * hp;
    }
    const lapVar = fastBoxBlurFloat(lapSq, w, h, 3);

    const smoothHole = new Uint8Array(w * h);
    let hasSmooth = false;
    for (let i = 0; i < w * h; i++) {
      const ay = globalAY[i];
      if (ay === 0) continue;
      const hpStd = Math.sqrt(Math.max(0, lapVar[i]));
      if (hpStd < 2.2 && Math.abs(ay) >= 0.014) {
        smoothHole[i] = 1;
        hasSmooth = true;
      } else if (ay > 0.14 && hpStd >= 2.2) {
        const p = i * 4;
        const d = Math.max(
          Math.abs(out[p] - med3[p]),
          Math.abs(out[p + 1] - med3[p + 1]),
          Math.abs(out[p + 2] - med3[p + 2])
        );
        if (d > 7.5) {
          out[p] = med3[p];
          out[p + 1] = med3[p + 1];
          out[p + 2] = med3[p + 2];
        }
      }
    }

    let resultImg = new ImageData(out, w, h);
    if (hasSmooth) {
      resultImg = clientContentAwareInpaint(resultImg, smoothHole, w, h, 6, false);
    }
    return { image: resultImg, cleanedMask: globalStripMask };
  }

  /**
   * Hybrid Removal inside Browser JS:
   * 1. Runs sub-pixel Dekel Multi-Instance Matte Recovery across any repeating copies.
   * 2. Processes any remaining non-repeating masked areas (e.g., letterboxes, badges,
   *    AI corner sparkles, or horizontal overlays) cleanly without throwing them away!
   */
  function clientStrokeDeblend(srcImgData, binMask, w, h, colorHint, alphaHint = 0.5, boxesHint = null) {
    let srcImg = srcImgData;
    let activeMask = binMask;
    const dekel = tryDekelMultiInstanceClient(srcImgData, binMask, w, h, colorHint, boxesHint);
    if (dekel) {
      srcImg = dekel.image;
      const { labels, comps } = connectedComponents(binMask, w, h);
      const covered = new Set();
      for (const c of comps) {
        let hits = 0;
        for (let y = c.y; y < c.y + c.height; y++) {
          const off = y * w;
          for (let x = c.x; x < c.x + c.width; x++) {
            if (labels[off + x] === c.id && dekel.cleanedMask[off + x]) hits++;
          }
        }
        if (hits >= 15) covered.add(c.id);
      }
      const remainingMask = new Uint8Array(w * h);
      let remCount = 0;
      for (let i = 0; i < remainingMask.length; i++) {
        if (labels[i] && !covered.has(labels[i])) {
          remainingMask[i] = 1;
          remCount++;
        }
      }
      if (remCount < 15) return srcImg;
      activeMask = remainingMask;
    }

    const src = srcImg.data;
    const out = new Uint8ClampedArray(src);
    const len = w * h;
    const ySrc = extractGray(src, w, h);
    const open3 = morphOpenGray(ySrc, w, h, 3);
    const close3 = morphCloseGray(ySrc, w, h, 3);

    let isLight = true;
    let wmTargetR = 255.0, wmTargetG = 255.0, wmTargetB = 255.0;
    if (colorHint && colorHint !== 'auto' && colorHint.startsWith('#') && colorHint.length === 7) {
      wmTargetR = parseInt(colorHint.slice(1, 3), 16);
      wmTargetG = parseInt(colorHint.slice(3, 5), 16);
      wmTargetB = parseInt(colorHint.slice(5, 7), 16);
      isLight = rgbToY(wmTargetR, wmTargetG, wmTargetB) > 128.0;
    } else {
      let sumTH = 0, sumBH = 0;
      for (let i = 0; i < len; i++) {
        if (!activeMask[i]) continue;
        sumTH += Math.max(0, ySrc[i] - open3[i]);
        sumBH += Math.max(0, close3[i] - ySrc[i]);
      }
      isLight = sumTH >= sumBH;
      wmTargetR = wmTargetG = wmTargetB = isLight ? 255.0 : 0.0;
    }
    const wmTargetY = rgbToY(wmTargetR, wmTargetG, wmTargetB);

    const strokeMask = new Uint8Array(len);
    let strokeCnt = 0, activeCnt = 0;
    for (let i = 0; i < len; i++) {
      if (!activeMask[i]) continue;
      activeCnt++;
      const dev = isLight ? (ySrc[i] - open3[i]) : (close3[i] - ySrc[i]);
      if (dev >= 4.8) {
        strokeMask[i] = 1;
        strokeCnt++;
      }
    }
    if (strokeCnt < 8) {
      return clientContentAwareInpaint(srcImg, activeMask, w, h, 14, false);
    }

    const softStroke = dilateBinary(strokeMask, w, h, 1);
    for (let i = 0; i < len; i++) {
      if (!activeMask[i]) softStroke[i] = 0;
    }

    const cleanBg = harmonicMaskedFill(src, softStroke, w, h);
    const bgY = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      bgY[i] = rgbToY(cleanBg.rArr[i], cleanBg.gArr[i], cleanBg.bArr[i]);
    }
    const bgBlur2 = fastBoxBlurFloat(bgY, w, h, 2);
    const lapSq = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      const d = bgY[i] - bgBlur2[i];
      lapSq[i] = d * d;
    }
    const lapVar = fastBoxBlurFloat(lapSq, w, h, 3);

    const smoothHole = new Uint8Array(len);
    let hasSmooth = false;
    for (let i = 0; i < len; i++) {
      if (!softStroke[i]) continue;
      const denom = isLight ? Math.max(30.0, wmTargetY - bgY[i]) : Math.max(30.0, bgY[i] - wmTargetY);
      const diff = isLight ? (ySrc[i] - bgY[i]) : (bgY[i] - ySrc[i]);
      const a = Math.max(-0.04, Math.min(Math.max(0.35, alphaHint * 1.15), diff / denom));
      if (Math.abs(a) < 0.025) continue;

      const p = i * 4;
      const r = src[p], g = src[p + 1], b = src[p + 2];
      const yVal = 0.299 * r + 0.587 * g + 0.114 * b;
      const cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
      const cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

      const inv = Math.max(0.45, 1.0 - a);
      const yRec = (yVal - a * wmTargetY) / inv;
      const cbRec = 128.0 + (cb - 128.0) / Math.max(0.65, 1.0 - a * 0.6);
      const crRec = 128.0 + (cr - 128.0) / Math.max(0.65, 1.0 - a * 0.6);

      out[p] = clamp255(yRec + 1.402 * (crRec - 128.0));
      out[p + 1] = clamp255(yRec - 0.344136 * (cbRec - 128.0) - 0.714136 * (crRec - 128.0));
      out[p + 2] = clamp255(yRec + 1.772 * (cbRec - 128.0));

      const hpStd = Math.sqrt(Math.max(0, lapVar[i]));
      if (a > 0.03 && hpStd < 3.2) {
        smoothHole[i] = 1;
        hasSmooth = true;
      }
    }

    if (hasSmooth) {
      return clientContentAwareInpaint(new ImageData(out, w, h), smoothHole, w, h, 6, false);
    }
    return new ImageData(out, w, h);
  }

  function processImageDataClient(srcImgData, maskImgData, mode = 'auto', colorHint = 'auto', alphaHint = 0.5, boxesHint = null) {
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

    // Hybrid routing: always clean solid opaque overlays (letterboxes, dark badges, AI corner sparkles)
    // via harmonic inpainting even when repeating translucent grids are present on the same image.
    let workingImg = srcImgData;
    let remainingBinMask = binMask;
    let remainingBoxes = boxesHint;
    if (Array.isArray(boxesHint) && boxesHint.length > 0 && (mode === 'auto' || mode === 'deblend')) {
      const solidTypes = new Set(['ai_sparkles', 'letterbox_bar', 'container_badge']);
      const solidBoxes = boxesHint.filter((b) => solidTypes.has(b.type));
      const transBoxes = boxesHint.filter((b) => !solidTypes.has(b.type));
      if (solidBoxes.length > 0 && transBoxes.length === 0 && mode === 'auto') {
        return clientContentAwareInpaint(srcImgData, binMask, w, h, 14, false);
      }
      if (solidBoxes.length > 0 && transBoxes.length > 0) {
        const solidMask = new Uint8Array(w * h);
        for (const sb of solidBoxes) {
          const x0 = Math.max(0, Math.floor(sb.x)), y0 = Math.max(0, Math.floor(sb.y));
          const x1 = Math.min(w, Math.ceil(sb.x + sb.width)), y1 = Math.min(h, Math.ceil(sb.y + sb.height));
          for (let yy = y0; yy < y1; yy++) {
            const off = yy * w;
            for (let xx = x0; xx < x1; xx++) {
              if (binMask[off + xx]) solidMask[off + xx] = 1;
            }
          }
        }
        workingImg = clientContentAwareInpaint(workingImg, solidMask, w, h, 14, false);
        remainingBinMask = new Uint8Array(w * h);
        for (let i = 0; i < remainingBinMask.length; i++) {
          if (binMask[i] && !solidMask[i]) remainingBinMask[i] = 1;
        }
        remainingBoxes = transBoxes;
      }
    }

    if (mode === 'deblend') {
      return clientStrokeDeblend(workingImg, remainingBinMask, w, h, colorHint, alphaHint, remainingBoxes);
    }
    if (mode === 'inpaint_telea') {
      return clientContentAwareInpaint(workingImg, remainingBinMask, w, h, 10, false);
    }
    if (mode === 'inpaint_ns') {
      return clientContentAwareInpaint(workingImg, remainingBinMask, w, h, 24, false);
    }

    const { comps } = connectedComponents(remainingBinMask, w, h);
    if (comps.length >= 3 || (Array.isArray(remainingBoxes) && remainingBoxes.filter((b) => b.type === 'translucent_grid').length >= 3)) {
      return clientStrokeDeblend(workingImg, remainingBinMask, w, h, colorHint, alphaHint, remainingBoxes);
    }
    return clientContentAwareInpaint(workingImg, remainingBinMask, w, h, 14, false);
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
