/**
 * Eraseon - Pure Browser Client-Side Engine
 * Enables 100% serverless execution on GitHub Pages (images & multi-page PDFs)
 * when the optional local FastAPI + Big-LaMa backend is not running.
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

  // Session store in browser memory
  const clientSessions = {};

  function rgbToY(r, g, b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
  }

  function fastBoxBlurGray(src, w, h, radius) {
    const tmp = new Float32Array(w * h);
    const dst = new Float32Array(w * h);
    const win = 2 * radius + 1;

    // Horizontal pass
    for (let y = 0; y < h; y++) {
      let sum = 0;
      const rowOff = y * w;
      for (let kx = -radius; kx <= radius; kx++) {
        const cx = Math.max(0, Math.min(w - 1, kx));
        sum += src[rowOff + cx];
      }
      for (let x = 0; x < w; x++) {
        tmp[rowOff + x] = sum / win;
        const xAdd = Math.min(w - 1, x + radius + 1);
        const xSub = Math.max(0, x - radius);
        sum += src[rowOff + xAdd] - src[rowOff + xSub];
      }
    }

    // Vertical pass
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let ky = -radius; ky <= radius; ky++) {
        const cy = Math.max(0, Math.min(h - 1, ky));
        sum += tmp[cy * w + x];
      }
      for (let y = 0; y < h; y++) {
        dst[y * w + x] = sum / win;
        const yAdd = Math.min(h - 1, y + radius + 1);
        const ySub = Math.max(0, y - radius);
        sum += tmp[yAdd * w + x] - tmp[ySub * w + x];
      }
    }
    return dst;
  }

  function extractGray(data, w, h) {
    const gray = new Float32Array(w * h);
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      gray[i] = rgbToY(data[p], data[p + 1], data[p + 2]);
    }
    return gray;
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

  function dilateBinary(src, w, h, r) {
    const tmp = new Uint8Array(w * h);
    const dst = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const off = y * w;
      for (let x = 0; x < w; x++) {
        let on = 0;
        const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
        for (let kx = x0; kx <= x1; kx++) {
          if (src[off + kx]) { on = 1; break; }
        }
        tmp[off + x] = on;
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let on = 0;
        const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
        for (let ky = y0; ky <= y1; ky++) {
          if (tmp[ky * w + x]) { on = 1; break; }
        }
        dst[y * w + x] = on;
      }
    }
    return dst;
  }

  function findEmbeddedPhotoRegions(gray, w, h) {
    // Check if outer page borders are bright white (PDF / document page)
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

  function detectCornerSparklesInRegion(imgData, gray, imgW, rx, ry, rw, rh, sensitivity) {
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

    const blur = fastBoxBlurGray(subGray, cw, ch, 14);
    const thThresh = Math.max(16, 28 - sensitivity * 12);
    const bin = new Uint8Array(cw * ch);
    for (let i = 0; i < bin.length; i++) {
      if (subGray[i] - blur[i] >= thThresh) bin[i] = 1;
    }

    const { comps } = connectedComponents(bin, cw, ch);
    for (const c of comps) {
      const aspect = c.width / Math.max(1, c.height);
      if (c.width >= 12 && c.width <= 55 && c.height >= 12 && c.height <= 55 && aspect >= 0.72 && aspect <= 1.38 && c.area >= 35 && c.area <= 1800) {
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
            x: Math.max(0, x1 + c.x - 4),
            y: Math.max(0, y1 + c.y - 4),
            width: c.width + 8,
            height: c.height + 8,
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

  function detectTranslucentGridClient(gray, w, h, sensitivity) {
    const boxes = [];
    const blur = fastBoxBlurGray(gray, w, h, 6);
    const thresh = Math.max(10, 20 - sensitivity * 10);
    const bin = new Uint8Array(w * h);

    // Ignore dark-ink white-paper body paragraphs by requiring smooth or midtone background
    for (let i = 0; i < bin.length; i++) {
      const diff = gray[i] - blur[i];
      if (diff >= thresh) bin[i] = 1;
    }

    const connected = dilateBinary(bin, w, h, 4);
    const { comps } = connectedComponents(connected, w, h);
    for (const c of comps) {
      const aspect = c.width / Math.max(1, c.height);
      if (c.area >= 90 && c.width >= 20 && c.width <= w * 0.42 && c.height >= 12 && c.height <= h * 0.35 && aspect >= 0.3 && aspect <= 5.0) {
        // Check that it's not bright white-paper text box edge
        let bgHigh = 0, darkLow = 0, samples = 0;
        for (let yy = c.y; yy < c.y + c.height; yy += 2) {
          for (let xx = c.x; xx < c.x + c.width; xx += 2) {
            const v = gray[yy * w + xx];
            if (v > 235) bgHigh++;
            if (v < 95) darkLow++;
            samples++;
          }
        }
        if (samples > 0 && (bgHigh / samples > 0.45 && darkLow / samples > 0.03)) {
          continue;
        }
        boxes.push({
          x: Math.max(0, c.x - 3),
          y: Math.max(0, c.y - 3),
          width: Math.min(w - Math.max(0, c.x - 3), c.width + 6),
          height: Math.min(h - Math.max(0, c.y - 3), c.height + 6),
          type: 'translucent_grid',
          label: 'Translucent Overlay'
        });
      }
    }
    return boxes;
  }

  function autoDetectClient(imageData, sensitivity = 0.5, targetType = 'all') {
    const { data, width: w, height: h } = imageData;
    const gray = extractGray(data, w, h);
    const embedded = findEmbeddedPhotoRegions(gray, w, h);
    const scanRegions = embedded.length > 0 ? embedded : [{ x: 0, y: 0, width: w, height: h }];

    const boxes = [];
    if (targetType === 'all' || targetType === 'ai_sparkles') {
      for (const r of scanRegions) {
        boxes.push(...detectCornerSparklesInRegion(data, gray, w, r.x, r.y, r.width, r.height, sensitivity));
      }
    }
    if (targetType === 'all' || targetType === 'letterbox') {
      for (const r of scanRegions) {
        boxes.push(...detectLetterboxInRegion(gray, w, r.x, r.y, r.width, r.height));
      }
    }
    let gridCount = 0;
    if (targetType === 'translucent_grid' || (targetType === 'all' && embedded.length === 0)) {
      const gridBoxes = detectTranslucentGridClient(gray, w, h, sensitivity);
      if (gridBoxes.length >= 4 || targetType === 'translucent_grid') {
        gridCount = gridBoxes.length;
        boxes.push(...gridBoxes);
      }
    }

    const recommendedMode = gridCount >= 4 ? 'deblend' : 'auto';
    return {
      detected_count: boxes.length,
      boxes,
      recommended_mode: recommendedMode
    };
  }

  /**
   * Client-side Fast Priority Inpainting (Telea-style distance-weighted gradient continuation)
   * for opaque logos, stars, badges, and letterbox bars.
   */
  function clientContentAwareInpaint(srcImgData, binMask, w, h) {
    const out = new Uint8ClampedArray(srcImgData.data);
    const filled = new Uint8Array(w * h);
    const dist = new Float32Array(w * h);

    for (let i = 0; i < filled.length; i++) {
      if (binMask[i]) {
        filled[i] = 0;
        dist[i] = 1e6;
      } else {
        filled[i] = 1;
        dist[i] = 0;
      }
    }

    // Build distance-ordered frontier via BFS rings from unmasked boundary
    const queue = [];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (filled[idx]) continue;
        let touchesKnown = false;
        for (let dy = -1; dy <= 1 && !touchesKnown; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            if (nx >= 0 && nx < w && filled[ny * w + nx]) {
              touchesKnown = true;
              break;
            }
          }
        }
        if (touchesKnown) {
          dist[idx] = 1;
          queue.push(idx);
        }
      }
    }

    // BFS ordering inward
    let head = 0;
    while (head < queue.length) {
      const cur = queue[head++];
      const cx = cur % w;
      const cy = (cur - cx) / w;
      const nextD = dist[cur] + 1;
      const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
      for (const [dx, dy] of dirs) {
        const nx = cx + dx, ny = cy + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
          const nIdx = ny * w + nx;
          if (!filled[nIdx] && dist[nIdx] > nextD) {
            dist[nIdx] = nextD;
            queue.push(nIdx);
          }
        }
      }
    }

    // Fill pixels in inward order using 5x5 inverse-distance weighted interpolation
    for (let q = 0; q < queue.length; q++) {
      const idx = queue[q];
      const cx = idx % w;
      const cy = (idx - cx) / w;
      let sumR = 0, sumG = 0, sumB = 0, sumW = 0;
      for (let dy = -3; dy <= 3; dy++) {
        const ny = cy + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -3; dx <= 3; dx++) {
          const nx = cx + dx;
          if (nx < 0 || nx >= w || (dx === 0 && dy === 0)) continue;
          const nIdx = ny * w + nx;
          if (!filled[nIdx]) continue;
          const weight = 1.0 / (dx * dx + dy * dy);
          const p = nIdx * 4;
          sumR += out[p] * weight;
          sumG += out[p + 1] * weight;
          sumB += out[p + 2] * weight;
          sumW += weight;
        }
      }
      if (sumW > 0) {
        const p = idx * 4;
        out[p] = Math.round(sumR / sumW);
        out[p + 1] = Math.round(sumG / sumW);
        out[p + 2] = Math.round(sumB / sumW);
        out[p + 3] = 255;
        filled[idx] = 1;
      }
    }

    // 2 smooth relaxation passes inside mask to eliminate any seam lines
    const tmp = new Uint8ClampedArray(out);
    for (let pass = 0; pass < 2; pass++) {
      for (let q = 0; q < queue.length; q++) {
        const idx = queue[q];
        const cx = idx % w;
        const cy = (idx - cx) / w;
        let sr = 0, sg = 0, sb = 0, cnt = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx;
            if (nx < 0 || nx >= w) continue;
            const p = (ny * w + nx) * 4;
            sr += out[p];
            sg += out[p + 1];
            sb += out[p + 2];
            cnt++;
          }
        }
        const p = idx * 4;
        tmp[p] = Math.round(sr / cnt);
        tmp[p + 1] = Math.round(sg / cnt);
        tmp[p + 2] = Math.round(sb / cnt);
      }
      for (let q = 0; q < queue.length; q++) {
        const p = queue[q] * 4;
        out[p] = tmp[p];
        out[p + 1] = tmp[p + 1];
        out[p + 2] = tmp[p + 2];
      }
    }

    return new ImageData(out, w, h);
  }

  /**
   * Client-side Zero-Blur YCrCb Stroke-Aware De-blending:
   * - Smooth regions (sky / calm water): thin-seam boundary interpolation
   * - Textured regions (hair / skin / wave crests): capped luminance subtraction
   */
  function clientStrokeDeblend(srcImgData, binMask, w, h, colorHint) {
    const src = srcImgData.data;
    const out = new Uint8ClampedArray(src);
    const yCh = extractGray(src, w, h);
    const yBlur = fastBoxBlurGray(yCh, w, h, 5);

    // Estimate local texture std from blurred background
    const yBlurSq = new Float32Array(w * h);
    for (let i = 0; i < yBlur.length; i++) yBlurSq[i] = yBlur[i] * yBlur[i];
    const meanB = fastBoxBlurGray(yBlur, w, h, 3);
    const meanB2 = fastBoxBlurGray(yBlurSq, w, h, 3);
    const bgStd = new Float32Array(w * h);
    for (let i = 0; i < bgStd.length; i++) {
      bgStd[i] = Math.sqrt(Math.max(0, meanB2[i] - meanB[i] * meanB[i]));
    }

    let isLight = true;
    if (colorHint && colorHint !== 'auto' && colorHint.startsWith('#') && colorHint.length === 7) {
      const rr = parseInt(colorHint.slice(1, 3), 16);
      const gg = parseInt(colorHint.slice(3, 5), 16);
      const bb = parseInt(colorHint.slice(5, 7), 16);
      isLight = (rr + gg + bb) > 384;
    } else {
      let sumD = 0, cnt = 0;
      for (let i = 0; i < binMask.length; i++) {
        if (binMask[i]) { sumD += yCh[i] - yBlur[i]; cnt++; }
      }
      isLight = cnt === 0 || (sumD / cnt) >= 0;
    }

    const strokeMask = new Uint8Array(w * h);
    let strokeCount = 0;
    for (let i = 0; i < binMask.length; i++) {
      if (!binMask[i]) continue;
      const dev = isLight ? (yCh[i] - yBlur[i]) : (yBlur[i] - yCh[i]);
      if (dev >= 4.5) {
        strokeMask[i] = 1;
        strokeCount++;
      }
    }

    if (strokeCount === 0) {
      return clientContentAwareInpaint(srcImgData, binMask, w, h);
    }

    // Capped luminance adjustment on textured pixels
    const smoothMask = new Uint8Array(w * h);
    const dilStroke = dilateBinary(strokeMask, w, h, 1);
    for (let i = 0; i < strokeMask.length; i++) {
      if (dilStroke[i] && bgStd[i] < 5.5) {
        smoothMask[i] = 1;
      } else if (strokeMask[i]) {
        const rawDev = isLight ? (yCh[i] - yBlur[i]) : (yBlur[i] - yCh[i]);
        const capped = Math.min(26.0, Math.max(0.0, rawDev));
        const delta = isLight ? -capped : capped;
        const p = i * 4;
        out[p] = Math.max(0, Math.min(255, Math.round(out[p] + delta)));
        out[p + 1] = Math.max(0, Math.min(255, Math.round(out[p + 1] + delta)));
        out[p + 2] = Math.max(0, Math.min(255, Math.round(out[p + 2] + delta)));
      }
    }

    // Thin-seam clean interpolation only on smooth sky/gradients
    let hasSmooth = false;
    for (let i = 0; i < smoothMask.length; i++) {
      if (smoothMask[i]) { hasSmooth = true; break; }
    }
    if (hasSmooth) {
      return clientContentAwareInpaint(new ImageData(out, w, h), smoothMask, w, h);
    }
    return new ImageData(out, w, h);
  }

  function processImageDataClient(srcImgData, maskImgData, mode = 'auto', colorHint = 'auto') {
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
      return clientStrokeDeblend(srcImgData, binMask, w, h, colorHint);
    }
    if (mode === 'auto') {
      const { comps } = connectedComponents(binMask, w, h);
      if (comps.length >= 5) {
        return clientStrokeDeblend(srcImgData, binMask, w, h, colorHint);
      }
    }
    return clientContentAwareInpaint(srcImgData, binMask, w, h);
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

  async function processAllPdfClient(sessionId, templateMaskDataUrl, pageMasksMap, mode, colorHint) {
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
        const cleaned = processImageDataClient(srcData, maskData, mode, colorHint);
        ctx.putImageData(cleaned, 0, 0);
      }

      const pngBytes = await new Promise((resolve) => c.toBlob((b) => b.arrayBuffer().then(resolve), 'image/png'));
      const embedded = await outPdf.embedPng(pngBytes);
      // Convert 2x raster pixels back to PDF points (72 DPI standard)
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
