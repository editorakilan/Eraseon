/**
 * Isolated Letterbox Black Bars & Container Aspect-Ratio Badge Module.
 * Uses directional vertical edge extrapolation for horizontal edge bars (zero mosaic)
 * and zero-mosaic SOR + exemplar fill for dark container badges.
 */
(function (global) {
  const M = (global.EraseonModules = global.EraseonModules || {});

  function detectLetterboxInRegion(gray, imgW, rx, ry, rw, rh) {
    const boxes = [];
    let topH = 0;
    const maxTop = Math.floor(rh * 0.16);
    for (let r = 0; r < maxTop; r++) {
      let sum = 0;
      const rowOff = (ry + r) * imgW + rx;
      for (let c = 0; c < rw; c++) sum += gray[rowOff + c];
      if (sum / rw < 30) topH = r + 1;
      else if (topH > 0) break;
    }
    if (topH >= 10) {
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
      if (sum / rw < 30) botH = rh - r;
      else if (botH > 0) break;
    }
    if (botH >= 10) {
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
    const { connectedComponents } = M.common;
    const boxes = [];
    const yStart = Math.floor(rh * 0.01);
    const yEnd = Math.floor(rh * 0.32);
    const xEnd = Math.floor(rw * 0.58);
    const subH = yEnd - yStart;
    if (xEnd < 40 || subH < 18) return boxes;

    const pillMask = new Uint8Array(xEnd * subH);
    for (let y = 0; y < subH; y++) {
      const rowOff = (ry + yStart + y) * imgW + rx;
      let fullRowDark = 0;
      for (let x = 0; x < rw; x++) {
        if (gray[rowOff + x] < 85) fullRowDark++;
      }
      if (fullRowDark > rw * 0.68) continue;
      for (let x = 0; x < xEnd; x++) {
        if (gray[rowOff + x] < 95) pillMask[y * xEnd + x] = 1;
      }
    }
    const closedPill = M.common.dilateBinary(pillMask, xEnd, subH, 3, 2);
    const { comps } = connectedComponents(closedPill, xEnd, subH);
    for (const c of comps) {
      const aspect = c.width / Math.max(1, c.height);
      const fill = c.area / Math.max(1, c.width * c.height);
      const inCorner = c.x <= rw * 0.14 || (c.x + c.width) >= rw * 0.72;
      if (c.width >= 50 && c.width <= rw * 0.52 && c.height >= 16 && c.height <= Math.max(84, Math.floor(rh * 0.14)) && aspect >= 2.0 && fill >= 0.62 && inCorner) {
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

  function extrapolateEdgeBars(imgData, barMask, w, h) {
    const { connectedComponents, clamp255 } = M.common;
    const out = new Uint8ClampedArray(imgData.data);
    const { labels, comps } = connectedComponents(barMask, w, h);

    for (const c of comps) {
      if (c.width <= w * 0.65) continue;
      const isTop = c.y <= Math.floor(h * 0.12);
      const isBot = (c.y + c.height) >= Math.floor(h * 0.88);
      if (!isTop && !isBot) continue;

      const refY = isTop ? Math.min(h - 1, c.y + c.height) : Math.max(0, c.y - 1);
      const refBandR = new Float32Array(c.width);
      const refBandG = new Float32Array(c.width);
      const refBandB = new Float32Array(c.width);

      // Average 4 clean active rows just inside image boundary
      for (let x = 0; x < c.width; x++) {
        const gx = c.x + x;
        let sr = 0, sg = 0, sb = 0, cnt = 0;
        for (let k = 0; k < 4; k++) {
          const sy = isTop ? Math.min(h - 1, refY + k) : Math.max(0, refY - k);
          if (!barMask[sy * w + gx]) {
            const p = (sy * w + gx) * 4;
            sr += out[p]; sg += out[p + 1]; sb += out[p + 2];
            cnt++;
          }
        }
        if (cnt > 0) {
          refBandR[x] = sr / cnt; refBandG[x] = sg / cnt; refBandB[x] = sb / cnt;
        }
      }

      for (let y = c.y; y < c.y + c.height; y++) {
        const dist = isTop ? (c.y + c.height - y) : (y - c.y + 1);
        const rad = Math.min(12, Math.max(1, Math.floor(dist * 0.25)));
        for (let x = 0; x < c.width; x++) {
          const gx = c.x + x;
          if (labels[y * w + gx] !== c.id) continue;
          let sr = 0, sg = 0, sb = 0, cnt = 0;
          const x0 = Math.max(0, x - rad), x1 = Math.min(c.width - 1, x + rad);
          for (let kx = x0; kx <= x1; kx++) {
            sr += refBandR[kx]; sg += refBandG[kx]; sb += refBandB[kx]; cnt++;
          }
          const p = (y * w + gx) * 4;
          out[p] = clamp255(sr / cnt);
          out[p + 1] = clamp255(sg / cnt);
          out[p + 2] = clamp255(sb / cnt);
          out[p + 3] = 255;
        }
      }
    }
    return new ImageData(out, w, h);
  }

  async function remove(srcImgData, binMask, w, h, onStatus = null) {
    const { connectedComponents, clientContentAwareInpaint } = M.common;
    const { labels, comps } = connectedComponents(binMask, w, h);
    const barMask = new Uint8Array(w * h);
    const badgeMask = new Uint8Array(w * h);
    let hasBar = false, hasBadge = false;

    for (const c of comps) {
      const isEdgeBar = (c.width > w * 0.65) && (c.height < h * 0.22) && (c.y <= Math.floor(h * 0.12) || (c.y + c.height) >= Math.floor(h * 0.88));
      for (let y = c.y; y < c.y + c.height; y++) {
        const off = y * w;
        for (let x = c.x; x < c.x + c.width; x++) {
          if (labels[off + x] === c.id) {
            if (isEdgeBar) { barMask[off + x] = 1; hasBar = true; }
            else { badgeMask[off + x] = 1; hasBadge = true; }
          }
        }
      }
    }

    let res = srcImgData;
    if (hasBar) res = extrapolateEdgeBars(res, barMask, w, h);
    if (hasBadge) {
      if (M.ai && M.ai.inpaint) res = await M.ai.inpaint(res, badgeMask, w, h, 64, onStatus);
      else res = clientContentAwareInpaint(res, badgeMask, w, h, 42, true);
    }
    return res;
  }

  M.letterbox = {
    detectLetterboxInRegion,
    detectContainerBadgesInRegion,
    remove
  };
})(window);
