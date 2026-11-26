/**
 * Eraseon Shared Client Utilities & Zero-Mosaic Harmonic SOR + Texture Synthesis Engine
 */
(function (global) {
  const M = (global.EraseonModules = global.EraseonModules || {});

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
    if (radius <= 0) return new Float32Array(src);
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

  /**
   * Zero-Mosaic Multi-Scale SOR Harmonic Pyramid Fill:
   * - Pins true unmasked pixels (v >= 0.99) strictly as Dirichlet boundaries.
   * - Relaxes EVERY level including lvl=0 using Successive Over-Relaxation (omega = 1.65)
   *   so coarse pyramid grid blocks are 100% eliminated.
   */
  function harmonicMaskedFill(srcData, binMask, w, h, fineSorPasses = 42) {
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
    while (cur.w > 3 && cur.h > 3 && levels.length < 10) {
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

    const relaxLevelSOR = (lv, iters, omega = 1.65) => {
      const lw = lv.w, lh = lv.h;
      const unk = [];
      for (let i = 0; i < lw * lh; i++) {
        if (lv.v[i] < 0.99) unk.push(i);
      }
      if (!unk.length) return;

      for (let pass = 0; pass < iters; pass++) {
        for (let u = 0; u < unk.length; u++) {
          const i = unk[u];
          const x = i % lw, y = (i - x) / lw;
          let sr = 0, sg = 0, sb = 0, sw = 0;
          if (x > 0) { sr += lv.r[i - 1]; sg += lv.g[i - 1]; sb += lv.b[i - 1]; sw += 1; }
          if (x + 1 < lw) { sr += lv.r[i + 1]; sg += lv.g[i + 1]; sb += lv.b[i + 1]; sw += 1; }
          if (y > 0) { sr += lv.r[i - lw]; sg += lv.g[i - lw]; sb += lv.b[i - lw]; sw += 1; }
          if (y + 1 < lh) { sr += lv.r[i + lw]; sg += lv.g[i + lw]; sb += lv.b[i + lw]; sw += 1; }
          if (sw > 0) {
            const avgR = sr / sw;
            const avgG = sg / sw;
            const avgB = sb / sw;
            const anchor = lv.v[i] * 0.72;
            const targetR = anchor * lv.r[i] + (1 - anchor) * avgR;
            const targetG = anchor * lv.g[i] + (1 - anchor) * avgG;
            const targetB = anchor * lv.b[i] + (1 - anchor) * avgB;
            // SOR extrapolation only on pure unknown pixels (v == 0) for unconditional stability
            const wSor = lv.v[i] === 0 ? omega : 1.0;
            lv.r[i] = Math.max(0, Math.min(255, lv.r[i] + wSor * (targetR - lv.r[i])));
            lv.g[i] = Math.max(0, Math.min(255, lv.g[i] + wSor * (targetG - lv.g[i])));
            lv.b[i] = Math.max(0, Math.min(255, lv.b[i] + wSor * (targetB - lv.b[i])));
          }
        }
      }
    };

    // Relax coarsest level completely
    relaxLevelSOR(levels[levels.length - 1], 45, 1.5);

    // Coarse-to-fine V-cycle with SOR at every single level INCLUDING lvl = 0!
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
          if (fine.v[fi] >= 0.99) continue;

          const cx = (x + 0.5) * (coarse.w / fine.w) - 0.5;
          const x0 = Math.max(0, Math.min(coarse.w - 1, Math.floor(cx)));
          const x1 = Math.max(0, Math.min(coarse.w - 1, x0 + 1));
          const fx = Math.max(0, Math.min(1, cx - x0));

          const i00 = y0 * coarse.w + x0, i10 = y0 * coarse.w + x1;
          const i01 = y1 * coarse.w + x0, i11 = y1 * coarse.w + x1;
          const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
          const norm = w00 + w10 + w01 + w11;
          const cr = (w00 * coarse.r[i00] + w10 * coarse.r[i10] + w01 * coarse.r[i01] + w11 * coarse.r[i11]) / norm;
          const cg = (w00 * coarse.g[i00] + w10 * coarse.g[i10] + w01 * coarse.g[i01] + w11 * coarse.g[i11]) / norm;
          const cb = (w00 * coarse.b[i00] + w10 * coarse.b[i10] + w01 * coarse.b[i01] + w11 * coarse.b[i11]) / norm;
          const mix = fine.v[fi];
          fine.r[fi] = mix * fine.r[fi] + (1 - mix) * cr;
          fine.g[fi] = mix * fine.g[fi] + (1 - mix) * cg;
          fine.b[fi] = mix * fine.b[fi] + (1 - mix) * cb;
        }
      }
      const lvlIters = lvl === 0 ? fineSorPasses : Math.min(64, 34 + lvl * 4);
      relaxLevelSOR(fine, lvlIters, lvl === 0 ? 1.55 : 1.62);
    }

    return { rArr, gArr, bArr };
  }

  /**
   * Zero-Mosaic Harmonic + Coherent Border-Reflected Exemplar Texture Synthesis + Sensor Grain.
   * Smoothly transitions high-frequency natural background texture (asphalt, grass, trees, clouds)
   * across masked logos/badges/stars on top of the seamless SOR harmonic surface.
   */
  function clientContentAwareInpaint(srcImgData, binMask, w, h, pdePasses = 42, useExemplar = true) {
    const src = srcImgData.data;
    const out = new Uint8ClampedArray(src);
    const len = w * h;

    const maskedIdx = [];
    for (let i = 0; i < len; i++) {
      if (binMask[i]) maskedIdx.push(i);
    }
    if (maskedIdx.length === 0) return srcImgData;

    const { rArr, gArr, bArr } = harmonicMaskedFill(src, binMask, w, h, Math.max(24, pdePasses));

    // Measure local ring texture & sensor noise
    const dilatedOuter = dilateBinary(binMask, w, h, 10);
    const srcR = new Float32Array(len);
    const srcG = new Float32Array(len);
    const srcB = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      const p = i * 4;
      srcR[i] = src[p];
      srcG[i] = src[p + 1];
      srcB[i] = src[p + 2];
    }

    let smoothR = null, smoothG = null, smoothB = null;
    if (useExemplar) {
      smoothR = fastBoxBlurFloat(srcR, w, h, 3);
      smoothG = fastBoxBlurFloat(srcG, w, h, 3);
      smoothB = fastBoxBlurFloat(srcB, w, h, 3);
    }

    const blurGray = fastBoxBlurFloat(extractGray(src, w, h), w, h, 2);
    let sqSum = 0, ringCnt = 0;
    for (let i = 0; i < len; i++) {
      if (dilatedOuter[i] && !binMask[i]) {
        const d = rgbToY(srcR[i], srcG[i], srcB[i]) - blurGray[i];
        sqSum += d * d;
        ringCnt++;
      }
    }
    const ringStd = ringCnt > 20 ? Math.sqrt(sqSum / ringCnt) : 0;
    const noiseStd = ringCnt > 20 ? Math.min(3.5, ringStd * 0.35) : 0.8;

    // Nearest clean boundary pixel BFS for coherent border-reflected high-frequency texture
    let nearestX = null, nearestY = null, distMap = null;
    if (useExemplar && ringStd > 2.4) {
      nearestX = new Int32Array(len);
      nearestY = new Int32Array(len);
      distMap = new Float32Array(len);
      const visited = new Uint8Array(len);
      const queue = new Int32Array(len);
      let head = 0, tail = 0;

      // Seed BFS with immediate clean ring just outside mask (2px inside unmasked region avoids edge bleed)
      const maskDil1 = dilateBinary(binMask, w, h, 1);
      for (let y = 0; y < h; y++) {
        const off = y * w;
        for (let x = 0; x < w; x++) {
          const i = off + x;
          if (!binMask[i] && maskDil1[i]) {
            visited[i] = 1;
            nearestX[i] = x;
            nearestY[i] = y;
            distMap[i] = 0;
            queue[tail++] = i;
          }
        }
      }

      while (head < tail) {
        const cur = queue[head++];
        const cx = cur % w;
        const cy = (cur - cx) / w;
        const nx0 = nearestX[cur];
        const ny0 = nearestY[cur];
        const dNext = distMap[cur] + 1;

        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = cx + dx;
            if (nx < 0 || nx >= w) continue;
            const ni = ny * w + nx;
            if (binMask[ni] && !visited[ni]) {
              visited[ni] = 1;
              nearestX[ni] = nx0;
              nearestY[ni] = ny0;
              distMap[ni] = dNext;
              queue[tail++] = ni;
            }
          }
        }
      }
    }

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
      const x = idx % w;
      const y = (idx - x) / w;

      let r = rArr[idx];
      let g = gArr[idx];
      let b = bArr[idx];

      if (nearestX !== null) {
        const bx = nearestX[idx];
        const by = nearestY[idx];
        const dist = distMap[idx];
        // Reflect vector across clean boundary (clamp reflection depth to 18px clean ring)
        const dx = Math.max(-18, Math.min(18, x - bx));
        const dy = Math.max(-18, Math.min(18, y - by));
        const rx = Math.max(0, Math.min(w - 1, bx - dx));
        const ry = Math.max(0, Math.min(h - 1, by - dy));
        const rIdx = ry * w + rx;

        if (!binMask[rIdx]) {
          // Only transfer subtle natural micro-texture (<=11); reject high-contrast text/object edges
          const dR = srcR[rIdx] - smoothR[rIdx];
          const dG = srcG[rIdx] - smoothG[rIdx];
          const dB = srcB[rIdx] - smoothB[rIdx];
          const edgeMag = Math.max(Math.abs(dR), Math.abs(dG), Math.abs(dB));
          if (edgeMag <= 11.0) {
            const atten = Math.max(0.20, Math.exp(-dist * 0.045));
            r += Math.max(-8, Math.min(8, dR * atten));
            g += Math.max(-8, Math.min(8, dG * atten));
            b += Math.max(-8, Math.min(8, dB * atten));
          }
        }
      }

      const n = nextNoise();
      const p = idx * 4;
      out[p] = clamp255(r + n);
      out[p + 1] = clamp255(g + n);
      out[p + 2] = clamp255(b + n);
      out[p + 3] = 255;
    }

    return new ImageData(out, w, h);
  }

  function enhanceCleanedImage(imgData) {
    const w = imgData.width, h = imgData.height;
    const data = imgData.data;
    const len = w * h;
    const yArr = new Float32Array(len);
    const cbArr = new Float32Array(len);
    const crArr = new Float32Array(len);

    for (let i = 0; i < len; i++) {
      const p = i * 4;
      const r = data[p], g = data[p + 1], b = data[p + 2];
      yArr[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      cbArr[i] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
      crArr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
    }

    const yBlur = fastBoxBlurFloat(yArr, w, h, 1);
    const cbSmooth = fastBoxBlurFloat(cbArr, w, h, 1);
    const crSmooth = fastBoxBlurFloat(crArr, w, h, 1);

    const out = new Uint8ClampedArray(data);
    for (let i = 0; i < len; i++) {
      const y = yArr[i];
      const detail = y - yBlur[i];
      const boost = Math.abs(detail) >= 2.2
        ? Math.max(-16.0, Math.min(16.0, detail * 0.30))
        : 0.0;
      const normY = y / 255.0;
      const sCurve = (normY - 0.5) * (1.0 - Math.abs(normY - 0.5) * 2.0) * 8.0;
      const yOut = Math.max(0.0, Math.min(255.0, y + boost + sCurve));

      const cbOut = cbArr[i] * 0.60 + cbSmooth[i] * 0.40;
      const crOut = crArr[i] * 0.60 + crSmooth[i] * 0.40;

      const p = i * 4;
      out[p] = clamp255(yOut + 1.402 * (crOut - 128.0));
      out[p + 1] = clamp255(yOut - 0.344136 * (cbOut - 128.0) - 0.714136 * (crOut - 128.0));
      out[p + 2] = clamp255(yOut + 1.772 * (cbOut - 128.0));
      out[p + 3] = 255;
    }
    return new ImageData(out, w, h);
  }

  M.common = {
    clamp255,
    rgbToY,
    extractGray,
    fastBoxBlurFloat,
    sampleBilinearFloat,
    sampleBilinearRGB,
    resizeGrayBilinear,
    morphOpenGray,
    morphCloseGray,
    dilateBinary,
    connectedComponents,
    findEmbeddedPhotoRegions,
    harmonicMaskedFill,
    clientContentAwareInpaint,
    enhanceCleanedImage
  };
})(window);
