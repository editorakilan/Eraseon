/**
 * Isolated Repeating Translucent Grid & Stock Watermark Module.
 * Contains PCA oriented polygon detection, 0.25px NCC template matching,
 * 2D Bravais lattice completion, and YCbCr sub-pixel median matte inversion.
 */
(function (global) {
  const M = (global.EraseonModules = global.EraseonModules || {});

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
      mx, my, ux, uy, vx, vy, minU, maxU, minV, maxV,
      polygon: [
        clampPt(minU, minV),
        clampPt(maxU, minV),
        clampPt(maxU, maxV),
        clampPt(minU, maxV)
      ]
    };
  }

  function findRepeatingTemplatePeaks(hpGray, w, h, seedBox, minNcc = 0.36) {
    const { resizeGrayBilinear, sampleBilinearFloat } = M.common;
    const sx0 = Math.max(0, Math.floor(seedBox.x));
    const sy0 = Math.max(0, Math.floor(seedBox.y));
    const sw = Math.min(w - sx0, Math.floor(seedBox.width));
    const sh = Math.min(h - sy0, Math.floor(seedBox.height));
    if (sw < 16 || sh < 16) return [];

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
        const signedNcc = dot / (Math.sqrt(normI) * sparseNorm);
        if (signedNcc >= minNcc * 0.80) {
          coarseCands.push({ px: px / proxyScale, py: py / proxyScale, ncc: signedNcc, signedNcc });
        }
      }
    }

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
      if (validCnt < fullPts.length * minFrac || normI < 1e-3) return { abs: -1, signed: -1 };
      const signed = dot / (Math.sqrt(normI) * normFull);
      return { abs: Math.abs(signed), signed };
    };

    for (const cand of coarseCands) {
      let tooClose = false;
      for (const pk of peaks) {
        const dx = cand.px - pk.px, dy = cand.py - pk.py;
        if (dx * dx + dy * dy < nmsDistSq) { tooClose = true; break; }
      }
      if (tooClose) continue;

      let bestX = Math.round(cand.px), bestY = Math.round(cand.py), bestNcc = -1, bestSigned = 0;
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
          const signed = dot / (Math.sqrt(normI) * normFull);
          if (signed > bestNcc) { bestNcc = signed; bestSigned = signed; bestX = rx; bestY = ry; }
        }
      }

      if (bestNcc < minNcc) continue;

      let subX = bestX, subY = bestY, subNcc = bestNcc, subSigned = bestSigned;
      for (let dy = -0.5; dy <= 0.501; dy += 0.25) {
        for (let dx = -0.5; dx <= 0.501; dx += 0.25) {
          if (dx === 0 && dy === 0) continue;
          const r = evalNccBilinear(bestX + dx, bestY + dy);
          if (r.signed > subNcc) {
            subNcc = r.signed;
            subSigned = r.signed;
            subX = bestX + dx;
            subY = bestY + dy;
          }
        }
      }

      let dup = false;
      for (const pk of peaks) {
        const dx = subX - pk.px, dy = subY - pk.py;
        if (dx * dx + dy * dy < nmsDistSq) { dup = true; break; }
      }
      if (!dup) {
        peaks.push({ px: subX, py: subY, ncc: subNcc, signedNcc: subSigned, sw, sh });
      }
      if (peaks.length >= 48) break;
    }

    const strongInitCount = peaks.filter((pk) => pk.ncc >= 0.55).length;
    let verifiedLattice = false;
    if (peaks.length >= 3 && strongInitCount >= 2) {
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
          if (inliers >= 3 && score > bestScore) {
            bestScore = score;
            bestLattice = { v1, v2, p0, det };
          }
        }
      }

      if (bestLattice) {
        let minPx = 1e9, maxPx = -1e9, minPy = 1e9, maxPy = -1e9;
        for (const pk of peaks) {
          if (pk.px < minPx) minPx = pk.px;
          if (pk.px > maxPx) maxPx = pk.px;
          if (pk.py < minPy) minPy = pk.py;
          if (pk.py > maxPy) maxPy = pk.py;
        }
        if ((maxPx - minPx) >= w * 0.20 && (maxPy - minPy) >= h * 0.20) {
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
              const rNcc = evalNccBilinear(gx, gy, 0.12);
              if (rNcc.abs >= 0.045) {
                peaks.push({ px: gx, py: gy, ncc: rNcc.abs, signedNcc: rNcc.signed, sw, sh });
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

  function detectInRegion(gray, imgW, imgH, rx, ry, rw, rh, sensitivity, dilatePx, allowNonRepeating = false) {
    const { resizeGrayBilinear, morphOpenGray, morphCloseGray, dilateBinary, connectedComponents, fastBoxBlurFloat } = M.common;
    const subFull = new Float32Array(rw * rh);
    for (let y = 0; y < rh; y++) {
      const srcOff = (ry + y) * imgW + rx;
      const dstOff = y * rw;
      for (let x = 0; x < rw; x++) {
        subFull[dstOff + x] = gray[srcOff + x];
      }
    }

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
      if (c.width > dw * 0.55 || c.height > dh * 0.45) continue;

      const rawPts = [];
      let bgHigh = 0, darkLow = 0, samples = 0;
      for (let yy = c.y; yy < c.y + c.height; yy++) {
        const off = yy * dw;
        for (let xx = c.x; xx < c.x + c.width; xx++) {
          if (labels[off + xx] === c.id && bin[off + xx]) rawPts.push([xx, yy]);
          if ((yy & 1) === 0 && (xx & 1) === 0) {
            const v = sub[off + xx];
            if (v > 235) bgHigh++;
            if (v < 95) darkLow++;
            samples++;
          }
        }
      }
      if (rawPts.length < 22) continue;
      if (samples > 0 && (bgHigh / samples > 0.45 && darkLow / samples > 0.03)) continue;

      const orient = computeOrientedPolygonFromPoints(rawPts, padScaled, dw, dh);
      if (!orient) continue;
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
        _seedScore: (orient.angleDeg > 8 ? 2.0 : 1.0) * Math.min(3.5, aspect) * Math.sqrt(rawPts.length) / (1.0 + bgVar * 0.55)
      };
      if (orient.angleDeg > 8) {
        box.polygon = orient.polygon.map(([px, py]) => [
          Math.max(0, Math.min(imgW - 1, Math.round(rx + px / detScale))),
          Math.max(0, Math.min(imgH - 1, Math.round(ry + py / detScale)))
        ]);
        box._orient = orient;
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
      const peaks = findRepeatingTemplatePeaks(hpSub, rw, rh, localSeed, 0.42);
      const strongCount = peaks.filter((p) => p.ncc >= 0.55).length;
      if (peaks._verifiedLattice && peaks.length >= 3 && strongCount >= 2) {
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
            if (bestSeed._orient) {
              const o = bestSeed._orient;
              const sc = 1.0 / detScale;
              const mxW = rx + o.mx * sc + dx;
              const myW = ry + o.my * sc + dy;
              const u0 = o.minU * sc - 14, u1 = o.maxU * sc + 14;
              const v0 = o.minV * sc - 3, v1 = o.maxV * sc + 3;
              const toPt = (u, v) => [
                Math.max(0, Math.min(imgW - 1, Math.round(mxW + u * o.ux + v * o.vx))),
                Math.max(0, Math.min(imgH - 1, Math.round(myW + u * o.uy + v * o.vy)))
              ];
              b.polygon = [toPt(u0, v0), toPt(u1, v0), toPt(u1, v1), toPt(u0, v1)];
            } else {
              b.polygon = bestSeed.polygon.map(([px, py]) => [
                Math.max(0, Math.min(imgW - 1, Math.round(px + dx))),
                Math.max(0, Math.min(imgH - 1, Math.round(py + dy)))
              ]);
            }
            b.angle = bestSeed.angle;
          }
          expandedBoxes.push(b);
        }
        return expandedBoxes;
      }
    }

    if (!allowNonRepeating) return [];
    boxes.sort((a, b) => (b._seedScore || 0) - (a._seedScore || 0));
    const cleanNonRep = boxes.filter((b) => Math.max(b.width, b.height) >= 40 / detScale && b.width <= imgW * 0.35 && b.height <= imgH * 0.20).slice(0, 6);
    for (const b of cleanNonRep) delete b._seedScore;
    return cleanNonRep;
  }

  async function tryDekelMultiInstanceClient(srcImgData, binMask, w, h, colorHint, boxesHint = null, onStatus = null) {
    const {
      extractGray, fastBoxBlurFloat, connectedComponents, sampleBilinearFloat,
      sampleBilinearRGB, dilateBinary, harmonicMaskedFill, clientContentAwareInpaint,
      rgbToY, clamp255
    } = M.common;

    const src = srcImgData.data;
    const ySrc = extractGray(src, w, h);
    const blurR = Math.max(5, Math.round(Math.min(w, h) * 0.018));
    const blurMap = fastBoxBlurFloat(ySrc, w, h, blurR);
    const hpY = new Float32Array(w * h);
    for (let i = 0; i < hpY.length; i++) hpY[i] = ySrc[i] - blurMap[i];

    let seedCands = [];
    if (Array.isArray(boxesHint) && boxesHint.length >= 3) {
      seedCands = boxesHint.filter((b) => b.type === 'translucent_grid' && b.width >= 16 && b.height >= 16);
    }
    if (seedCands.length < 3) {
      const { comps } = connectedComponents(binMask, w, h);
      seedCands = comps.filter((c) => c.width >= 16 && c.height >= 16 && c.area >= 80);
    }
    if (seedCands.length < 3) return null;

    let maxSeedArea = 1;
    for (const c of seedCands) {
      if (c.width * c.height > maxSeedArea) maxSeedArea = c.width * c.height;
    }

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
      const areaFrac = (c.width * c.height) / maxSeedArea;
      const borderPen = (c.x <= 2 || c.y <= 2 || c.x + c.width >= w - 2 || c.y + c.height >= h - 2) ? 0.72 : 1.0;
      const score = borderPen * (areaFrac * areaFrac) * (hpEnergy / cnt) * Math.sqrt(cnt) / (1.0 + stdBg * 0.35);
      if (score > bestScore) {
        bestScore = score;
        bestComp = c;
      }
    }
    if (!bestComp) return null;

    // Pad bestComp symmetrically so leading glyphs (e.g. '©') separated by a space are included in the learned template
    const padComp = 14;
    const cx0 = Math.max(0, bestComp.x - padComp);
    const cy0 = Math.max(0, bestComp.y - padComp);
    const cx1 = Math.min(w, bestComp.x + bestComp.width + padComp);
    const cy1 = Math.min(h, bestComp.y + bestComp.height + padComp);
    const learnComp = { x: cx0, y: cy0, width: cx1 - cx0, height: cy1 - cy0 };

    const peaks = findRepeatingTemplatePeaks(hpY, w, h, learnComp, 0.36);
    if (peaks.length < 3) return null;

    const pw = learnComp.width;
    const ph = learnComp.height;
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

    const mattePeaks = peaks.filter((pk) => pk.ncc >= 0.36);
    const learnPeaks = mattePeaks.length >= 3 ? mattePeaks : peaks;

    const medHp = new Float32Array(patchLen);
    const hpSamples = new Float32Array(learnPeaks.length);
    for (let i = 0; i < patchLen; i++) {
      const lx = i % pw;
      const ly = (i - lx) / pw;
      let k = 0;
      for (let pIdx = 0; pIdx < learnPeaks.length; pIdx++) {
        const pk = learnPeaks[pIdx];
        const gx = pk.px + lx;
        const gy = pk.py + ly;
        if (gx < 0 || gx > w - 1 || gy < 0 || gy > h - 1) continue;
        const val = sampleBilinearFloat(hpY, w, h, gx, gy);
        const sgn = (pk.signedNcc !== undefined && pk.signedNcc < 0) ? -1 : 1;
        hpSamples[k++] = (isLight ? val : -val) * sgn;
      }
      if (k >= 2) {
        const sub = hpSamples.subarray(0, k).slice().sort();
        medHp[i] = sub[Math.floor(k * 0.5)];
      }
    }

    const rawStroke = new Uint8Array(patchLen);
    let rawStrokeCount = 0;
    for (let i = 0; i < patchLen; i++) {
      if (medHp[i] >= 2.8) {
        rawStroke[i] = 1;
        rawStrokeCount++;
      }
    }
    if (rawStrokeCount < 20) return null;
    const stripPatch = dilateBinary(rawStroke, pw, ph, 1);

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

    const manualMaskOnly = !Array.isArray(boxesHint) || boxesHint.length === 0;
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
          if ((manualMaskOnly || overlapPeaks <= 1) && !binMask[gIdx]) continue;
          if (stripPatch[y * pw + x]) globalStripMask[gIdx] = 1;
        }
      }
    }

    const cleanBg = harmonicMaskedFill(src, globalStripMask, w, h, 24);

    let effTargetY = wmTargetY;
    if (!colorHint || colorHint === 'auto') {
      let sumB = 0, sumD = 0, sumBB = 0, sumBD = 0, regN = 0;
      for (let i = 0; i < patchLen; i++) {
        if (!rawStroke[i]) continue;
        const lx = i % pw;
        const ly = (i - lx) / pw;
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
          const d = yT - yB;
          sumB += yB; sumD += d; sumBB += yB * yB; sumBD += yB * d; regN++;
        }
      }
      if (regN >= 40) {
        const meanB = sumB / regN;
        const meanD = sumD / regN;
        const varB = (sumBB / regN) - meanB * meanB;
        const covBD = (sumBD / regN) - meanB * meanD;
        if (varB > 180.0) {
          const slopeAlpha = -covBD / varB;
          if (slopeAlpha >= 0.08 && slopeAlpha <= 0.72) {
            effTargetY = Math.max(0.0, Math.min(255.0, meanB + meanD / slopeAlpha));
          }
        }
      }
    }

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
        const diffW = effTargetY - yB;
        if (Math.abs(diffW) >= 22.0) {
          ySamples[ky++] = Math.max(-0.05, Math.min(0.55, (yT - yB) / diffW));
        }

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
        const ay = rawMed > 0 ? Math.min(0.54, rawMed * 1.05) : Math.max(-0.045, rawMed);
        alphaY[i] = Math.abs(ay) >= 0.009 ? ay : 0.0;
      }
      if (kc >= 2) {
        const subC = cSamples.subarray(0, kc).slice().sort();
        const ac = Math.max(0.0, Math.min(0.46, subC[Math.floor(kc * 0.5)] * 1.05));
        alphaC[i] = ac >= 0.010 ? ac : 0.0;
      }
    }

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
          if ((manualMaskOnly || overlapPeaks <= 1) && !binMask[gIdx]) continue;
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

      const inv = Math.max(0.45, 1.0 - ay);
      const yRawRec = (yVal - ay * effTargetY) / inv;
      const cleanB = rgbToY(cleanBg.rArr[i], cleanBg.gArr[i], cleanBg.bArr[i]);
      let yRec = yRawRec;
      if (Math.abs(effTargetY - cleanB) < 26.0) {
        yRec = cleanB * 0.55 + yRawRec * 0.45;
      }
      if (ay > 0 && effTargetY > cleanB && yRec < cleanB * 0.86) {
        yRec = Math.max(0.0, cleanB * 0.86);
      } else if (ay > 0 && effTargetY < cleanB && yRec > cleanB * 1.14) {
        yRec = Math.min(255.0, cleanB * 1.14);
      }

      const lumFactor = yRec < yVal ? Math.max(0.45, (yRec + 18.0) / (yVal + 18.0)) : 1.0;
      const cInv = Math.max(0.65, 1.0 - ac * 0.65);
      const cbRec = 128.0 + ((cb - 128.0) / cInv) * lumFactor;
      const crRec = 128.0 + ((cr - 128.0) / cInv) * lumFactor;

      out[p] = clamp255(yRec + 1.402 * (crRec - 128.0));
      out[p + 1] = clamp255(yRec - 0.344136 * (cbRec - 128.0) - 0.714136 * (crRec - 128.0));
      out[p + 2] = clamp255(yRec + 1.772 * (cbRec - 128.0));
    }

    const med3 = new Uint8ClampedArray(out);
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
      }
    }

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
      if (ay === 0 && !globalStripMask[i]) continue;
      const hpStd = Math.sqrt(Math.max(0, lapVar[i]));
      const cleanB = cleanGray[i];
      const p = i * 4;
      const outY = rgbToY(out[p], out[p + 1], out[p + 2]);
      const residResidual = Math.abs(outY - cleanB);
      const isCrossover = Math.abs(effTargetY - cleanB) < 28.0 && (Math.abs(ay) > 0.10 || residResidual > 16.0);
      if ((Math.abs(ay) >= 0.012 && hpStd < 3.6) || (isCrossover && hpStd < 6.0)) {
        smoothHole[i] = 1;
        hasSmooth = true;
      } else if (ay > 0.14 && hpStd >= 3.6) {
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
      resultImg = clientContentAwareInpaint(resultImg, smoothHole, w, h, 24, false);
    }
    return { image: resultImg, cleanedMask: globalStripMask };
  }

  async function remove(srcImgData, binMask, w, h, colorHint = 'auto', alphaHint = 0.5, boxesHint = null, onStatus = null) {
    const { connectedComponents, extractGray, morphOpenGray, fastBoxBlurFloat, dilateBinary } = M.common;
    let srcImg = srcImgData;
    let activeMask = binMask;
    const dekel = await tryDekelMultiInstanceClient(srcImgData, binMask, w, h, colorHint, boxesHint, onStatus);
    if (dekel) {
      srcImg = dekel.image;
      if (M.ai && M.ai.inpaint) {
        const curGray = extractGray(srcImg.data, w, h);
        const open6 = morphOpenGray(curGray, w, h, 6);
        const close6 = M.common.morphCloseGray(curGray, w, h, 6);
        const blur2 = fastBoxBlurFloat(curGray, w, h, 2);
        const hpSq = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) { const d = curGray[i] - blur2[i]; hpSq[i] = d * d; }
        const hpVar = fastBoxBlurFloat(hpSq, w, h, 3);
        const cleanZone = dilateBinary(dekel.cleanedMask, w, h, 1, 1);
        const residualRaw = new Uint8Array(w * h);
        let resCnt = 0;
        for (let i = 0; i < w * h; i++) {
          if (!cleanZone[i]) continue;
          const hpStd = Math.sqrt(Math.max(0, hpVar[i]));
          if (hpStd >= 16.0 && curGray[i] < 80) continue; // Protect sharp dark eye/eyelash features
          const th = curGray[i] - open6[i];
          const bh = close6[i] - curGray[i];
          if (th >= 3.2 || (bh >= 3.5 && curGray[i] > 110)) {
            residualRaw[i] = 1;
            resCnt++;
          }
        }
        if (resCnt >= 20) {
          const residualMask = dilateBinary(residualRaw, w, h, 1, 1);
          srcImg = await M.ai.inpaint(srcImg, residualMask, w, h, 64, onStatus);
        }
      }
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

    return await M.text.remove(srcImg, activeMask, w, h, colorHint, alphaHint, onStatus);
  }

  M.grid = { detectInRegion, remove };
})(window);
