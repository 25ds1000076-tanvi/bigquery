'use strict';

const express = require('express');
const crypto = require('crypto');

const app = express();
app.use(express.json({ limit: '15mb' }));

// ---------------------------------------------------------------------------
// In-memory "database". Keyed by runId. Only SELECT requests are persisted
// here (that's the only phase the spec asks us to persist/replay/conflict-
// check). EVALUATE phase reads from this store to verify lineage, but does
// not write to it.
// ---------------------------------------------------------------------------
const store = new Map();

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function isPlainObject(v) {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function isSafeNonNegInt(n) {
  return typeof n === 'number' && Number.isInteger(n) && Number.isSafeInteger(n) && n >= 0;
}

function isPositiveInt(n) {
  return isSafeNonNegInt(n) && n > 0;
}

function isFiniteNum(n) {
  return typeof n === 'number' && Number.isFinite(n);
}

function isNonEmptyString(v, maxLen) {
  return typeof v === 'string' && v.length > 0 && (maxLen === undefined || v.length <= maxLen);
}

function round12(x) {
  return Math.round(x * 1e12) / 1e12;
}

function utf8Bytes(str) {
  return Buffer.from(str, 'utf8');
}

function utf8Compare(a, b) {
  return Buffer.compare(utf8Bytes(a), utf8Bytes(b));
}

function sortByUtf8(arr) {
  return [...arr].sort(utf8Compare);
}

function dedupeSortedCodes(codes) {
  const sorted = sortByUtf8(codes);
  const out = [];
  for (const c of sorted) {
    if (out.length === 0 || out[out.length - 1] !== c) out.push(c);
  }
  return out;
}

function sha256Hex(str) {
  return crypto.createHash('sha256').update(str, 'utf8').digest('hex');
}

// deep structural equality, key-order independent
function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (a === null || b === null) return a === b;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) return false;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!deepEqual(a[i], b[i])) return false;
    return true;
  }
  if (typeof a === 'object') {
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.length !== kb.length) return false;
    for (const k of ka) {
      if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
      if (!deepEqual(a[k], b[k])) return false;
    }
    return true;
  }
  return a === b;
}

// ---------------------------------------------------------------------------
// Timestamp parsing: YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)
// Returns epoch milliseconds (UTC) or null if invalid.
// ---------------------------------------------------------------------------
const TS_RE = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$/;

function parseTimestamp(s) {
  if (typeof s !== 'string') return null;
  const m = TS_RE.exec(s);
  if (!m) return null;
  const [, yy, mo, dd, hh, mi, se, frac, tz] = m;
  const year = +yy, month = +mo, day = +dd, hour = +hh, min = +mi, sec = +se;
  if (month < 1 || month > 12) return null;
  if (day < 1 || day > 31) return null;
  if (hour > 23) return null;
  if (min > 59) return null;
  if (sec > 59) return null;

  let ms = 0;
  if (frac) ms = parseInt((frac + '000').slice(0, 3), 10);

  let offsetMin = 0;
  if (tz !== 'Z') {
    const sign = tz[0] === '-' ? -1 : 1;
    const th = parseInt(tz.slice(1, 3), 10);
    const tm = parseInt(tz.slice(4, 6), 10);
    if (th > 23 || tm > 59) return null;
    offsetMin = sign * (th * 60 + tm);
  }

  const utcMs = Date.UTC(year, month - 1, day, hour, min, sec, ms) - offsetMin * 60000;

  // Reject calendar overflow (e.g. Feb 30) by round-tripping the *local*
  // components (pre-offset) through Date.UTC and checking they survived.
  const check = new Date(Date.UTC(year, month - 1, day, hour, min, sec, ms));
  if (
    check.getUTCFullYear() !== year ||
    check.getUTCMonth() !== month - 1 ||
    check.getUTCDate() !== day ||
    check.getUTCHours() !== hour ||
    check.getUTCMinutes() !== min ||
    check.getUTCSeconds() !== sec
  ) {
    return null;
  }
  return utcMs;
}

// ---------------------------------------------------------------------------
// SELECT phase
// ---------------------------------------------------------------------------

function validateSelectBody(body) {
  const errors = [];
  if (!isNonEmptyString(body.runId, 128)) errors.push('runId');
  if (!Array.isArray(body.forbiddenFeatures) || !body.forbiddenFeatures.every((f) => typeof f === 'string')) {
    errors.push('forbiddenFeatures');
  }
  if (!isPositiveInt(body.numTrialsLimit)) errors.push('numTrialsLimit');
  if (!Array.isArray(body.rows) || body.rows.length === 0) errors.push('rows');
  if (!Array.isArray(body.trials)) errors.push('trials');

  if (errors.length) return { ok: false };

  // Validate each row
  const rowIds = new Set();
  const parsedRows = [];
  for (const r of body.rows) {
    if (!isPlainObject(r)) return { ok: false };
    if (!isNonEmptyString(r.id)) return { ok: false };
    if (rowIds.has(r.id)) return { ok: false };
    rowIds.add(r.id);
    if (!isNonEmptyString(r.entity)) return { ok: false };
    const eventMs = parseTimestamp(r.eventTime);
    if (eventMs === null) return { ok: false };
    const predMs = parseTimestamp(r.predictionTime);
    if (predMs === null) return { ok: false };
    if (!isSafeNonNegInt(r.version)) return { ok: false };
    if (r.split !== 'TRAIN' && r.split !== 'EVAL') return { ok: false };
    if (!isPlainObject(r.features)) return { ok: false };
    const features = {};
    for (const [name, fv] of Object.entries(r.features)) {
      if (!isPlainObject(fv)) return { ok: false };
      if (!('value' in fv)) return { ok: false };
      const availMs = parseTimestamp(fv.availableAt);
      if (availMs === null) return { ok: false };
      features[name] = { value: fv.value, availableAtMs: availMs };
    }
    parsedRows.push({
      id: r.id,
      entity: r.entity,
      eventMs,
      predMs,
      version: r.version,
      split: r.split,
      features,
    });
  }

  // Validate trials
  const trialIds = new Set();
  const parsedTrials = [];
  for (const t of body.trials) {
    if (!isPlainObject(t)) return { ok: false };
    if (!isSafeNonNegInt(t.trialId)) return { ok: false };
    if (trialIds.has(t.trialId)) return { ok: false };
    trialIds.add(t.trialId);
    if (t.status !== 'SUCCEEDED' && t.status !== 'FAILED') return { ok: false };
    if (typeof t.evalMetric !== 'number') return { ok: false };
    parsedTrials.push({ trialId: t.trialId, status: t.status, evalMetric: t.evalMetric });
  }

  return {
    ok: true,
    runId: body.runId,
    forbiddenFeatures: new Set(body.forbiddenFeatures),
    numTrialsLimit: body.numTrialsLimit,
    rows: parsedRows,
    trials: parsedTrials,
  };
}

function processSelect(v) {
  // 1. Dedupe by [entity, UTC(eventTime)]; keep highest version, then
  //    UTF-8-byte-smallest id.
  const groups = new Map();
  for (const row of v.rows) {
    const key = row.entity + '\u0000' + row.eventMs;
    const existing = groups.get(key);
    if (!existing) {
      groups.set(key, row);
    } else if (
      row.version > existing.version ||
      (row.version === existing.version && utf8Compare(row.id, existing.id) < 0)
    ) {
      groups.set(key, row);
    }
  }
  const retained = [...groups.values()];

  // 2. Eligible features: present in every retained row, not forbidden,
  //    and availableAt <= predictionTime in every retained row.
  let candidateNames = null;
  for (const row of retained) {
    const names = new Set(Object.keys(row.features));
    candidateNames = candidateNames === null ? names : new Set([...candidateNames].filter((n) => names.has(n)));
  }
  candidateNames = candidateNames || new Set();

  const eligible = [];
  for (const name of candidateNames) {
    if (v.forbiddenFeatures.has(name)) continue;
    let ok = true;
    for (const row of retained) {
      const fv = row.features[name];
      if (!fv || fv.availableAtMs > row.predMs) {
        ok = false;
        break;
      }
    }
    if (ok) eligible.push(name);
  }

  const trainRowIds = sortByUtf8(retained.filter((r) => r.split === 'TRAIN').map((r) => r.id));
  const evalRowIds = sortByUtf8(retained.filter((r) => r.split === 'EVAL').map((r) => r.id));
  const featureNames = sortByUtf8(eligible);

  const digestPayload = JSON.stringify({ trainRowIds, evalRowIds, featureNames });
  const datasetDigest = sha256Hex(digestPayload);

  // 3. Trial selection
  let selectedTrialId = null;
  const reasonCodes = [];

  if (v.trials.length > v.numTrialsLimit) {
    reasonCodes.push('TRIAL_LIMIT_EXCEEDED');
  } else {
    const eligibleTrials = v.trials.filter((t) => t.status === 'SUCCEEDED' && Number.isFinite(t.evalMetric));
    if (eligibleTrials.length === 0) {
      reasonCodes.push('NO_SUCCESSFUL_TRIAL');
    } else {
      eligibleTrials.sort((a, b) => {
        if (b.evalMetric !== a.evalMetric) return b.evalMetric - a.evalMetric;
        return a.trialId - b.trialId;
      });
      selectedTrialId = eligibleTrials[0].trialId;
    }
  }

  return {
    runId: v.runId,
    selectedTrialId,
    trainRowIds,
    evalRowIds,
    featureNames,
    datasetDigest,
    reasonCodes,
  };
}

function handleSelect(req, res) {
  const body = req.body;
  const validated = validateSelectBody(body);

  if (!validated.ok) {
    const response = {
      runId: isNonEmptyString(body.runId, 128) ? body.runId : null,
      selectedTrialId: null,
      trainRowIds: [],
      evalRowIds: [],
      featureNames: [],
      datasetDigest: null,
      reasonCodes: ['INVALID_INPUT'],
    };
    // Only persist/replay-check if we at least have a usable runId key.
    if (isNonEmptyString(body.runId, 128)) {
      const existing = store.get(body.runId);
      if (existing) {
        if (deepEqual(existing.requestBody, body)) {
          return res.status(200).json(existing.response);
        }
        return res.status(409).json({ error: 'RUN_ID_CONFLICT' });
      }
      store.set(body.runId, { requestBody: body, response });
    }
    return res.status(200).json(response);
  }

  const existing = store.get(validated.runId);
  if (existing) {
    if (deepEqual(existing.requestBody, body)) {
      return res.status(200).json(existing.response);
    }
    return res.status(409).json({ error: 'RUN_ID_CONFLICT' });
  }

  const response = processSelect(validated);
  store.set(validated.runId, { requestBody: body, response });
  return res.status(200).json(response);
}

// ---------------------------------------------------------------------------
// EVALUATE phase
// ---------------------------------------------------------------------------

const HEX64_RE = /^[0-9a-f]{64}$/;

function validateEvaluateBody(body) {
  if (!isNonEmptyString(body.runId, 128)) return { ok: false };
  if (!isSafeNonNegInt(body.selectedTrialId)) return { ok: false };
  if (!isNonEmptyString(body.datasetDigest) || !HEX64_RE.test(body.datasetDigest)) return { ok: false };
  if (!isFiniteNum(body.metricFloor) || body.metricFloor < 0 || body.metricFloor > 1) return { ok: false };
  if (!isPlainObject(body.requiredSlices)) return { ok: false };
  for (const [, floor] of Object.entries(body.requiredSlices)) {
    if (!isFiniteNum(floor) || floor < 0 || floor > 1) return { ok: false };
  }
  if (!Array.isArray(body.rows)) return { ok: false };
  if (!isSafeNonNegInt(body.bytesProcessed)) return { ok: false };
  if (!isSafeNonNegInt(body.maxBytes)) return { ok: false };
  return { ok: true };
}

function isValidTestRow(r) {
  if (!isPlainObject(r)) return false;
  if (r.label !== 0 && r.label !== 1) return false;
  if (r.prediction !== 0 && r.prediction !== 1) return false;
  if (!isNonEmptyString(r.slice)) return false;
  return true;
}

function handleEvaluate(req, res) {
  const body = req.body;
  const check = validateEvaluateBody(body);

  const bytesEcho = isSafeNonNegInt(body.bytesProcessed) ? body.bytesProcessed : 0;

  if (!check.ok) {
    return res.status(200).json({
      runId: isNonEmptyString(body.runId, 128) ? body.runId : null,
      selectedTrialId: isSafeNonNegInt(body.selectedTrialId) ? body.selectedTrialId : null,
      datasetDigest: isNonEmptyString(body.datasetDigest) ? body.datasetDigest : null,
      testMetric: null,
      criticalSlicePass: false,
      decision: 'reject',
      bytesProcessed: bytesEcho,
      reasonCodes: ['INVALID_INPUT'],
    });
  }

  const codes = [];

  // Lineage check
  const stored = store.get(body.runId);
  const lineageOk =
    !!stored &&
    stored.response.selectedTrialId !== null &&
    stored.response.selectedTrialId === body.selectedTrialId &&
    stored.response.datasetDigest === body.datasetDigest;
  if (!lineageOk) codes.push('INVALID_LINEAGE');

  // Row validity
  const rows = body.rows;
  const allRowsValid = rows.every(isValidTestRow);
  if (rows.length > 0 && !allRowsValid) codes.push('INVALID_TEST_ROW');

  let testMetric = null;
  let aggregatePass = false;
  let allRequiredSlicesOk = Object.keys(body.requiredSlices).length === 0;

  if (rows.length > 0 && allRowsValid) {
    const total = rows.length;
    const correct = rows.filter((r) => r.prediction === r.label).length;
    const aggregate = round12(correct / total);
    testMetric = aggregate;
    if (aggregate < body.metricFloor) codes.push('AGGREGATE_FLOOR');
    aggregatePass = aggregate >= body.metricFloor;

    allRequiredSlicesOk = true;
    for (const [name, floor] of Object.entries(body.requiredSlices)) {
      const sliceRows = rows.filter((r) => r.slice === name);
      if (sliceRows.length === 0) {
        codes.push(`MISSING_SLICE:${name}`);
        allRequiredSlicesOk = false;
        continue;
      }
      const sliceCorrect = sliceRows.filter((r) => r.prediction === r.label).length;
      const sliceAcc = round12(sliceCorrect / sliceRows.length);
      if (sliceAcc < floor) {
        codes.push(`SLICE_FLOOR:${name}`);
        allRequiredSlicesOk = false;
      }
    }
  }

  const bytesOk = body.bytesProcessed <= body.maxBytes;
  if (!bytesOk) codes.push('BYTE_LIMIT');

  const finalCodes = dedupeSortedCodes(codes);

  const criticalSlicePass = !finalCodes.some(
    (c) => c === 'INVALID_INPUT' || c === 'INVALID_LINEAGE' || c === 'INVALID_TEST_ROW' || c.startsWith('MISSING_SLICE:') || c.startsWith('SLICE_FLOOR:')
  );

  const decision =
    lineageOk && rows.length > 0 && allRowsValid && aggregatePass && allRequiredSlicesOk && bytesOk
      ? 'admit'
      : 'reject';

  return res.status(200).json({
    runId: body.runId,
    selectedTrialId: body.selectedTrialId,
    datasetDigest: body.datasetDigest,
    testMetric,
    criticalSlicePass,
    decision,
    bytesProcessed: body.bytesProcessed,
    reasonCodes: finalCodes,
  });
}

// ---------------------------------------------------------------------------
// Route
// ---------------------------------------------------------------------------

app.post('/bqml', (req, res) => {
  const body = req.body;
  if (!isPlainObject(body) || (body.phase !== 'select' && body.phase !== 'evaluate')) {
    return res.status(400).json({ error: 'INVALID_INPUT' });
  }
  if (body.phase === 'select') return handleSelect(req, res);
  return handleEvaluate(req, res);
});

app.get('/', (req, res) => res.send('bqml experiment gate is running'));

// Malformed JSON body -> treat as unknown/invalid
app.use((err, req, res, next) => {
  if (err) return res.status(400).json({ error: 'INVALID_INPUT' });
  next();
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`bqml service listening on port ${PORT}`));

module.exports = app;
