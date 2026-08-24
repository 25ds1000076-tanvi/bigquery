const assert = require('assert');
const http = require('http');
const app = require('./server.js');

const server = app.listen(0, run);

function post(port, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = http.request(
      { hostname: 'localhost', port, path: '/bqml', method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) } },
      (res) => {
        let chunks = '';
        res.on('data', (c) => (chunks += c));
        res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(chunks) }));
      }
    );
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function run() {
  const port = server.address().port;
  let failures = 0;
  const check = (name, cond) => {
    if (!cond) {
      console.error('FAIL:', name);
      failures++;
    } else {
      console.log('ok  :', name);
    }
  };

  // --- 1. Basic select: dedupe, feature eligibility, tie-break trials ---
  const selReq = {
    phase: 'select',
    runId: 'run-1',
    forbiddenFeatures: ['secret'],
    numTrialsLimit: 10,
    rows: [
      { id: 'b', entity: 'e1', eventTime: '2024-01-01T00:00:00Z', predictionTime: '2024-01-01T01:00:00Z', version: 1, split: 'TRAIN',
        features: { f1: { value: '1', availableAt: '2024-01-01T00:30:00Z' }, secret: { value: 'x', availableAt: '2024-01-01T00:00:00Z' } } },
      { id: 'a', entity: 'e1', eventTime: '2024-01-01T00:00:00Z', predictionTime: '2024-01-01T01:00:00Z', version: 1, split: 'TRAIN',
        features: { f1: { value: '2', availableAt: '2024-01-01T00:30:00Z' } } }, // same [entity,eventTime], same version -> smaller id 'a' wins
      { id: 'c', entity: 'e2', eventTime: '2024-01-02T00:00:00Z', predictionTime: '2024-01-02T01:00:00Z', version: 1, split: 'EVAL',
        features: { f1: { value: '3', availableAt: '2024-01-02T02:00:00Z' } } }, // availableAt AFTER predictionTime -> f1 not eligible everywhere
    ],
    trials: [
      { trialId: 9, status: 'SUCCEEDED', evalMetric: 0.9 },
      { trialId: 4, status: 'SUCCEEDED', evalMetric: 0.9 }, // tie -> smallest id (4) wins per spec example
      { trialId: 1, status: 'FAILED', evalMetric: 0.99 },
    ],
  };
  const r1 = await post(port, selReq);
  check('select status 200', r1.status === 200);
  check('dedupe keeps smaller id "a"', r1.body.trainRowIds.includes('a') && !r1.body.trainRowIds.includes('b'));
  check('f1 excluded (availableAt > predictionTime in row c)', !r1.body.featureNames.includes('f1'));
  check('secret excluded (forbidden)', !r1.body.featureNames.includes('secret'));
  check('tie-break picks trial 4', r1.body.selectedTrialId === 4);
  check('digest present', /^[0-9a-f]{64}$/.test(r1.body.datasetDigest));

  // --- 2. Replay identical -> same response ---
  const r1b = await post(port, selReq);
  check('replay identical returns same body', JSON.stringify(r1b.body) === JSON.stringify(r1.body));

  // --- 3. Conflict on reuse with different input ---
  const r1c = await post(port, { ...selReq, numTrialsLimit: 20 });
  check('conflict on reuse -> 409', r1c.status === 409 && r1c.body.error === 'RUN_ID_CONFLICT');

  // --- 4. TRIAL_LIMIT_EXCEEDED still computes dataset digest ---
  const r2 = await post(port, { ...selReq, runId: 'run-2', numTrialsLimit: 1 });
  check('trial limit exceeded code', r2.body.reasonCodes.includes('TRIAL_LIMIT_EXCEEDED'));
  check('selectedTrialId null on limit exceeded', r2.body.selectedTrialId === null);
  check('digest still computed on limit exceeded', r2.body.datasetDigest !== null);

  // --- 5. NO_SUCCESSFUL_TRIAL ---
  const r3 = await post(port, { ...selReq, runId: 'run-3', trials: [{ trialId: 1, status: 'FAILED', evalMetric: 0.5 }] });
  check('no successful trial code', r3.body.reasonCodes.includes('NO_SUCCESSFUL_TRIAL'));

  // --- 6. Malformed select -> INVALID_INPUT, null digest ---
  const r4 = await post(port, { phase: 'select', runId: 'run-4', rows: [] });
  check('malformed select -> INVALID_INPUT', r4.body.reasonCodes.includes('INVALID_INPUT'));
  check('malformed select -> null digest', r4.body.datasetDigest === null);

  // --- 7. Unknown phase -> 400 ---
  const r5 = await post(port, { phase: 'nope' });
  check('unknown phase -> 400', r5.status === 400 && r5.body.error === 'INVALID_INPUT');

  // --- 8. Evaluate happy path (admit) ---
  const evalReq = {
    phase: 'evaluate',
    runId: 'run-1',
    selectedTrialId: r1.body.selectedTrialId,
    datasetDigest: r1.body.datasetDigest,
    metricFloor: 0.7,
    requiredSlices: { critical: 0.5 },
    rows: [
      { label: 1, prediction: 1, slice: 'critical' },
      { label: 0, prediction: 0, slice: 'critical' },
      { label: 1, prediction: 1, slice: 'other' },
      { label: 1, prediction: 0, slice: 'other' },
    ],
    bytesProcessed: 500,
    maxBytes: 1000,
  };
  const r6 = await post(port, evalReq);
  check('evaluate admit', r6.body.decision === 'admit');
  check('evaluate testMetric = 0.75', r6.body.testMetric === 0.75);
  check('criticalSlicePass true', r6.body.criticalSlicePass === true);

  // --- 9. Evaluate bad lineage ---
  const r7 = await post(port, { ...evalReq, datasetDigest: '0'.repeat(64) });
  check('bad lineage -> INVALID_LINEAGE + reject', r7.body.reasonCodes.includes('INVALID_LINEAGE') && r7.body.decision === 'reject');
  check('bad lineage -> criticalSlicePass false', r7.body.criticalSlicePass === false);

  // --- 10. Byte limit exceeded but rows still fine -> reject, criticalSlicePass true ---
  const r8 = await post(port, { ...evalReq, bytesProcessed: 2000 });
  check('byte limit exceeded -> BYTE_LIMIT + reject', r8.body.reasonCodes.includes('BYTE_LIMIT') && r8.body.decision === 'reject');
  check('byte limit exceeded -> criticalSlicePass still true', r8.body.criticalSlicePass === true);

  // --- 11. Missing slice ---
  const r9 = await post(port, { ...evalReq, requiredSlices: { ghost: 0.5 } });
  check('missing slice code', r9.body.reasonCodes.includes('MISSING_SLICE:ghost'));
  check('missing slice -> reject', r9.body.decision === 'reject');

  // --- 12. Empty rows -> testMetric null, reject ---
  const r10 = await post(port, { ...evalReq, rows: [] });
  check('empty rows -> testMetric null', r10.body.testMetric === null);
  check('empty rows -> reject', r10.body.decision === 'reject');

  console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} TEST(S) FAILED`);
  server.close();
  process.exit(failures === 0 ? 0 : 1);
}
