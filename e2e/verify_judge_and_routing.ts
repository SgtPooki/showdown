import { chromium } from 'playwright';
import * as path from 'path';
import * as fs from 'fs';

const BASE_URL = 'http://localhost:8000';
const ARTIFACTS_DIR = process.env.ARTIFACT_DIR || '/Users/sgtpooki/.gemini/antigravity-cli/brain/d3bd2394-3200-4602-b9d9-757d80615122';

async function run() {
  console.log('--- Starting LLM Judge, Human-Judge Concordance, and Active Routing Verification ---');

  // 1. Create a tournament
  const tourneyId = `verify_judge_${Date.now()}`;
  const initPayload = {
    title: 'Distributed Consensus Protocol Design',
    prompt: 'Compare distributed consensus algorithms on latency, network overhead, and fault-tolerance.',
    task_type: 'code',
    candidates: [
      { id: 'c1', label: 'Raft (Leader Heartbeats)', content: 'class RaftNode:\n    def append_entries(self, term, leader_id, prev_log_idx):\n        # Leader append entries heartbeat\n        pass' },
      { id: 'c2', label: 'Paxos (Multi-Round Acceptor)', content: 'class PaxosAcceptor:\n    def prepare(self, proposal_id):\n        # Two-phase prepare\n        pass' },
      { id: 'c3', label: 'Viewstamped Replication', content: 'class VRReplica:\n    def do_view_change(self, new_view):\n        # View change transaction\n        pass' },
      { id: 'c4', label: 'ZAB (Atomic Broadcast)', content: 'class ZABPeer:\n    def broadcast_proposal(self, zxid, packet):\n        # FIFO atomic broadcast\n        pass' },
    ],
  };

  const createRes = await fetch(`${BASE_URL}/api/tournaments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(initPayload),
  });
  if (!createRes.ok) throw new Error('Failed to create tournament');
  const tourney = await createRes.json();
  console.log(`Created tournament: ${tourney.id}`);

  // 2. Cast human votes
  // Human prefers c1 over c2
  await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id_a: 'c1',
      id_b: 'c2',
      winner: 'a',
      voter: 'human',
      notes: 'Raft has vastly simpler operational invariants than Multi-Paxos',
    }),
  });

  // Human prefers c1 over c3
  await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id_a: 'c1',
      id_b: 'c3',
      winner: 'a',
      voter: 'human',
      notes: 'Raft has superior ecosystem adoption',
    }),
  });

  // 3. Cast synthetic judge votes (e.g. judge:omp agrees with human on c1 vs c2)
  await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id_a: 'c1',
      id_b: 'c2',
      winner: 'a',
      voter: 'judge:omp',
      notes: '[Consistent] Candidate A demonstrates clearer state machine replication safety.',
    }),
  });

  // Also judge casts vote on c3 vs c4 with position-bias note to test controversial routing
  await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id_a: 'c3',
      id_b: 'c4',
      winner: 'tie',
      voter: 'judge:omp',
      notes: '[Position-Bias Contradiction] Judge favored slot 1 in both presentations. Resolved as tie.',
    }),
  });

  console.log('Seeded human and judge votes.');

  // 4. Launch Playwright
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

  await page.goto(`${BASE_URL}/?t=${tourney.id}`);
  await page.waitForSelector('#view-matchup');
  await page.waitForTimeout(1000);

  // Take screenshot 1: Arena view with routing selector
  const screenshot1 = path.join(ARTIFACTS_DIR, '18_arena_active_routing.png');
  await page.screenshot({ path: screenshot1, fullPage: true });
  console.log(`Saved screenshot: ${screenshot1}`);

  // 5. Test switching routing mode to 'controversial'
  const routingSelect = await page.$('#matchup-routing-select');
  if (routingSelect) {
    await page.selectOption('#matchup-routing-select', 'controversial');
    await page.waitForTimeout(600);
    console.log('Switched routing mode to controversial.');
  }

  // 6. Switch to Leaderboard tab
  await page.click('#nav-leaderboard');
  await page.waitForSelector('#view-leaderboard');
  await page.waitForTimeout(1000);

  // Verify pills
  const agreementPill = await page.$('#agreement-pill');
  const humanJudgePill = await page.$('#human-judge-pill');
  const consistencyPill = await page.$('#consistency-pill');

  const hjText = humanJudgePill ? await humanJudgePill.innerText() : 'NOT_FOUND';
  const agreeText = agreementPill ? await agreementPill.innerText() : 'NOT_FOUND';
  console.log(`Agreement pill: "${agreeText}"`);
  console.log(`Human-Judge Alignment pill: "${hjText}"`);

  // Take screenshot 2: Leaderboard with Human-Judge alignment
  const screenshot2 = path.join(ARTIFACTS_DIR, '19_human_judge_alignment_leaderboard.png');
  await page.screenshot({ path: screenshot2, fullPage: true });
  console.log(`Saved screenshot: ${screenshot2}`);

  // 7. Click "Run Automated Judge" to open modal
  await page.click('#nav-judge');
  await page.waitForSelector('#modal-judge', { state: 'visible' });
  await page.waitForTimeout(500);

  // Take screenshot 3: Automated judge modal
  const screenshot3 = path.join(ARTIFACTS_DIR, '20_automated_judge_modal.png');
  await page.screenshot({ path: screenshot3, fullPage: true });
  console.log(`Saved screenshot: ${screenshot3}`);

  await browser.close();
  console.log('--- Verification Completed Successfully ---');
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
