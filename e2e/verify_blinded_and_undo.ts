import { chromium } from 'playwright';
import * as path from 'path';
import * as fs from 'fs';

const BASE_URL = 'http://localhost:8000';
const ARTIFACTS_DIR = process.env.ARTIFACT_DIR || '/Users/sgtpooki/.gemini/antigravity-cli/brain/d3bd2394-3200-4602-b9d9-757d80615122';

async function run() {
  console.log('--- Starting Blinded Mode, Undo, and Consistency Verification ---');

  // 1. Create a tournament via API
  const tourneyId = `verify_blind_${Date.now()}`;
  const initPayload = {
    title: 'Model Alignment Benchmark Arena',
    prompt: 'Evaluate reasoning fidelity and stylistic clarity on distributed systems.',
    task_type: 'text',
    candidates: [
      { id: 'c1', label: 'Candidate 1 (Claude 3.7)', content: 'Decoupled event streams with backpressure propagation and zero-copy ring buffers.' },
      { id: 'c2', label: 'Candidate 2 (Codex DeepSeek)', content: 'Kernel bypass socket layer using eBPF filtering and SIMD batch deserialization.' },
      { id: 'c3', label: 'Candidate 3 (Homelab 32B)', content: 'LXC containerized agent pipeline coordinated by transactional SQLite state machines.' },
      { id: 'c4', label: 'Candidate 4 (Baseline Monolith)', content: 'Standard synchronous blocking REST service with thread-pool scaling.' },
    ],
  };

  const createRes = await fetch(`${BASE_URL}/api/tournaments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(initPayload),
  });
  if (!createRes.ok) throw new Error('Failed to create tournament');
  const tourney = await createRes.json();
  console.log(`Created test tournament: ${tourney.id}`);

  // 2. Cast some votes from evaluators with both shared agreement and self-consistency
  // Judge 1 (Alice): votes c1 over c2 twice (100% self-consistent)
  await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id_a: 'c1', id_b: 'c2', winner: 'a', voter: 'evaluator_alice', notes: 'Cleaner backpressure API' }),
  });
  await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id_a: 'c1', id_b: 'c2', winner: 'a', voter: 'evaluator_alice', notes: 'Confirmed backpressure superiority' }),
  });

  // Judge 2 (Bob): votes c1 over c2 once (agrees with Alice -> 100% agreement on c1-c2)
  await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id_a: 'c1', id_b: 'c2', winner: 'a', voter: 'evaluator_bob', notes: 'Agreed on decoupling' }),
  });

  // Judge 1 (Alice): votes c2 over c3
  await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id_a: 'c2', id_b: 'c3', winner: 'a', voter: 'evaluator_alice', notes: 'eBPF is faster than LXC' }),
  });

  // 3. Launch browser
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  console.log(`Navigating to ${BASE_URL}/?t=${tourney.id}`);
  await page.goto(`${BASE_URL}/?t=${tourney.id}`);
  await page.waitForSelector('#meta-a', { state: 'visible' });
  await page.waitForTimeout(600);

  // Capture Screenshot 14: Blind Mode OFF
  const shot14 = path.join(ARTIFACTS_DIR, '14_blind_mode_off.png');
  await page.screenshot({ path: shot14, fullPage: true });
  console.log(`Saved screenshot: ${shot14}`);

  // Toggle Blind Mode ON
  console.log('Toggling Blind Mode ON...');
  await page.click('#btn-blind-toggle');
  await page.waitForTimeout(400);

  // Capture Screenshot 15: Blind Mode ON (Masked identities)
  const shot15 = path.join(ARTIFACTS_DIR, '15_blind_mode_on.png');
  await page.screenshot({ path: shot15, fullPage: true });
  console.log(`Saved screenshot: ${shot15}`);

  // Submit a vote in the UI
  console.log('Submitting a vote via button...');
  await page.click('.vote-btn-a');
  await page.waitForTimeout(600);

  // Now click Undo button
  console.log('Clicking Undo button...');
  await page.click('#btn-undo');
  await page.waitForTimeout(600);

  // Capture Screenshot 16: Vote Undone Toast and Recalibrated State
  const shot16 = path.join(ARTIFACTS_DIR, '16_vote_undone.png');
  await page.screenshot({ path: shot16, fullPage: true });
  console.log(`Saved screenshot: ${shot16}`);

  // Switch to Leaderboard tab
  console.log('Switching to Leaderboard view...');
  await page.click('#nav-leaderboard');
  await page.waitForSelector('#leaderboard-body tr', { state: 'visible' });
  await page.waitForTimeout(800);

  // Capture Screenshot 17: Leaderboard with Agreement and Consistency pills
  const shot17 = path.join(ARTIFACTS_DIR, '17_agreement_and_consistency_leaderboard.png');
  await page.screenshot({ path: shot17, fullPage: true });
  console.log(`Saved screenshot: ${shot17}`);

  await browser.close();
  console.log('Verification completed successfully!');
}

run().catch((err) => {
  console.error('Test run failed:', err);
  process.exit(1);
});
