import { chromium } from 'playwright';
import * as path from 'path';
import * as fs from 'fs';

const BASE_URL = 'http://localhost:8000';
const ARTIFACTS_DIR = process.env.ARTIFACT_DIR || '/Users/sgtpooki/.gemini/antigravity-cli/brain/d3bd2394-3200-4602-b9d9-757d80615122';

async function run() {
  console.log('--- Starting Multi-Tenant UI Verification ---');

  // 1. Create a tournament via API
  const tourneyId = `verify_ui_${Date.now()}`;
  const initPayload = {
    title: 'Multi-Tenant Divergence Arena',
    prompt: 'Evaluate technical depth vs narrative voice in systems architecture.',
    task_type: 'text',
    candidates: [
      { id: 'c1', label: 'Candidate 1 (Claude Favorite)', content: 'Focuses on elegant design patterns and expressive semantics.' },
      { id: 'c2', label: 'Candidate 2 (Homelab Favorite)', content: 'Focuses on bare-metal IOPS, pagecache tuning, and CPU pinning.' },
      { id: 'c3', label: 'Candidate 3 (Middle Ground)', content: 'Balances microservices with pragmatism and metric pipelines.' },
      { id: 'c4', label: 'Candidate 4 (Baseline)', content: 'Standard monolith deployment with PostgreSQL.' },
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

  // 2. Cast divergent votes
  // claude-code heavily favors c1
  for (let i = 0; i < 3; i++) {
    await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id_a: 'c1',
        id_b: 'c2',
        winner: 'a',
        voter: 'claude-code',
        notes: 'Superior architectural abstraction',
      }),
    });
  }

  // omp:homelab-default heavily favors c2
  for (let i = 0; i < 3; i++) {
    await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id_a: 'c1',
        id_b: 'c2',
        winner: 'b',
        voter: 'omp:homelab-default',
        notes: 'Demonstrated deterministic memory bounds and raw throughput',
      }),
    });
  }

  // Launch Playwright
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 850 } });

  console.log(`Navigating to arena: ${BASE_URL}/?t=${tourney.id}`);
  await page.goto(`${BASE_URL}/?t=${tourney.id}`);
  await page.waitForSelector('#convergence-badge');

  // Verify convergence badge
  const badgeText = await page.locator('#convergence-badge').innerText();
  console.log(`Convergence indicator text: ${badgeText}`);

  const matchupScreenshot = path.join(ARTIFACTS_DIR, '08_convergence_matchup_view.png');
  await page.screenshot({ path: matchupScreenshot });
  console.log(`Saved screenshot: ${matchupScreenshot}`);

  // Switch to Leaderboard tab
  await page.click('#nav-leaderboard');
  await page.waitForSelector('#evaluator-filter');

  // Verify evaluator dropdown has both voters
  const selectOptions = await page.$$eval('#evaluator-filter option', (options) =>
    options.map((opt) => opt.value)
  );
  console.log(`Evaluator dropdown options: ${selectOptions.join(', ')}`);
  if (!selectOptions.includes('claude-code') || !selectOptions.includes('omp:homelab-default')) {
    throw new Error('Missing voters in evaluator filter!');
  }

  // Capture Pooled Leaderboard
  const pooledScreenshot = path.join(ARTIFACTS_DIR, '09_pooled_leaderboard.png');
  await page.screenshot({ path: pooledScreenshot });
  console.log(`Saved screenshot: ${pooledScreenshot}`);

  // Filter by claude-code
  console.log('Selecting evaluator: claude-code');
  await page.selectOption('#evaluator-filter', 'claude-code');
  await page.waitForTimeout(500);

  const claudeScreenshot = path.join(ARTIFACTS_DIR, '10_claude_sliced_leaderboard.png');
  await page.screenshot({ path: claudeScreenshot });
  console.log(`Saved screenshot: ${claudeScreenshot}`);

  // Filter by omp:homelab-default
  console.log('Selecting evaluator: omp:homelab-default');
  await page.selectOption('#evaluator-filter', 'omp:homelab-default');
  await page.waitForTimeout(500);

  const homelabScreenshot = path.join(ARTIFACTS_DIR, '11_homelab_sliced_leaderboard.png');
  await page.screenshot({ path: homelabScreenshot });
  console.log(`Saved screenshot: ${homelabScreenshot}`);

  // Switch back to pooled, accept winner c1
  await page.selectOption('#evaluator-filter', 'pooled');
  await page.waitForTimeout(300);

  console.log('Accepting candidate c1 as tournament winner');
  const acceptBtn = page.locator('button:has-text("Accept")').first();
  await acceptBtn.click();
  await page.waitForSelector('text=Accepted Winner');

  // Verify "+ Continue Stage" button appears
  const continueBtn = page.locator('button:has-text("+ Continue Stage")');
  await continueBtn.waitFor({ state: 'visible' });

  // Open Chain Stage Modal
  console.log('Opening Continue Stage modal...');
  await continueBtn.click();
  await page.waitForSelector('#chain-modal[style*="display: flex"]');

  // Fill in new stage details
  await page.fill('#chain-title', 'Stage 2: Implementation Details');
  await page.fill('#chain-prompt', 'Elaborate concrete storage subsystems based on the chosen design.');
  await page.fill(
    '#chain-candidates',
    'Implementation Draft A: NVMe-oF with Democratic CSI\n---\nImplementation Draft B: Local ZFS replica daemon'
  );

  const modalScreenshot = path.join(ARTIFACTS_DIR, '12_chain_stage_modal.png');
  await page.screenshot({ path: modalScreenshot });
  console.log(`Saved screenshot: ${modalScreenshot}`);

  // Submit and create child stage
  console.log('Submitting child stage...');
  await page.click('#btn-create-chain');
  await page.waitForTimeout(1000);

  // Verify navigation to new chained stage
  const currentTitle = await page.locator('#prompt-text').innerText();
  console.log(`Navigated to chained stage prompt: ${currentTitle}`);

  const chainedScreenshot = path.join(ARTIFACTS_DIR, '13_chained_child_stage.png');
  await page.screenshot({ path: chainedScreenshot });
  console.log(`Saved screenshot: ${chainedScreenshot}`);

  await browser.close();
  console.log('--- Multi-Tenant UI Verification Completed Successfully ---');
}

run().catch((err) => {
  console.error('Test error:', err);
  process.exit(1);
});
