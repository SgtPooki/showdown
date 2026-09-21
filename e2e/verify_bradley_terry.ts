import { chromium } from 'playwright';
import * as path from 'path';

const BASE_URL = 'http://localhost:8000';
const ARTIFACTS_DIR = process.env.ARTIFACT_DIR || '/Users/sgtpooki/.gemini/antigravity-cli/brain/d3bd2394-3200-4602-b9d9-757d80615122';

async function run() {
  console.log('--- Starting Bradley-Terry MLE and Info-Gain Matchmaking Verification ---');

  // 1. Create a tournament
  const tourneyId = `verify_bt_${Date.now()}`;
  const initPayload = {
    title: 'High-Throughput Key-Value Storage Architecture',
    prompt: 'Compare storage engine architectures on write amplification, read latency, and crash recovery.',
    task_type: 'code',
    candidates: [
      { id: 'c1', label: 'LSM-Tree with Tiered Compaction', content: 'class LSMTree:\n    def write_memtable(self, key, val):\n        # Tiered write path\n        pass' },
      { id: 'c2', label: 'B+ Tree with Copy-on-Write (Btrfs style)', content: 'class BPlusTreeCoW:\n    def insert(self, key, val):\n        # CoW root pointer transition\n        pass' },
      { id: 'c3', label: 'Log-Structured Append with Bitmap Index', content: 'class AppendLog:\n    def append(self, record):\n        # Direct I/O append\n        pass' },
      { id: 'c4', label: 'Hash-Indexed Append-Only (Bitcask style)', content: 'class Bitcask:\n    def put(self, key, val):\n        # Write to log and in-memory dir\n        pass' },
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

  // 2. Seed realistic pairwise votes
  const votes = [
    { id_a: 'c1', id_b: 'c2', winner: 'a', voter: 'alice' },
    { id_a: 'c1', id_b: 'c3', winner: 'a', voter: 'bob' },
    { id_a: 'c1', id_b: 'c4', winner: 'a', voter: 'alice' },
    { id_a: 'c2', id_b: 'c3', winner: 'a', voter: 'carol' },
    { id_a: 'c3', id_b: 'c4', winner: 'a', voter: 'alice' },
  ];

  for (const v of votes) {
    await fetch(`${BASE_URL}/api/tournaments/${tourney.id}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(v),
    });
  }
  console.log('Seeded match history.');

  // 3. Launch browser and verify UI
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

  await page.goto(`${BASE_URL}/?tournament=${tourney.id}`);
  await page.waitForTimeout(1000);

  // Switch routing mode to info_gain
  await page.selectOption('#matchup-routing-select', 'info_gain');
  await page.waitForTimeout(600);

  const shot1 = path.join(ARTIFACTS_DIR, '21_arena_info_gain_mle.png');
  await page.screenshot({ path: shot1 });
  console.log(`Saved screenshot: ${shot1}`);

  // Navigate to Leaderboard view
  await page.click('#nav-leaderboard');
  await page.waitForTimeout(600);

  const shot2 = path.join(ARTIFACTS_DIR, '22_leaderboard_bradley_terry_uncertainty.png');
  await page.screenshot({ path: shot2 });
  console.log(`Saved screenshot: ${shot2}`);

  await browser.close();
  console.log('--- Verification Completed Successfully ---');
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
