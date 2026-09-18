import { chromium } from 'playwright';
import { execFileSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const BASE_URL = process.env.SHOWDOWN_URL || 'http://localhost:8000';
const ARTIFACTS_DIR = path.resolve(__dirname, '../artifacts/logo');

function ensureDir(dir: string) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

export interface JudgeDecision {
  winner: 'a' | 'b' | 'both_bad' | 'tie';
  notes: string;
  judge: string;
}

// 1. Query Local OMP Homelab Model for visual & vector critique
export function queryHomelabVisualJudge(svgA: string, svgB: string, promptText: string): JudgeDecision {
  const query = `You are a principal brand designer and design systems architect evaluating two candidate inline SVG marks.
Task: "${promptText}"

Candidate Mark A (SVG):
${svgA}

Candidate Mark B (SVG):
${svgB}

Evaluate which mark is superior based on:
1. Geometric balance, clarity, and visual impact at small/icon scale.
2. Aesthetic appeal and modern brand resonance.
3. SVG structure (clean paths, elegant strokes, contrast).

Output STRICTLY a JSON object with keys "winner" ('a', 'b', 'both_bad', 'tie') and "notes" (1 concise sentence explaining the visual rationale).
Example: {"winner": "a", "notes": "Candidate A has superior geometric contrast and scales better as an app icon."}`;

  try {
    const raw = execFileSync('omp', ['-p', '--model=homelab-default', '--no-session', '--no-tools', query], {
      encoding: 'utf-8',
      timeout: 30000,
    }).trim();

    const clean = raw.replace(/^Working\.\.\.\s*/i, '').replace(/^```[a-z]*\n?/i, '').replace(/\n?```$/i, '').trim();
    const match = clean.match(/\{[\s\S]*\}/);
    if (match) {
      const parsed = JSON.parse(match[0]);
      return { winner: parsed.winner || 'a', notes: parsed.notes || 'Judged by local homelab-default', judge: 'omp:homelab-default' };
    }
    const parsed = JSON.parse(clean);
    return { winner: parsed.winner || 'a', notes: parsed.notes || 'Judged by local homelab-default', judge: 'omp:homelab-default' };
  } catch (err) {
    console.warn(`[Homelab OMP Fallback]:`, err);
    return {
      winner: svgA.length < svgB.length ? 'a' : 'b',
      notes: 'Leaner vector structure favored (fallback heuristic).',
      judge: 'omp:fallback'
    };
  }
}

// 2. Query Claude Judge for design critique
export function queryClaudeVisualJudge(svgA: string, svgB: string, promptText: string): JudgeDecision {
  const query = `You are an elite creative director evaluating two candidate vector SVG logos.
Goal: "${promptText}"

Candidate A:
${svgA}

Candidate B:
${svgB}

Choose the winning logo ('a', 'b', 'both_bad', or 'tie') based on iconic identity, elegance, and visual execution.
Output STRICTLY a JSON object with keys "winner" and "notes" (1 sentence).
Example: {"winner": "b", "notes": "Candidate B has a much stronger vector presence and distinctive kinetic energy."}`;

  try {
    const raw = execFileSync('claude', ['-p', query], { encoding: 'utf-8', timeout: 30000 }).trim();
    const clean = raw.replace(/^```[a-z]*\n?/i, '').replace(/\n?```$/i, '').trim();
    const match = clean.match(/\{[\s\S]*\}/);
    if (match) {
      const parsed = JSON.parse(match[0]);
      return { winner: parsed.winner || 'a', notes: parsed.notes || 'Judged by Claude Code', judge: 'claude-code' };
    }
    const parsed = JSON.parse(clean);
    return { winner: parsed.winner || 'a', notes: parsed.notes || 'Judged by Claude Code', judge: 'claude-code' };
  } catch (err) {
    console.warn(`[Claude Judge Fallback]:`, err);
    return {
      winner: 'a',
      notes: 'Vector balance favored (Claude fallback).',
      judge: 'claude:fallback'
    };
  }
}

// Initial SVG Generation Pool (Generation 1)
const INITIAL_LOGOS = [
  {
    label: "Hexagonal Flux Core",
    content: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <defs>
    <linearGradient id="gradHex" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#06B6D4" />
      <stop offset="100%" stop-color="#8B5CF6" />
    </linearGradient>
  </defs>
  <rect width="200" height="200" rx="40" fill="#090D16" />
  <polygon points="100,30 160,65 160,135 100,170 40,135 40,65" fill="none" stroke="url(#gradHex)" stroke-width="8" stroke-linejoin="round" />
  <circle cx="100" cy="100" r="28" fill="#06B6D4" opacity="0.9" />
  <circle cx="100" cy="100" r="14" fill="#8B5CF6" />
  <line x1="100" y1="30" x2="100" y2="72" stroke="#06B6D4" stroke-width="4" stroke-linecap="round" />
  <line x1="160" y1="135" x2="124" y2="114" stroke="#8B5CF6" stroke-width="4" stroke-linecap="round" />
  <line x1="40" y1="135" x2="76" y2="114" stroke="#06B6D4" stroke-width="4" stroke-linecap="round" />
</svg>`
  },
  {
    label: "Vector Monogram N",
    content: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <defs>
    <linearGradient id="gradN1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8" />
      <stop offset="100%" stop-color="#2563EB" />
    </linearGradient>
    <linearGradient id="gradN2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#8B5CF6" />
      <stop offset="100%" stop-color="#EC4899" />
    </linearGradient>
  </defs>
  <rect width="200" height="200" rx="40" fill="#0B0F19" />
  <path d="M 50 150 L 50 50 L 80 50 L 120 120 L 120 50 L 150 50 L 150 150 L 120 150 L 80 80 L 80 150 Z" fill="url(#gradN1)" />
  <polygon points="80,80 120,150 150,150" fill="url(#gradN2)" opacity="0.85" />
</svg>`
  },
  {
    label: "Orbital Tri-Ring Consensus",
    content: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <defs>
    <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#00F0FF" />
      <stop offset="100%" stop-color="#00F0FF" stop-opacity="0" />
    </radialGradient>
  </defs>
  <rect width="200" height="200" rx="40" fill="#0A0E1A" />
  <ellipse cx="100" cy="100" rx="65" ry="24" fill="none" stroke="#06B6D4" stroke-width="5" transform="rotate(30 100 100)" />
  <ellipse cx="100" cy="100" rx="65" ry="24" fill="none" stroke="#8B5CF6" stroke-width="5" transform="rotate(90 100 100)" />
  <ellipse cx="100" cy="100" rx="65" ry="24" fill="none" stroke="#F43F5E" stroke-width="5" transform="rotate(150 100 100)" />
  <circle cx="100" cy="100" r="30" fill="url(#nodeGlow)" />
  <circle cx="100" cy="100" r="12" fill="#FFFFFF" />
</svg>`
  },
  {
    label: "Stacked Chamfer Prism",
    content: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <defs>
    <linearGradient id="prismGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#6366F1" />
      <stop offset="100%" stop-color="#A855F7" />
    </linearGradient>
  </defs>
  <rect width="200" height="200" rx="40" fill="#080C14" />
  <polygon points="100,40 155,70 100,100 45,70" fill="url(#prismGrad)" opacity="0.9" />
  <polygon points="100,80 155,110 100,140 45,110" fill="url(#prismGrad)" opacity="0.65" />
  <polygon points="100,120 155,150 100,180 45,150" fill="url(#prismGrad)" opacity="0.4" />
</svg>`
  },
  {
    label: "Velocity Wave Crest",
    content: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <defs>
    <linearGradient id="waveGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#10B981" />
      <stop offset="100%" stop-color="#06B6D4" />
    </linearGradient>
  </defs>
  <rect width="200" height="200" rx="40" fill="#090D16" />
  <path d="M 40 130 C 60 70 80 50 110 60 C 140 70 130 140 160 110" fill="none" stroke="url(#waveGrad)" stroke-width="12" stroke-linecap="round" />
  <circle cx="160" cy="110" r="10" fill="#10B981" />
  <circle cx="40" cy="130" r="6" fill="#06B6D4" />
</svg>`
  },
  {
    label: "Cyber Shield Telemetry",
    content: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <rect width="200" height="200" rx="40" fill="#0A0E1A" />
  <path d="M 100 35 L 155 60 L 155 110 C 155 145 100 170 100 170 C 100 170 45 145 45 110 L 45 60 Z" fill="none" stroke="#6366F1" stroke-width="7" stroke-linejoin="round" />
  <polygon points="100,70 130,95 100,135 70,95" fill="#38BDF8" opacity="0.8" />
  <circle cx="100" cy="100" r="6" fill="#FFFFFF" />
</svg>`
  }
];

async function main() {
  ensureDir(ARTIFACTS_DIR);
  console.log('--- STARTING NOVASYNC LOGO ARENA SIMULATION ---');

  const tourneyId = `novasync_logo_${Date.now().toString().slice(-4)}`;
  const promptText = "Design an iconic vector brand mark for Novasync: distributed event broker and microsecond consensus engine. Clean, high-contrast inline SVG.";

  // 1. Create Tournament via API
  console.log(`Creating tournament '${tourneyId}' (task_type: svg)...`);
  const createRes = await fetch(`${BASE_URL}/api/tournaments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: tourneyId,
      title: "Novasync Brand Mark Arena",
      prompt: promptText,
      task_type: "svg",
      candidates: INITIAL_LOGOS.map((c, i) => ({
        id: `gen1_${i + 1}`,
        label: c.label,
        content: c.content,
        generation: 1,
        metadata: { source: "curated_gen1" }
      }))
    })
  });

  if (!createRes.ok) {
    throw new Error(`Failed to create tournament: ${await createRes.text()}`);
  }
  console.log(`Tournament ${tourneyId} created successfully.`);

  // 2. Launch Browser via Playwright
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 } });
  const page = await context.newPage();

  await page.goto(`${BASE_URL}/?t=${tourneyId}`);
  await page.waitForSelector('#view-matchup', { timeout: 10000 });
  await page.waitForTimeout(1000);

  // Take initial matchup screenshot
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '01_initial_svg_matchup.png') });
  console.log('Saved 01_initial_svg_matchup.png');

  // 3. Run Generation 1 Tournament Matchups (8 rounds)
  console.log('\n--- EVALUATING GENERATION 1 LOGOS ---');
  for (let round = 1; round <= 8; round++) {
    // Get matchup data via API
    const matchRes = await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/matchup`);
    if (!matchRes.ok) break;
    const matchup = await matchRes.json();
    if (!matchup.candidate_a || !matchup.candidate_b) break;

    const candA = matchup.candidate_a;
    const candB = matchup.candidate_b;

    // Alternate judges
    let decision: JudgeDecision;
    if (round % 2 === 1) {
      console.log(`[Round ${round}] Judging with OMP Homelab-Default: "${candA.label}" vs "${candB.label}"`);
      decision = queryHomelabVisualJudge(candA.content, candB.content, promptText);
    } else {
      console.log(`[Round ${round}] Judging with Claude Code: "${candA.label}" vs "${candB.label}"`);
      decision = queryClaudeVisualJudge(candA.content, candB.content, promptText);
    }

    console.log(` -> Winner: [${decision.winner}] by ${decision.judge}. Note: "${decision.notes}"`);

    // Submit vote
    await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id_a: candA.id,
        id_b: candB.id,
        winner: decision.winner,
        voter: decision.judge,
        notes: decision.notes
      })
    });

    // On round 2 and 4, record triage feedback (e.g. favorite top mark, dislike clunky mark)
    if (round === 2) {
      await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/triage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidate_id: candA.id,
          status: "favorite",
          notes: "Exceptional contrast and scalable monogram energy."
        })
      });
      console.log(` [Triage] Favorited candidate ${candA.label}`);
    } else if (round === 4) {
      await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/triage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidate_id: candB.id,
          status: "disliked",
          notes: "Too generic and lacks the microsecond high-velocity tech aesthetic."
        })
      });
      console.log(` [Triage] Disliked candidate ${candB.label}`);
    }

    await page.reload();
    await page.waitForTimeout(500);
  }

  // Capture midway screenshot
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '02_midway_svg_matchup.png') });
  console.log('Saved 02_midway_svg_matchup.png');

  // 4. Trigger Evolution to Generation 2
  console.log('\n--- EVOLVING GENERATION 2 LOGO CANDIDATES ---');
  const evolveRes = await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/evolve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      count: 4,
      instructions: "Refine high-performing geometric monograms and concentric hexagonal pulse motifs. Use neon cyan (#00F0FF), deep violet (#8B5CF6), dark background, and sharp microsecond speed angles.",
      backend: "omp"
    })
  });

  if (!evolveRes.ok) {
    console.warn(`OMP evolve failed (${await evolveRes.text()}), falling back to auto backend...`);
    const fallbackRes = await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/evolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        count: 4,
        instructions: "Refine geometric monograms and hexagonal vector cores. Strict inline SVG with viewBox='0 0 200 200'.",
        backend: "auto"
      })
    });
    const fallbackData = await fallbackRes.json();
    console.log(`Evolved ${fallbackData.count} candidates via ${fallbackData.backend}`);
  } else {
    const evolveData = await evolveRes.json();
    console.log(`Evolved ${evolveData.count} candidates via ${evolveData.backend}!`);
    console.log(`Summary: ${evolveData.summary}`);
  }

  // 5. Run Generation 2 Matchups (6 rounds)
  console.log('\n--- EVALUATING EVOLVED CANDIDATES IN ARENA ---');
  for (let round = 1; round <= 6; round++) {
    const matchRes = await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/matchup`);
    if (!matchRes.ok) break;
    const matchup = await matchRes.json();
    if (!matchup.candidate_a || !matchup.candidate_b) break;

    const candA = matchup.candidate_a;
    const candB = matchup.candidate_b;

    const decision = round % 2 === 1
      ? queryHomelabVisualJudge(candA.content, candB.content, promptText)
      : queryClaudeVisualJudge(candA.content, candB.content, promptText);

    console.log(`[Gen2 Round ${round}] ${decision.judge}: "${candA.label}" vs "${candB.label}" -> Winner: ${decision.winner}`);

    await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id_a: candA.id,
        id_b: candB.id,
        winner: decision.winner,
        voter: decision.judge,
        notes: decision.notes
      })
    });
  }

  // 6. View Leaderboard, Accept Winner, and Export
  await page.click('#nav-leaderboard');
  await page.waitForTimeout(1000);

  // Fetch tournament status to determine top candidate
  const finalTourneyRes = await fetch(`${BASE_URL}/api/tournaments/${tourneyId}`);
  const finalTourney = await finalTourneyRes.json();

  const sortedCandidates = [...finalTourney.candidates].sort((a, b) => {
    const eloA = finalTourney.stats[a.id]?.elo || 1200;
    const eloB = finalTourney.stats[b.id]?.elo || 1200;
    return eloB - eloA;
  });

  const winner = sortedCandidates[0];
  const winnerStat = finalTourney.stats[winner.id];
  console.log(`\n🏆 CHAMPION LOGO: "${winner.label}" (${winner.id}) with Elo ${Math.round(winnerStat.elo)}`);

  // Accept winner
  await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/accept`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ candidate_id: winner.id })
  });

  // Save winning SVG file
  fs.writeFileSync(path.join(ARTIFACTS_DIR, 'novasync_winner.svg'), winner.content);
  console.log(`Saved winning SVG to ${path.join(ARTIFACTS_DIR, 'novasync_winner.svg')}`);

  // Reload page to reflect accepted status
  await page.reload();
  await page.click('#nav-leaderboard');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '03_final_logo_leaderboard.png'), fullPage: true });
  console.log('Saved 03_final_logo_leaderboard.png');

  // Export DPO dataset
  const dpoRes = await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/export/dpo`);
  const dpoData = await dpoRes.json();
  fs.writeFileSync(path.join(ARTIFACTS_DIR, 'novasync_preference_pairs.json'), JSON.stringify(dpoData, null, 2));
  console.log(`Saved DPO dataset with ${dpoData.length} pairs to novasync_preference_pairs.json`);

  await browser.close();
  console.log('--- NOVASYNC LOGO ARENA COMPLETE ---\n');
}

main().catch(err => {
  console.error('Logo simulation error:', err);
  process.exit(1);
});
