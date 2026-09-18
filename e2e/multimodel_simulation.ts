import { chromium } from 'playwright';
import { execFileSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const BASE_URL = process.env.SHOWDOWN_URL || 'http://localhost:8000';
const ARTIFACTS_DIR = path.resolve(__dirname, '../artifacts/multimodel');

async function ensureDir(dir: string) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

export type JudgeBackend = 'homelab' | 'claude';

export interface JudgeDecision {
  winner: 'a' | 'b' | 'both_bad' | 'tie';
  notes: string;
  judge: string;
}

// 1. Query Local OMP Homelab Model
export function queryHomelabJudge(textA: string, textB: string, promptText: string): JudgeDecision {
  const query = `You are an impartial judge evaluating two options for the following prompt:
Prompt: "${promptText}"

Option A:
${textA}

Option B:
${textB}

Evaluate which option is superior based on wit, punchiness, and relevance.
Answer strictly in JSON format with keys "winner" ('a', 'b', 'both_bad', 'tie') and "notes" (1 concise sentence explanation).
Example: {"winner": "a", "notes": "Option A has much sharper comic timing and brevity."}`;

  try {
    const raw = execFileSync('omp', ['-p', '--model=homelab-default', '--no-session', '--no-tools', query], {
      encoding: 'utf-8',
      timeout: 30000,
    }).trim();

    // Clean preamble
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
      winner: textA.length < textB.length ? 'a' : 'b',
      notes: 'Brevity favored (fallback heuristic).',
      judge: 'omp:fallback'
    };
  }
}

// 2. Query Claude Judge
export function queryClaudeJudge(textA: string, textB: string, promptText: string): JudgeDecision {
  const query = `You are a discerning comedic and editorial critic evaluating two candidate items.
Prompt: "${promptText}"

Option A:
${textA}

Option B:
${textB}

Choose 'a', 'b', 'both_bad', or 'tie'. Provide a concise 1-sentence critique.
Output STRICTLY a JSON object with keys "winner" and "notes".
Example: {"winner": "a", "notes": "Better setup and unexpected punchline."}`;

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
      notes: 'Judged by Claude fallback heuristic.',
      judge: 'claude:fallback'
    };
  }
}

// 3. Multi-Model Judge Router
export function getJudgeDecision(backend: JudgeBackend, textA: string, textB: string, prompt: string): JudgeDecision {
  if (backend === 'homelab') {
    return queryHomelabJudge(textA, textB, prompt);
  } else {
    return queryClaudeJudge(textA, textB, prompt);
  }
}

// 4. Run Evolutionary Joke Convergence Simulation
export async function runJokeConvergenceSimulation(totalGenerations = 3, jokesPerGen = 5) {
  const tId = `homelab_humor_${Date.now().toString().slice(-4)}`;
  await ensureDir(ARTIFACTS_DIR);

  console.log(`\n=======================================================`);
  console.log(`🚀 STARTING MULTI-MODEL JOKE CONVERGENCE SIMULATION`);
  console.log(`Tournament ID: ${tId}`);
  console.log(`Generations: ${totalGenerations} | Jokes Per Generation: ${jokesPerGen}`);
  console.log(`Judges: Local 'homelab-default' (OMP) + Cloud 'claude-code'`);
  console.log(`=======================================================\n`);

  // Seed Initial Jokes
  const initialJokes = [
    { id: "joke_01", label: "DNS Root Cause", content: "There are only two hard things in Computer Science: cache invalidation, naming things, and it was actually DNS all along." },
    { id: "joke_02", label: "YAML Indentation", content: "My therapist told me to establish boundaries, so I spent 4 hours fixing tabs in a Kubernetes Helm chart." },
    { id: "joke_03", label: "Bare Metal Regret", content: "I bought an enterprise rack server for energy efficiency. My apartment is now 92 degrees and the breaker trips when the toaster runs." },
    { id: "joke_04", label: "GitOps Self-Heal", content: "ArgoCD is great because whenever I manually break production in panic, it immediately undoes my fix and breaks it the original way." },
    { id: "joke_05", label: "Local Model Quants", content: "I quantized my 70B model down to 1.5 bits. It runs at 400 tokens/second and only hallucinates in Shakespearean iambic pentameter." }
  ];

  console.log(`[1] Seeding Generation 1 with ${initialJokes.length} initial candidates...`);
  const createRes = await fetch(`${BASE_URL}/api/tournaments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: tId,
      title: "Homelab & Systems Engineering Dark Humor Arena",
      prompt: "Compare and rank the funniest, sharpest developer and homelab humor jokes.",
      task_type: "text",
      candidates: initialJokes,
      overwrite: true
    })
  });

  if (!createRes.ok) {
    throw new Error(`Failed to create tournament: ${await createRes.text()}`);
  }

  // Launch Playwright Browser
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  await page.goto(`${BASE_URL}/?t=${tId}`);
  await page.waitForSelector('#view-matchup');
  await page.waitForSelector('#body-a');
  await page.waitForSelector('#body-b');

  // Multi-generational loop
  for (let gen = 1; gen <= totalGenerations; gen++) {
    console.log(`\n-------------------------------------------------------`);
    console.log(`🏆 GENERATION ${gen} OF ${totalGenerations}`);
    console.log(`-------------------------------------------------------`);

    // Run 4 evaluation matchups per generation, alternating between Homelab model and Claude Code
    for (let m = 1; m <= 4; m++) {
      await page.waitForFunction(() => {
        const a = document.getElementById('body-a')?.textContent?.trim();
        const b = document.getElementById('body-b')?.textContent?.trim();
        return Boolean(a && b);
      });

      const bodyA = (await page.textContent('#body-a'))?.trim() || '';
      const bodyB = (await page.textContent('#body-b'))?.trim() || '';
      const metaA = (await page.textContent('#meta-a'))?.trim() || '';
      const metaB = (await page.textContent('#meta-b'))?.trim() || '';
      const prompt = (await page.textContent('#prompt-text'))?.trim() || '';

      // Alternate judges: odd matches homelab-default, even matches claude
      const judgeType: JudgeBackend = (m % 2 === 1) ? 'homelab' : 'claude';
      const decision = getJudgeDecision(judgeType, bodyA, bodyB, prompt);

      console.log(`[Gen ${gen} Match ${m}] [${decision.judge}]`);
      console.log(`  A [${metaA}]: "${bodyA.substring(0, 50)}..."`);
      console.log(`  B [${metaB}]: "${bodyB.substring(0, 50)}..."`);
      console.log(`  → Winner: [${decision.winner.toUpperCase()}] Rationale: "${decision.notes}"`);

      // Fill notes and blur
      await page.fill('#matchup-notes', decision.notes);
      await page.keyboard.press('Enter');
      await page.waitForTimeout(100);

      // Keypress vote
      const key = decision.winner === 'a' ? '1' : decision.winner === 'b' ? '2' : decision.winner === 'both_bad' ? '3' : 't';
      const oldA = bodyA;
      await page.keyboard.press(key);

      // Wait for card refresh
      await page.waitForFunction((old) => {
        const cur = document.getElementById('body-a')?.textContent?.trim();
        return Boolean(cur && cur !== old);
      }, oldA, { timeout: 3000 }).catch(() => {});
      await page.waitForTimeout(300);
    }

    // Evolve if not the last generation
    if (gen < totalGenerations) {
      console.log(`\n✨ Synthesizing preferences and evolving Generation ${gen + 1}...`);
      await page.keyboard.press('e');
      await page.waitForSelector('#evolve-modal', { state: 'visible' });

      // Alternate evolution backend: Gen 1 with Homelab, Gen 2 with Claude
      const evoBackend = (gen % 2 === 1) ? 'omp' : 'claude';
      console.log(`→ Evolution backend: ${evoBackend}`);

      await page.fill('#evolve-instructions', 'Keep the jokes deadpan, punchy, and grounded in real homelab disasters like DNS, ZFS rebuilds, thermal throttling, and GPU VRAM OOM.');
      await page.fill('#evolve-count', jokesPerGen.toString());

      await page.click('#btn-run-evolve');
      await page.waitForSelector('#evolve-modal', { state: 'hidden', timeout: 90000 });
      console.log(`✓ Generation ${gen + 1} successfully injected into arena!`);
      await page.waitForTimeout(1000);
    }
  }

  // Check Leaderboard
  console.log(`\n→ Navigating to Leaderboard...`);
  await page.evaluate(() => {
    const tabs = Array.from(document.querySelectorAll('.nav-tab'));
    const lTab = tabs.find(t => t.textContent?.includes('Leaderboard')) as HTMLElement;
    if (lTab) lTab.click();
  });
  await page.waitForSelector('#view-leaderboard', { state: 'visible' });
  await page.waitForSelector('#leaderboard-body tr');

  const rowCount = await page.$$eval('#leaderboard-body tr', rows => rows.length);
  const topCandidate = await page.$eval('#leaderboard-body tr:first-child td:nth-child(2)', el => el.textContent?.trim());
  const topElo = await page.$eval('#leaderboard-body tr:first-child td:nth-child(3)', el => el.textContent?.trim());

  console.log(`\n=======================================================`);
  console.log(`🎉 CONVERGENCE SIMULATION COMPLETE`);
  console.log(`- Total Candidates in Leaderboard: ${rowCount}`);
  console.log(`- Top Ranked Joke: "${topCandidate}" (Elo ${topElo})`);
  console.log(`=======================================================`);

  // Accept Champion
  const acceptBtn = await page.$('#leaderboard-body tr:first-child button');
  if (acceptBtn) {
    await acceptBtn.click();
    console.log(`✓ Winner accepted as tournament champion.`);
  }

  await page.screenshot({ path: path.join(ARTIFACTS_DIR, `${tId}_final_leaderboard.png`) });
  await browser.close();
  return { tournamentId: tId, totalCandidates: rowCount, winner: topCandidate, elo: topElo };
}

// Self-run when executed directly
if (require.main === module || process.argv[1]?.includes('multimodel_simulation')) {
  runJokeConvergenceSimulation(3, 5).catch(err => {
    console.error("Multi-model simulation failed:", err);
    process.exit(1);
  });
}
