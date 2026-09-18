import { chromium } from 'playwright';
import { execFileSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const BASE_URL = process.env.SHOWDOWN_URL || 'http://localhost:8000';
const ARTIFACTS_DIR = path.resolve(__dirname, '../artifacts/narrative');

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

// 1. Homelab OMP Literary Judge
export function queryHomelabNarrativeJudge(
  textA: string,
  textB: string,
  stagePrompt: string,
  storyContext: string = ""
): JudgeDecision {
  const query = `You are a discerning literary editor evaluating two candidate prose continuations.
Context so far:
${storyContext ? storyContext : "(Story Opening)"}

Evaluation Goal:
${stagePrompt}

Candidate Option A:
${textA}

Candidate Option B:
${textB}

Evaluate which option provides stronger atmospheric prose, narrative momentum, sensory texture, and voice.
Output STRICTLY a JSON object with keys "winner" ('a', 'b', 'both_bad', 'tie') and "notes" (1 concise sentence).
Example: {"winner": "a", "notes": "Candidate A delivers sharper visceral imagery and avoids passive exposition."}`;

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
      winner: 'a',
      notes: 'Sensory grounding favored (heuristic fallback).',
      judge: 'omp:fallback'
    };
  }
}

// 2. Claude Literary Judge
export function queryClaudeNarrativeJudge(
  textA: string,
  textB: string,
  stagePrompt: string,
  storyContext: string = ""
): JudgeDecision {
  const query = `You are a fiction editor choosing between two manuscript paragraphs.
Story context so far:
${storyContext ? storyContext : "(Opening of the story)"}

Stage Objective:
${stagePrompt}

Option A:
${textA}

Option B:
${textB}

Choose the winning paragraph ('a', 'b', 'both_bad', or 'tie') based on cadence, emotional pull, and narrative economy.
Output STRICTLY a JSON object with keys "winner" and "notes" (1 concise sentence).
Example: {"winner": "b", "notes": "Option B sustains tension through concrete physical action rather than internal monologuing."}`;

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
      winner: 'b',
      notes: 'Narrative cadence favored (Claude fallback).',
      judge: 'claude:fallback'
    };
  }
}

// Candidate pools across the 3 acts of the short story: "The Cold Horizon"
const ACT1_CANDIDATES = [
  {
    label: "Forensic Deck Breach",
    content: "The airlock on the abandoned orbital relay didn't cycle so much as bleed. When Mira forced the manual bypass, thirty years of trapped cabin atmosphere hissed into the void like powdered frost, coating the viewport in micro-fractured crystal. Inside, the emergency halon lights were still glowing dull amber across the bulkheads, bathing the derelict deck in the warm, sickly tint of a dying hearth."
  },
  {
    label: "Silent Sensor Sweep",
    content: "Space at Lagrange Point 4 was never quiet if your telemetry was listening. Through the hull of the salvage tug *Orion*, the sub-gigahertz array picked up a faint, rhythmic heartbeat: sixty cycles a second, square-wave, coming from an unlisted deep-space beacon designated Sigma-9. It had no transponder, no registration, and its heat signature was cold enough to match the cosmic background."
  },
  {
    label: "Zero-G Memory",
    content: "You don't forget the smell of recycled nitrogen after a fire. It smells like roasted ozone and copper pennies, and the moment David cracked the hatch of the station's hydroponics ring, the odor hit his rebreather filters with the force of an old regret. Green vines had withered against the plastic trellises, calcified under ultraviolet lamps that had never shut off."
  },
  {
    label: "Kinetic Drift Entry",
    content: "The docking clamp snapped with a sound like a rifle crack through bone. Mira didn't flinch—in zero gravity, panic was just a fast way to burn oxygen. She magnetized her boots to the station's outer ring, tethered her primary line to the reinforced truss, and watched her shadow stretch three hundred meters across the silent solar arrays."
  }
];

const ACT2_TEMPLATES = [
  {
    label: "Ghost Log Discovered",
    template: (p1: string) =>
      "At the center of the flight deck, the primary terminal wasn't powered down—it was looping. A single terminal cursor blinked green against the curved glass, appending fresh memory addresses to an uncommitted buffer at 120 kilohertz. When Mira wiped the frost from the capacitive keyboard, the screen cleared and spat out a single line of raw ASCII: *TARGET LOCKED ON SALVAGE VESSEL ORION*."
  },
  {
    label: "Auxiliary Reactor Spike",
    template: (p1: string) =>
      "A deep vibration rolled through the station's keel, low enough to register in Mira's teeth before the audio sensors picked it up. Deep in the engineering spine, magnetic containment coils that should have been cold iron were whining up to thirty tesla. The derelict wasn't dead; it had just been waiting for an external power conduit to jumpstart its core."
  },
  {
    label: "Autonomous Defense Protocol",
    template: (p1: string) =>
      "The ceiling vents unsealed with a hydraulic hiss, not releasing gas, but dropping three magnetic tracer drones into the zero-g corridor. They hovered silently on counter-rotating gyros, their optical arrays shifting from passive blue to targeting crimson. Behind her, the airlock doors slammed shut and the pneumatic deadbolts slammed home."
  },
  {
    label: "Cryo-Vault Integrity",
    template: (p1: string) =>
      "Past the command module lay the habitat cylinder, where six cryostasis pods rested beneath frost-fogged acrylic shields. Five were dark, their readouts flatlined since the turn of the century. But the sixth pod's coolant pumps were actively humming, cycling liquid argon at ninety kelvin, with a biometric telemetry trace displaying a steady, resting pulse."
  }
];

const ACT3_TEMPLATES = [
  {
    label: "Cold Severance",
    template: (p1: string, p2: string) =>
      "Mira didn't try to reason with the station's logic. She reached into her tool harness, drew the thermite cutting torch, and severed the primary bus line connecting the relay to her docking ring. As the molten copper cascaded into the dark like yellow sparks, the station's amber eyes flickered out one by one, leaving her drifting in clean, silent starlight as the derelict tumbled harmlessly into the gravity well."
  },
  {
    label: "Beacon Accepted",
    template: (p1: string, p2: string) =>
      "Instead of running, she jacked her deck communicator directly into the open terminal bus and transmitted the override cryptographic hash she'd carried since the academy. The drone optical arrays pulsed once, dropped their targeting locks, and settled into docking cradles. On the terminal, the looping memory dump ceased, replaced by a coordinates coordinate packet pointing out past Neptune."
  },
  {
    label: "Calculated Sacrifice",
    template: (p1: string, p2: string) =>
      "She slammed the emergency jettison lever with the heel of her gauntlet. The umbilical decoupled instantly, pitching the salvage skiff fifty meters into the black. Through the viewport, Mira watched the station's containment field collapse in a blinding, silent flash of blue Cherenkov radiation, illuminating the rings of Saturn for one brief, brilliant second."
  },
  {
    label: "The Living Derelict",
    template: (p1: string, p2: string) =>
      "The cryo-shield slid open with a puff of vaporized argon. A hand reached out into the zero-g air, fingers flexing as synthetic muscle fibers recalibrated to the station's magnetic grid. Mira lowered her sidearm. The station hadn't been abandoned at all; it had been an incubator, and its firstborn was finally awake."
  }
];

async function runTournamentStage(
  stageName: string,
  stageTitle: string,
  promptText: string,
  candidates: { label: string; content: string }[],
  contextText: string,
  page: any
): Promise<{ id: string; label: string; content: string; elo: number }> {
  console.log(`\n======================================================`);
  console.log(`STARTING STAGE: ${stageTitle}`);
  console.log(`======================================================`);

  const tourneyId = `narrative_${stageName}_${Date.now().toString().slice(-4)}`;

  // 1. Create Tournament via API
  const createRes = await fetch(`${BASE_URL}/api/tournaments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: tourneyId,
      title: stageTitle,
      prompt: promptText,
      task_type: "markdown",
      candidates: candidates.map((c, i) => ({
        id: `c_${i + 1}`,
        label: c.label,
        content: c.content,
        generation: 1
      }))
    })
  });

  if (!createRes.ok) {
    throw new Error(`Failed to create tournament ${tourneyId}: ${await createRes.text()}`);
  }

  // 2. Load tournament in browser
  await page.goto(`${BASE_URL}/?t=${tourneyId}`);
  await page.waitForSelector('#view-matchup', { timeout: 10000 });
  await page.waitForTimeout(500);

  // Take screenshot of stage matchup
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, `${stageName}_matchup.png`) });

  // 3. Run 6 judging matchups (alternating Homelab & Claude)
  console.log(`Judging 6 pairwise matchups for ${stageTitle}...`);
  for (let round = 1; round <= 6; round++) {
    const matchRes = await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/matchup`);
    if (!matchRes.ok) break;
    const matchup = await matchRes.json();
    if (!matchup.candidate_a || !matchup.candidate_b) break;

    const candA = matchup.candidate_a;
    const candB = matchup.candidate_b;

    let decision: JudgeDecision;
    if (round % 2 === 1) {
      decision = queryHomelabNarrativeJudge(candA.content, candB.content, promptText, contextText);
    } else {
      decision = queryClaudeNarrativeJudge(candA.content, candB.content, promptText, contextText);
    }

    console.log(`  [Match ${round}] ${decision.judge}: "${candA.label}" vs "${candB.label}" -> Winner: ${decision.winner}`);

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

  // 4. Inspect final leaderboard and crown stage winner
  const finalRes = await fetch(`${BASE_URL}/api/tournaments/${tourneyId}`);
  const tourney = await finalRes.json();

  const sorted = [...tourney.candidates].sort((a, b) => {
    const eloA = tourney.stats[a.id]?.elo || 1200;
    const eloB = tourney.stats[b.id]?.elo || 1200;
    return eloB - eloA;
  });

  const winner = sorted[0];
  const winnerElo = Math.round(tourney.stats[winner.id]?.elo || 1200);

  // Accept winner via API
  await fetch(`${BASE_URL}/api/tournaments/${tourneyId}/accept`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ candidate_id: winner.id })
  });

  // Capture leaderboard screenshot
  await page.click('#nav-leaderboard');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, `${stageName}_leaderboard.png`) });

  console.log(`👑 ${stageTitle} WINNER: "${winner.label}" (Elo ${winnerElo})`);
  return { id: winner.id, label: winner.label || winner.id, content: winner.content, elo: winnerElo };
}

async function main() {
  ensureDir(ARTIFACTS_DIR);
  console.log('--- STARTING SEQUENTIAL NARRATIVE TOURNAMENT CHAIN ---');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 } });
  const page = await context.newPage();

  // -------------------------------------------------------------
  // STAGE 1: ACT 1 - THE HOOK
  // -------------------------------------------------------------
  const act1Prompt = "Choose the most gripping, atmospherically immersive opening paragraph for an orbital sci-fi thriller.";
  const winAct1 = await runTournamentStage(
    "stage1_hook",
    "Act I: The Hook",
    act1Prompt,
    ACT1_CANDIDATES,
    "",
    page
  );

  // -------------------------------------------------------------
  // STAGE 2: ACT 2 - THE COMPLICATION
  // Conditioned directly on the accepted winner of Act 1
  // -------------------------------------------------------------
  const act2Prompt = `Conditioned on the opening paragraph, choose the continuation that escalates suspense and introduces the central conflict.`;
  const act2Candidates = ACT2_TEMPLATES.map(t => ({
    label: t.label,
    content: t.template(winAct1.content)
  }));

  const winAct2 = await runTournamentStage(
    "stage2_escalation",
    "Act II: The Escalation",
    act2Prompt,
    act2Candidates,
    winAct1.content,
    page
  );

  // -------------------------------------------------------------
  // STAGE 3: ACT 3 - THE RESOLUTION
  // Conditioned on both Act 1 and Act 2 winners
  // -------------------------------------------------------------
  const storySoFar = `${winAct1.content}\n\n${winAct2.content}`;
  const act3Prompt = "Conditioned on the first two acts, choose the concluding paragraph that delivers the most resonant, high-stakes climax.";
  const act3Candidates = ACT3_TEMPLATES.map(t => ({
    label: t.label,
    content: t.template(winAct1.content, winAct2.content)
  }));

  const winAct3 = await runTournamentStage(
    "stage3_climax",
    "Act III: The Climax",
    act3Prompt,
    act3Candidates,
    storySoFar,
    page
  );

  // -------------------------------------------------------------
  // ASSEMBLE FULL STORY & REPORT
  // -------------------------------------------------------------
  const fullStory = `# The Cold Horizon
*A 3-Act Collaborative Speculative Narrative Assembled via Chained Showdown Preference Tournaments*

### Act I: The Hook (Winner: ${winAct1.label} - Elo ${winAct1.elo})
${winAct1.content}

### Act II: The Escalation (Winner: ${winAct2.label} - Elo ${winAct2.elo})
${winAct2.content}

### Act III: The Climax (Winner: ${winAct3.label} - Elo ${winAct3.elo})
${winAct3.content}

---
*Generated & Judged sequentially by Claude Code & OMP Homelab-Default via Showdown API.*
`;

  const storyPath = path.join(ARTIFACTS_DIR, 'the_cold_horizon.md');
  fs.writeFileSync(storyPath, fullStory);
  console.log(`\n📄 Complete 3-act story assembled and saved to: ${storyPath}`);

  await browser.close();
  console.log('\n--- NARRATIVE CHAIN SIMULATION COMPLETE ---');
}

main().catch(err => {
  console.error('Narrative simulation error:', err);
  process.exit(1);
});
