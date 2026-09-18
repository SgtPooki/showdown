#!/usr/bin/env bash
# dogfood.sh — Automated Playwright battle-test and roleplay simulation for Showdown
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running Showdown Playwright dogfooding & roleplay simulation..."
bun run e2e/battle_test.ts
