## Phase 1: Basic Touch
Learn to reliably approach and make contact with the ball
- Rewards:
	- Touch reward: 50.0
	- Gentle nudge for moving toward puck: 0.05 per step
- Environment config:
	- center_serve_prob: 0.3
- Example command:
	```powershell
	python -m airhockey.train `
		--timesteps 100000 `
		--center-serve-prob 0.3 `
		--touch-reward 50.0 `
		--nudge-reward 0.05 `
		--visible
	```
## Phase 2: Multi-Touch and Control
Develop fine control and ball handling
- Rewards:
	- Touch reward: 25.0 per touch
	- No nudge
	- Consecutive touch reward: 25.0
- Environment config:
	- center_serve_prob: 0.15
- Example command:
	```powershell
	python -m airhockey.train `
	  --timesteps 100000 `
	  --center-serve-prob 0.15 `
	  --touch-reward 25.0 `
	  --nudge-reward 0.01 `
	  --consecutive-touch-reward 10.0 `
	  --visible
	```
## Phase 3: Basic Competition (Opponent Simulation)
Agent must compete for ball control and scoring against a simple scripted opponent
- Rewards:
	- Touch reward: 1.0
	- Scoring reward: 1.0 (for scoring a goal)
	- Concede penalty: -1.0 (for conceding a goal)
- Environment config:
	- center_serve_prob: 0.15
- Goal: Learn basic competitive behaviors (offense and defense)
- Example command:
	```powershell
	python -m airhockey.train `
	  --timesteps 100000 `
	  --center-serve-prob 0.15 `
	  --touch-reward 1.0 `
	  --nudge-reward 0.0 `
	  --opponent scripted `
	  --visible
	```
## Phase 4: Adaptive Curriculum & Full Match Play
- Agent must play full matches with scoring, defense, and resets; progression based on performance (e.g., advance when >90% success)
- Rewards:
	- Scoring reward: 1.0
	- Winning reward: 1.0 (end-of-episode bonus)
- Environment config:
	- center_serve_prob: 0.15
- Goal: Achieve robust play in realistic air hockey matches
- Example command:
	```powershell
	python -m airhockey.train `
	  --timesteps 100000 `
	  --center-serve-prob 0.15 `
	  --touch-reward 0.5 `
	  --full-match `
	  --visible
	```
# Notes:
- Reward values are rough recommendations; tune as needed for your environment and agent.
- Touch reward can be reduced or removed in later phases if agent is reliably making contact.
- Center serve probability is reduced after Phase 1 for more realistic play.
	--visible

```

