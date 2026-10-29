# Soccer: simple inside-foot pass (v1)

The Asimov 1 humanoid learns to pass a ball with the inside of its right foot, then step back into a steady stance.
Simulation only (Isaac Lab).

## What the robot does

1. Starts just before the kick, left foot planted beside the ball and pointing where the pass should go.
2. Swings the right leg and hits the middle of the ball with the inside of the foot.
3. The ball rolls along the ground toward the target at about 1.5 m/s.
4. Brings the right leg back, steps the left foot back next to it, and stands still, upright, facing the pass.

Throughout, the head faces where the plant foot points, the torso stays upright, and the arms stay low.

## How we built it

**1. Gave the feet something to kick with.** The original robot model only has four small contact points under each
foot, so a ball would pass straight through it. We added a box collider shaped like the foot
(`assets/robots/asimov_1_soccer.py`).

**2. Made a reference motion without motion capture.** `scripts/soccer/pass_motion.py` places the feet at a few key
poses (plant, backswing, contact, follow-through, step back, stand) and solves for joint angles frame by frame. It
also keeps the robot balanced, the torso upright, the head facing the plant foot and the arms relaxed. There is one
clip per ball size, in `motions/`.

**3. Trained with reinforcement learning plus a style score (AMP).** The policy is rewarded for the result, and a
second network scores how much its motion looks like the reference clip, which keeps the movement natural. The main
rewards (`mdp/pass_terms.py`):

- **The touch:** a bonus when the inside of the right foot hits the middle of the ball first, and a penalty if
  another body part touches it first.
- **The pass:** the ball's launch speed close to 1.5 m/s, its direction close to the target, and the ball staying on
  the ground.
- **The recovery:** after the kick, upright, still, feet side by side, standing pose, facing the pass.
- **The upper body:** arms near a relaxed pose, hands below chest height, head facing the plant foot.
- **Staying up:** a reward for every step alive and a large penalty for falling.

Every episode starts from a random moment of the reference clip just before the kick, rotated to a random direction.

**4. Grew the ball.** We trained on a 5 cm (radius) ball first, then fine-tuned for 7, 9 and 11 cm. Bigger balls
need the knee lifted so the foot meets the middle of the ball.

## Results

Tested over 16 passes per ball size, after about 7,000 iterations:

| Ball radius | Clean inside-foot touches | Ball speed | Sideways error after ~2 m | Falls |
|---|---|---|---|---|
| 7 cm | 16 / 16 | 1.6 m/s | 6 cm | 0 |
| 11 cm | 16 / 16 | 1.6 m/s | 5 cm | 1 |

## Things we learned

- A kick can start and end within one control step, so the ball's contact sensor has to keep every physics substep,
  or most touches are missed.
- Rewarding "ball faster than X" makes the robot blast the ball; rewarding a target speed gives a pass.
- Without an upright term in the reference motion, the solver balances by tilting the whole torso.

## Try it

```bash
# Train (5 cm ball; use Asimov1-Pass-R07-v0 / -R09-v0 / -R11-v0 for bigger balls)
./cyclotron.sh --train --task Asimov1-Pass-v0 --num_envs 4096 --headless

# Make a reference clip for a ball size
MUJOCO_GL=egl python scripts/soccer/pass_motion.py --ball_radius 0.11

# Score a trained policy and render GIFs (on the cluster)
TASK=Asimov1-Pass-R11 REPLAY_ARGS="--slow 3" sbatch slurm/record_pass.sbatch <run folder>
```

## Next

Harder starts (from further away, other angles, pushes), then walking up to the ball with the walking policy before
passing.
