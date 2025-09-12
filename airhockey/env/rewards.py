def compute_rewards(env, reward, touched, rel_norm_pos):
	env._prev_approach_dist = None
	if touched:
		reward += env.touch_reward
		impact_speed = rel_norm_pos
		HIT_REWARD_SCALE = 0.01
		MAX_HIT_BONUS = 5.0
		impact_bonus = min(impact_speed * HIT_REWARD_SCALE, MAX_HIT_BONUS)
		reward += float(impact_bonus)
		if env._ep_touches > 0:
			if env.consecutive_touch_reward > 0.0:
				reward += env.consecutive_touch_reward
			env._ep_consecutive_touches += 1
	env._prev_touched = touched
	return reward
