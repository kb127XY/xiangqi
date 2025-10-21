import sys, os
sys.path.append(os.path.dirname(__file__))  # ✅ 手动加入当前目录
import random
import math
import pickle
from collections import defaultdict
from typing import Any, Dict, Tuple, Optional, List
from Xiangqi_env import XiangqiEnv, opponent_policy # 你已有的环境


State = Any
Action = Any
QTable = Dict[Tuple[State, Action], float]


def set_global_seed(seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)


def epsilon_greedy(Q: QTable, state: State, actions: List[Action], epsilon: float) -> Optional[Action]:
    if not actions:
        return None
    if random.random() < epsilon:
        return random.choice(actions)
    # 找到最大Q值的所有动作并随机选择其一，避免系统性偏置
    values = [Q[(state, a)] for a in actions]
    max_val = max(values)
    bests = [a for a, v in zip(actions, values) if v == max_val]
    return random.choice(bests)


def greedy_action(Q: QTable, state: State, actions: List[Action]) -> Optional[Action]:
    """纯贪心（用于评估或对局时）。"""
    return epsilon_greedy(Q, state, actions, epsilon=0.0)


def evaluate_policy(env: XiangqiEnv, Q: QTable, episodes: int = 50, max_steps: int = 400) -> Dict[str, float]:
    """用当前Q表进行对局评估（红方贪心 vs 既定对手策略），返回胜/平/负比例与平均回报。"""
    wins = draws = losses = 0
    total_reward = 0.0

    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        while not done and steps < max_steps:
            if env.is_red_to_move():
                actions = env.legal_actions(state)
                a = greedy_action(Q, state, actions)
                if a is None:
                    # 无合法动作，环境应在 step 内部判定结果，这里给出保守处理
                    a = random.choice(actions) if actions else None
                next_state, r_env, done, info = env.step(a)
                ep_reward += r_env
                state = next_state
            else:
                actions = env.legal_actions(state)
                if not actions:
                    # 对手无合法动作，同理让环境处理
                    a_op = None
                else:
                    # 这里用你提供的对手策略；也可换成纯随机：random.choice(actions)
                    a_op = opponent_policy(env, state, Q, epsilon=0.2)
                    if a_op not in actions:  # 防御策略返回非法动作
                        a_op = random.choice(actions)
                next_state, _, done, _ = env.step(a_op)
                state = next_state

            steps += 1

        total_reward += ep_reward
        # 需要你的 env 在 info 或终局奖励里有结果归属（例：'result' in info or 依据 r_env）
        # 常见做法：r_env > 0 视为红胜，r_env < 0 视为红负，0 为和棋
        if done:
            res = info.get("result", None)
            if res == "win":
                wins += 1
            elif res == "loss":
                losses += 1
            else:
                draws += 1
    return {
        "win_rate": wins / episodes,
        "draw_rate": draws / episodes,
        "loss_rate": losses / episodes,
        "avg_return": total_reward / episodes,
    }


def train_q_learning_basic(
    episodes: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 0.5,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,  # 指数衰减；也可用线性
    seed: Optional[int] = 42,
    max_steps_per_episode: int = 400,
    eval_every: int = 1000,
    eval_episodes: int = 50,
    verbose: bool = True,
) -> QTable:
    """
    训练基本 Q-Learning（红方学习），黑方由 opponent_policy 或随机走子。
    - 仅在红方回合更新Q。
    - 使用 ε 指数衰减（默认每回合衰减一次）。
    """
    set_global_seed(seed)
    env = XiangqiEnv()
    Q: QTable = defaultdict(float)

    epsilon = epsilon_start

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            if env.is_red_to_move():
                actions = env.legal_actions(state)
                a = epsilon_greedy(Q, state, actions, epsilon)
                if a is None:
                    # 理论上到这说明红方无合法动作，让环境来判终局
                    next_state, r_env, done, info = env.step(a)
                else:
                    next_state, r_env, done, info = env.step(a)

                # Q-learning 目标
                if done:
                    target = 0.0
                else:
                    next_actions = env.legal_actions(next_state)
                    if next_actions:
                        target = max(Q[(next_state, a2)] for a2 in next_actions)
                    else:
                        target = 0.0

                Q[(state, a)] += alpha * (r_env + gamma * target - Q[(state, a)])
                state = next_state

            else:
                # 黑方走子（对手策略）
                actions = env.legal_actions(state)
                if not actions:
                    a_op = None
                else:
                    # 你的对手策略；也可替换为纯随机：random.choice(actions)
                    a_op = opponent_policy(env, state, Q, epsilon=0.2)
                    if a_op not in actions:
                        a_op = random.choice(actions)

                next_state, _, done, _ = env.step(a_op)
                state = next_state

            steps += 1

        # 每回合衰减 ε（更可控）
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # 定期评估
        if verbose and (ep % eval_every == 0 or ep == 1):
            stats = evaluate_policy(env, Q, episodes=eval_episodes, max_steps=max_steps_per_episode)
            print(
                f"[Episode {ep:>6}] ε={epsilon:.3f} | "
                f"Win {stats['win_rate']:.2%}  Draw {stats['draw_rate']:.2%}  "
                f"Loss {stats['loss_rate']:.2%} | AvgRet {stats['avg_return']:.3f}"
            )

    return Q


def save_q_table(Q: QTable, path: str = "q_table.pkl"):
    with open(path, "wb") as f:
        pickle.dump(dict(Q), f)  # 存成普通 dict，避免 defaultdict 的反序列化陷阱


def load_q_table(path: str = "q_table.pkl") -> QTable:
    with open(path, "rb") as f:
        data = pickle.load(f)
    Q: QTable = defaultdict(float)
    Q.update(data)
    return Q


def play_one_game(Q: QTable, seed: Optional[int] = None, max_steps: int = 400, verbose: bool = True):
    """用训练好的 Q 表进行一次演示对局（红方贪心），打印每一步（取决于你的 env 是否提供便捷展示）。"""
    set_global_seed(seed)
    env = XiangqiEnv()
    state = env.reset()
    done = False
    steps = 0
    ep_reward = 0.0

    if verbose and hasattr(env, "render"):
        env.render()

    while not done and steps < max_steps:
        if env.is_red_to_move():
            actions = env.legal_actions(state)
            a = greedy_action(Q, state, actions) or (random.choice(actions) if actions else None)
            next_state, r_env, done, info = env.step(a)
            ep_reward += r_env
            if verbose:
                print(f"[RED] {a} | r={r_env}")
        else:
            actions = env.legal_actions(state)
            a_op = opponent_policy(env, state, Q, epsilon=0.0)
            if a_op not in actions:
                a_op = random.choice(actions) if actions else None
            next_state, r_env, done, info = env.step(a_op)
            if verbose:
                print(f"[BLK] {a_op}")

        state = next_state
        steps += 1

        if verbose and hasattr(env, "render"):
            env.render()

    if verbose:
        if ep_reward > 0:
            print(f"Game over. RED wins. Return={ep_reward:.2f}")
        elif ep_reward < 0:
            print(f"Game over. RED loses. Return={ep_reward:.2f}")
        else:
            print(f"Game over. Draw. Return={ep_reward:.2f}")


if __name__ == "__main__":
    # 1) 训练
    Q = train_q_learning_basic(
        episodes=20000,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=0.6,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        seed=123,
        max_steps_per_episode=500,
        eval_every=10,
        eval_episodes=30,
        verbose=True,
    )

    # 2) 保存
    save_q_table(Q, "q_table_basic.pkl")
    print("Q table saved to q_table_basic.pkl")

    # 3) 演示一局
    play_one_game(Q, seed=123, verbose=False)
