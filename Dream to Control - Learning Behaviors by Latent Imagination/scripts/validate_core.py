"""Hand-checked return calculation and environment invariants."""
import json
import pathlib
import common
from common import np, tf, tools, wrappers

def main():
    reward = tf.constant([[1.], [2.], [3.]])
    value = tf.constant([[10.], [20.], [30.]])
    bootstrap = tf.constant([40.])
    checks = {}
    for lam, expected in [(0.0, [19., 29., 39.]),
                          (1.0, [34.39, 37.1, 39.]),
                          (0.5, [24.8725, 33.05, 39.])]:
        # For lambda=0.5, G2=39, G1=15.5+.45*39=33.05,
        # G0=10+.45*33.05=24.8725.
        actual = tools.lambda_return(reward, value, 0.9, bootstrap, lam, 0).numpy()[:, 0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        checks[f'lambda_{lam}'] = actual.tolist()
    actual = tools.lambda_return(reward, value, 0.0, bootstrap, 0.95, 0).numpy()
    np.testing.assert_allclose(actual, reward.numpy())
    checks['zero_discount'] = 'passed'
    env1 = wrappers.DeepMindControl('walker_walk', seed=7)
    env2 = wrappers.DeepMindControl('walker_walk', seed=7)
    np.testing.assert_array_equal(env1.reset()['image'], env2.reset()['image'])
    spec = env1.action_space
    np.testing.assert_array_equal(env1.step(np.zeros(spec.shape))[0]['image'],
                                  env2.step(np.zeros(spec.shape))[0]['image'])
    env1.close()
    env2.close()
    checks['environment_seed_repeatability'] = 'passed'
    output = common.ROOT / 'output' / 'core_validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'status': 'passed', 'checks': checks}, indent=2))
    print(output.read_text(), flush=True)

if __name__ == '__main__':
    common.configure(0, 32)
    with tf.device('/CPU:0'):
        main()
