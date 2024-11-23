from pytabkit.bench.data.paths import Paths
from pytabkit.bench.eval.runtimes import get_avg_train_times, get_avg_predict_times

if __name__ == '__main__':
    paths = Paths.from_env_variables()
    for coll_name in ['meta-train-class', 'meta-train-reg']:
        times_dict = get_avg_train_times(paths, coll_name, per_1k_samples=True)
        print(f'Average training times per 1K samples for {coll_name}:')
        for alg_name, time_s in times_dict.items():
            print(f'{alg_name}: {time_s:g} s')
        print(times_dict)
    for coll_name in ['meta-train-class', 'meta-train-reg']:
        times_dict = get_avg_predict_times(paths, coll_name, per_1k_samples=True)
        print(f'Average inference times per 1K samples for {coll_name}:')
        for alg_name, time_s in times_dict.items():
            print(f'{alg_name}: {time_s:g} s')
        print(times_dict)
