import os
from multiprocessing import Pool
from tqdm import tqdm


def run_function_parallel(callable_function, njobs=-1, *args):
    if njobs < 0:
        processes = os.cpu_count() - 1
    else:
        processes = min(os.cpu_count() - 1, njobs)
    results = []

    with Pool(processes=processes) as p:
        with tqdm(total=len(args)) as pbar:
            for i, result in enumerate(p.imap(callable_function, args)):
                results.append(result)
                pbar.update()
    return results
