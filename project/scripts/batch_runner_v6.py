"""Shared v6/v7 runner; v7 accumulates statistics without recording history."""
import time


def run_batch(field_class, jobs, sim_config, fixed_params):
    import torch
    from src.fields.batch_v6 import BatchV6
    from .simu_every_para_up_v6 import deterministic_execution, mean_and_std
    started = time.perf_counter()
    fields = [field_class(**{**fixed_params, **combo}, device=sim_config['device'],
                          seed=sim_config['base_seed'] + run_id)
              for combo, run_id in jobs]
    batch = BatchV6(fields)
    del fields
    record_history = getattr(field_class, 'record_history', True)
    records = []
    series, genotypes = ([], []) if record_history else ((), ())
    total_steps = sim_config['total_steps']
    eval_start = sim_config['eval_start_step']
    stats = torch.zeros((batch.size, 7), dtype=torch.float64, device=batch.device)
    interval = (torch.zeros((batch.size, 2), dtype=torch.float64, device=batch.device)
                if record_history else None)
    record_rows = [i for i, (_, run_id) in enumerate(jobs) if run_id == 0] if record_history else ()

    def record_sample(step):
        sample = torch.cat((batch.populations[:, None], interval), 1)[record_rows].cpu().tolist()
        genotype_counts = batch.genotype_counts()[record_rows].cpu().tolist()
        for row, values, counts in zip(record_rows, sample, genotype_counts):
            combo, run_id = jobs[row]
            identifiers = {**combo, 'run_id': run_id,
                           'seed': sim_config['base_seed'] + run_id, 'step': step}
            n, denominator, altruistic = values
            series.append({**identifiers, 'population': int(n), 'altruism_count': int(altruistic),
                           'altruism_rate': altruistic / denominator if denominator > 0 else None})
            genotypes.extend({**identifiers, 'genotype': format(code, '08b'), 'population': count}
                             for code, count in enumerate(counts) if count)
        interval.zero_()

    with deterministic_execution():
        for step in range(1, total_steps + 1):
            if any(batch.counts):
                batch.step()
                deaths = batch.death_ages.to(torch.float64)
                death_counts = batch.death_counts
            else:
                # Extinction contributes zero for every remaining evaluation step.
                deaths = torch.zeros((batch.size, 1), device=batch.device, dtype=torch.float64)
                death_counts = torch.zeros_like(batch.populations)
                batch.altruism_count.zero_()
            population = batch.populations.to(torch.float64)
            altruism = batch.altruism_count.to(torch.float64)
            if record_history:
                interval[:, 0] += population
                interval[:, 1] += altruism
            if step >= eval_start:
                stats += torch.stack((population, population.square(), altruism, altruism.square(),
                                      death_counts, deaths.sum(1), deaths.square().sum(1)), 1)
            if record_history and (step == 1 or step % 1000 == 0):
                record_sample(step)
            if not any(batch.counts):
                if record_history:
                    for sample_step in range((step // 1000 + 1) * 1000, total_steps + 1, 1000):
                        record_sample(sample_step)
                break
        values = stats.cpu().tolist()  # Also waits for outstanding CUDA work.
    for (combo, run_id), sums in zip(jobs, values):
        window = total_steps - eval_start + 1
        population, population_std = mean_and_std(sums[0], sums[1], window)
        altruism, altruism_std = mean_and_std(sums[2], sums[3], window)
        age, age_std = mean_and_std(sums[5], sums[6], sums[4])
        records.append({'run_id': run_id, 'seed': sim_config['base_seed'] + run_id,
                        'avg_population': population, 'std_population': population_std,
                        'avg_altruism_count': altruism, 'std_altruism_count': altruism_std,
                        'avg_altruism_rate': altruism / population if population > 0 else None,
                        'avg_death_age': age, 'std_death_age': age_std, **combo})
    print(f"Completed batch: {len(jobs)} simulations, elapsed: {time.perf_counter()-started:.3f} s", flush=True)
    return records, series, genotypes


def _attempt_batch(field_class, jobs, sim_config, fixed_params):
    import torch
    try:
        return run_batch(field_class, jobs, sim_config, fixed_params)
    except torch.cuda.OutOfMemoryError:
        if not str(sim_config['device']).startswith('cuda'):
            raise
        # Return from this frame before retrying, releasing traceback-held buffers.
        return None


def run_with_retry(field_class, jobs, sim_config, fixed_params):
    import torch
    result = _attempt_batch(field_class, jobs, sim_config, fixed_params)
    if result is not None:
        yield result
        return
    torch.cuda.empty_cache()
    if len(jobs) == 1:
        version = field_class.__name__.removeprefix('Field').lower()
        raise RuntimeError(f'A single {version} simulation does not fit in CUDA memory; no agents were discarded.')
    middle = len(jobs) // 2
    print(f"CUDA memory exhausted; retrying {len(jobs)} simulations in smaller batches.", flush=True)
    yield from run_with_retry(field_class, jobs[:middle], sim_config, fixed_params)
    yield from run_with_retry(field_class, jobs[middle:], sim_config, fixed_params)
