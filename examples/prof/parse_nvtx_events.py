import argparse
import sqlite3
import warnings
import statistics

import pandas as pd


def get_all_event_texts():
    sql = f"""
    SELECT DISTINCT text
    FROM NVTX_EVENTS;
    """
    df = pd.read_sql(sql, con)
    return [row['text'] for _, row in df.iterrows()]


def get_event_start_end(event_text):
    sql = f"""
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = '{event_text}';
    """
    df = pd.read_sql(sql, con)
    return [(row['start'], row['end']) for _, row in df.iterrows()]


def get_total_time_in_event(target_table_name, event_start, event_end):
    sql = f"""
    SELECT SUM(target.end - target.start) AS total_time
    FROM {target_table_name} target
    INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime
      ON target.correlationId = runtime.correlationId
    WHERE runtime.start BETWEEN {event_start} AND {event_end};
    """
    df = pd.read_sql(sql, con)
    time = df['total_time'].iloc[0]
    if time is None:
        return 0
    return time


def get_runtime_in_event(event_start, event_end):
    return event_end - event_start


def get_kernel_time_in_event(event_start, event_end):
    return get_total_time_in_event('CUPTI_ACTIVITY_KIND_KERNEL', event_start, event_end)


def get_memset_time_in_event(event_start, event_end):
    return get_total_time_in_event('CUPTI_ACTIVITY_KIND_MEMSET', event_start, event_end)


def get_memcpy_time_in_event(event_start, event_end):
    return get_total_time_in_event('CUPTI_ACTIVITY_KIND_MEMCPY', event_start, event_end)


def get_sync_time_in_event(event_start, event_end):
    return get_total_time_in_event('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION', event_start, event_end)


def main():
    if args.event_texts is None:
        event_texts = get_all_event_texts()
        if args.event_keywords is not None:
            event_keywords = args.event_keywords.split(',')
            event_texts = [txt for txt in event_texts
                           if any([kwd in txt for kwd in event_keywords])]
    else:
        event_texts = args.event_texts.split(',')
        if args.event_keywords is not None:
            warnings.warn('As event_texts is specified, event_keywords will be ignored.')

    times = {'ncalls': []}
    for key in ['runtime', 'kernel', 'memset', 'memcpy', 'sync']:
        times[key] = []
        times[f'{key}_stdev'] = []
    index = []
    print(f'Collecting time for {event_texts}')
    for txt in event_texts:
        event_start_end = get_event_start_end(txt)
        if len(event_start_end) == 0:
            continue
        index.append(txt)
        if args.ignore_first_event:
            # ignore first NVTX event
            event_start_end = event_start_end[1:]
        times['ncalls'].append(len(event_start_end))
        for key, f in {'runtime': get_runtime_in_event,
                       'kernel': get_kernel_time_in_event,
                       'memset': get_memset_time_in_event,
                       'memcpy': get_memcpy_time_in_event,
                       'sync': get_sync_time_in_event}.items():
            _times = [f(s, e) for s, e in event_start_end]
            times[key].append(statistics.mean(_times))
            times[f'{key}_stdev'].append(statistics.stdev(_times))

    df = pd.DataFrame(times, index=index)
    print(df)
    pickle_path = args.pickle_path
    print(f'Writing results to "{pickle_path}"')
    df.to_pickle(pickle_path)

    if args.wandb_run_path is not None:
        data = df.to_dict('index')
        import wandb
        run = wandb.Api().run(args.wandb_run_path)
        run.summary['times'] = 0
        run.summary['times'] = {key: 0 for key in data}
        run.summary.update({'times': data})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlite_path', type=str)
    parser.add_argument('--pickle_path', type=str, default='nvtx_events.pickle')
    parser.add_argument('--ignore_first_event', action='store_true')
    parser.add_argument('--event_texts', type=str)
    parser.add_argument('--event_keywords', type=str)
    parser.add_argument('--wandb_run_path', type=str, default=None)
    args = parser.parse_args()
    con = sqlite3.connect(args.sqlite_path)
    main()