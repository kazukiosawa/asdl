import os
import argparse
import sqlite3

import pandas as pd


def get_correlation_ids_in_range(start, end):
    sql = f"""
    SELECT correlationId
    FROM CUPTI_ACTIVITY_KIND_RUNTIME
    WHERE end BETWEEN {start} AND {end};
    """
    df = pd.read_sql(sql, con)
    return [row['correlationId'] for _, row in df.iterrows()]


def get_markers():
    print('Collecting markers')
    sql = f"""
    SELECT marker.id, StringTable.value AS name
    FROM CUPTI_ACTIVITY_KIND_MARKER marker
    INNER JOIN StringTable 
      ON marker.name = StringTable._id_
    WHERE marker.name != 0
    """
    df = pd.read_sql(sql, con)
    markers = []
    for _, row in df.iterrows():
        marker_id = row['id']
        marker_name = row['name']
        sql = f"""
        SELECT timestamp
        FROM CUPTI_ACTIVITY_KIND_MARKER
        WHERE id = {marker_id}
        """
        _df = pd.read_sql(sql, con)
        assert len(_df.index) == 2, f'Got {len(_df.index)} markers of the same id (name: {marker_name}).' \
                                    f' This has to be 2.'
        start = _df['timestamp'].iloc[0]
        end = _df['timestamp'].iloc[1]
        markers.append(
            {
                'name': marker_name,
                'start': start,
                'end': end,
                'correlation_ids': get_correlation_ids_in_range(start, end)
            })
    return markers


def get_kernels():
    print('Collecting kernels')
    sql = f"""
    SELECT kernel.correlationId, kernel.start, kernel.end, StringTable.value AS name
    FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL kernel
    INNER JOIN StringTable
      ON kernel.name = StringTable._id_
    """
    df = pd.read_sql(sql, con)
    kernels = []
    for _, row in df.iterrows():
        kernels.append(
            {
                'correlation_id': row['correlationId'],
                'start': row['start'],
                'end': row['end'],
                'name': row['name'],
            })
    return kernels


def main():
    markers = get_markers()
    markers = sorted(markers, key=lambda m: len(m['correlation_ids']), reverse=True)
    kernels = get_kernels()
    max_num_markers = 0
    save_kernels = []
    print('Assigning markers to kernels')
    for kernel in kernels:
        _markers = [marker for marker in markers
                    if kernel['correlation_id'] in marker['correlation_ids']]
        if len(_markers) == 0:
            continue
        skip = False
        for i in range(len(_markers)):
            marker_name = _markers[i]['name']
            if marker_name in exclude_names:
                skip = True
                break
            kernel[f'marker{i}'] = marker_name
        if not skip:
            max_num_markers = max(max_num_markers, len(_markers))
            save_kernels.append(kernel)

    print('Constructing dataframe and csv')
    df = pd.DataFrame(
        {
            'Kernel': [k['name'] for k in save_kernels],
            'duration': [k['end'] - k['start'] for k in save_kernels],
        }
    )
    for i in range(max_num_markers):
        df[f'Marker{i}'] = [k.get(f'marker{i}', '-') for k in save_kernels]
    csv_path = args.csv_path
    if csv_path is None:
        csv_path = os.path.splitext(args.sqlite_path)[0] + '.csv'
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlite_path', type=str)
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--exclude_names', type=str, default=None)
    args = parser.parse_args()
    con = sqlite3.connect(args.sqlite_path)
    if args.exclude_names is not None:
        exclude_names = args.exclude_names.split(',')
    else:
        exclude_names = []
    main()