import sqlite3
import json

conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()
cursor.execute('SELECT job_id, results, options FROM jobs WHERE job_id LIKE "f90ae1e3%"')
row = cursor.fetchone()
if row:
    job_id, results, options = row
    print(f'Job ID: {job_id}')
    print(f'Options: {options}')
    if results:
        try:
            result_data = json.loads(results)
            print(f'Result keys: {list(result_data.keys())}')
            if 'model_results' in result_data:
                print(f'Multi-model results found: {len(result_data["model_results"])} models')
                print(f'Models: {list(result_data["model_results"].keys())}')
            else:
                print('No multi-model results found')
        except Exception as e:
            print(f'Error parsing result: {e}')
else:
    print('Job not found')
conn.close()