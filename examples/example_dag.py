from datetime import datetime, timedelta
import logging
from textwrap import dedent

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

process_data_args = {
    'owner': 'shubhab',
    'depends_on_past': False,
    'email': ['shubha.bhaskaran@alleninstitute.org'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'stitch_data',
    default_args=process_data_args,
    description='Create overview and stitched links',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2020, 11, 1),
    catchup=False,
    tags=['image_processing'],
) as tiff2n5_dag:
    def run_convert(run_input):
        import os
        from acpreprocessing.stitching_modules.convert_to_n5 import acquisition_dir_to_n5_dir

        convert_input = {
            "input_dir": run_input["rootDir"],
            "output_dir": run_input["outputDir"],
            "max_mip": 5,
            "position_concurrency": 5
            }
        dirname = run_input["ds_name"]
        if not os.path.isdir(convert_input['output_dir']):
            acquisition_dir_to_n5_dir.AcquisitionDirToN5Dir(input_data=convert_input).run()
        else:
            print(f"Skipping conversion, {dirname} directory already exists")
    
    def run_create_overview(run_input):
        from acpreprocessing.utils.nglink import create_overview
        from acpreprocessing.utils.metadata import parse_metadata

        md_input = {
                "rootDir": run_input['rootDir'],
                "fname": run_input['md_filename']
                }
        deskew = 0
        n_channels = parse_metadata.ParseMetadata(input_data=md_input).get_number_of_channels()
        n_pos = parse_metadata.ParseMetadata(input_data=md_input).get_number_of_positions()
        dirname = run_input["ds_name"]

        state = {"showDefaultAnnotations": False, "layers": []}
        overview_input = {
            "run_input": run_input
        }
        create_overview.Overview(input_data=overview_input).run(n_channels, n_pos, dirname, deskew, state=state)
    
    def run_stitch(run_input):
        from acpreprocessing.stitching_modules.stitch import create_json, stitch
        import os
        
        create_json_input = {
                'rootDir': run_input['rootDir']+"/",
                'outputDir': run_input['outputDir']+"/",
                "mip_level": run_input['mip_level'],
                "reverse": run_input['reverse_stitch'],
                "dirname": run_input["ds_name"],
                "stitch_channel": run_input['stitch_channel'],
                "stitch_json": run_input['stitch_json']
                }
        create_json.CreateJson(input_data=create_json_input).run()

        # Run Stitching with stitch.json
        stitchjsonpath = os.path.join(create_json_input['outputDir'], run_input['stitch_json'])
        stitch_input = {
                "stitchjson": stitchjsonpath
                }
        # Perform stitching if not done yet
        if not os.path.exists(os.path.join(create_json_input['outputDir'], run_input["stitch_final"])):
            stitch.Stitch(input_data=stitch_input).run()
        else:
            print("Skipped stitching - already computed")

    def run_update_state(run_input):
        from acpreprocessing.utils.nglink import update_state
        from acpreprocessing.utils.metadata import parse_metadata
        md_input = {
                "rootDir": run_input['rootDir'],
                "fname": run_input['md_filename']
                }
        n_channels = parse_metadata.ParseMetadata(input_data=md_input).get_number_of_channels()

        update_state_input = {
            'rootDir': run_input['rootDir'],
            "outputDir": run_input['outputDir'],
            'mip_level': run_input['mip_level'],
            "fname": "stitched-nglink.txt",
            "consolidate_pos": run_input['consolidate_pos'],
            "n_channels": n_channels,
            "state_json": run_input["state_json"],
            "stitch_final": run_input["stitch_final"],
            "stitched_nglink": run_input["stitched_nglink"],
            "stitched_state": run_input["stitched_state"]
        }
        update_state.UpdateState(input_data=update_state_input).run()
    
    task_convert = PythonOperator(
        task_id='convert',
        python_callable=run_convert,
        op_kwargs={"run_input": '{{ dag_run.conf }}'}
    )

    task_create_overview = PythonOperator(
        task_id='create_overview',
        python_callable=run_create_overview,
        op_kwargs={"run_input": '{{ dag_run.conf }}'}
    )
    
    task_stitch = PythonOperator(
        task_id='stitch',
        python_callable=run_stitch,
        op_kwargs={"run_input": '{{ dag_run.conf}}'}
    )
    
    task_update_state = PythonOperator(
        task_id='update_state',
        python_callable=run_update_state,
        op_kwargs={"run_input": '{{ dag_run.conf}}'}
    )

    task_convert >> task_create_overview
    task_create_overview >> task_stitch
    task_stitch >> task_update_state