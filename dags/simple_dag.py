from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'simple_github_dag',  # The name of the DAG
    default_args=default_args,
    description='A simple DAG synced from GitHub',
    schedule_interval='@daily',  # Run the DAG every day
)

# Define tasks using BashOperator
task_1 = BashOperator(
    task_id='print_hello',
    bash_command='echo Hello from GitHub Synced DAG!',
    dag=dag,
)

task_2 = BashOperator(
    task_id='print_goodbye',
    bash_command='echo Goodbye from GitHub Synced DAG!',
    dag=dag,
)

# Set task dependencies
task_1 >> task_2
