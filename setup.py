from setuptools import setup

setup(
    name='dataset-merger',
    version='1.0',
    py_modules=['merge_datasets'], # The name of your python file without .py
    install_requires=[
        'opencv-python',
        'numpy',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'merge-ds=merge_datasets:main', # Command=FileName:FunctionName
        ],
    },
)