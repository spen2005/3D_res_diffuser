from setuptools import setup, find_packages

setup(
    name="resrl",
    version="0.1.0",
    packages=[
        "RL",
        "flower_vla_calvin"
    ],
    package_dir={
        "RL": "RL",
        "flower_vla_calvin": "flower_vla_calvin"
    },
)