from setuptools import setup

setup(
    name='XGBoost-Ranking',
    version='0.6',
    packages=['xgboostextension'],
    author='bigdong89',
    author_email='dongjiaquan89@gmail.com',
    maintainer='FlorisHoogenboom',
    url='https://github.com/bigdong89/xgboostExtension',
    license='Apache 2.0',
    description='XGBoost Extension for Easy Ranking & TreeFeature.',
    install_requires=[
        'xgboost'
    ],
    setup_requires=[]
)
